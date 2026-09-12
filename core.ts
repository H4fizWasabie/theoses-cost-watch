export type DataHandling = "zdr" | "cache_only" | "trains" | "unknown";

export interface CatalogueEntry {
  model: string;
  provider: string;
  input: number;
  output: number;
  cache: number;
  discount: number;
  dataHandling: DataHandling;
  quantization?: string;
  status?: number;
  uptime?: number;
  latency?: number;
}

export interface EndpointPayload {
  data?: {
    endpoints?: Array<{
      name?: unknown;
      pricing?: {
        prompt?: unknown;
        completion?: unknown;
        input_cache_read?: unknown;
        discount?: unknown;
      };
      quantization?: unknown;
      latency_last_30m?: { p50?: unknown };
      uptime_last_30m?: unknown;
      status?: unknown;
    }>;
  };
}

// Quantizations known to be worse than fp8 (mirrors mino's cost-watch, added
// after issue #495: an fp4 endpoint outranked fp8 alternatives on price alone
// and was implicated in a live GLM decode-collapse incident). This is a
// floor, not a ladder — anything not in this set (fp8, fp16, bf16, fp32,
// "unknown", or missing) ranks in the same top tier, since there's no
// evidence those are worse.
const precisionWorseThanFP8 = new Set(["fp4", "fp2", "fp1", "int4", "int2"]);

function precisionTier(quantization: string | undefined): number {
  return quantization && precisionWorseThanFP8.has(quantization.toLowerCase()) ? 1 : 0;
}

// Relative cost difference below which two providers count as "similarly
// priced" and defer to health/uptime/latency instead of raw cost order.
const TIE_BREAK_TOLERANCE = 0.02;

const numberValue = (value: unknown): number => {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed * 1_000_000 : 0;
};

const metricValue = (value: unknown): number | undefined => {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const providerName = (name: unknown): string => {
  const value = typeof name === "string" ? name.trim() : "";
  const separator = value.indexOf(" | ");
  return separator === -1 ? value : value.slice(0, separator);
};

export function parseEndpointPayload(
  model: string,
  payload: EndpointPayload,
  dataHandling: Record<string, DataHandling> = {},
): CatalogueEntry[] {
  return (payload.data?.endpoints ?? []).flatMap((endpoint) => {
    const provider = providerName(endpoint.name);
    const pricing = endpoint.pricing;
    if (!provider || !pricing) return [];
    return [{
      model,
      provider,
      input: numberValue(pricing.prompt),
      output: numberValue(pricing.completion),
      cache: numberValue(pricing.input_cache_read),
      discount: typeof pricing.discount === "number" ? pricing.discount : 0,
      dataHandling: dataHandling[provider] ?? "unknown",
      quantization: typeof endpoint.quantization === "string" ? endpoint.quantization : undefined,
      status: metricValue(endpoint.status),
      uptime: metricValue(endpoint.uptime_last_30m),
      latency: metricValue(endpoint.latency_last_30m?.p50),
    }];
  });
}

export function rankCatalogueEntries(entries: CatalogueEntry[]): CatalogueEntry[] {
  const complete = entries.filter((entry) => entry.input > 0 && entry.cache > 0 && entry.output > 0);
  const incomplete = entries.filter((entry) => !complete.includes(entry));
  if (complete.length === 0) return entries;

  const minimum = (field: "input" | "cache" | "output") => Math.min(...complete.map((entry) => entry[field]));
  const minInput = minimum("input");
  const minCache = minimum("cache");
  const minOutput = minimum("output");
  return [...complete].sort((a, b) => {
    const score = (entry: CatalogueEntry) => entry.input / minInput + entry.cache / minCache + entry.output / minOutput;
    const health = (entry: CatalogueEntry) => entry.status === undefined ? 0 : entry.status === 0 ? 1 : -1;
    const uptime = (entry: CatalogueEntry) => entry.uptime ?? -Infinity;
    const latency = (entry: CatalogueEntry) => entry.latency ?? Infinity;

    const tierDiff = precisionTier(a.quantization) - precisionTier(b.quantization);
    if (tierDiff !== 0) return tierDiff;

    const scoreA = score(a);
    const scoreB = score(b);
    const relativeDiff = Math.abs(scoreA - scoreB) / Math.max(scoreA, scoreB);
    if (relativeDiff > TIE_BREAK_TOLERANCE) return scoreA - scoreB;

    return health(b) - health(a) || uptime(b) - uptime(a)
      || latency(a) - latency(b) || a.cache - b.cache || a.output - b.output || a.input - b.input;
  }).concat(incomplete);
}

export function chooseProviderOrder(
  entries: CatalogueEntry[],
  maxPins = 5,
): string[] {
  return rankCatalogueEntries(entries
    .filter((entry) => entry.dataHandling !== "trains"))
    .slice(0, maxPins)
    .map((entry) => entry.provider);
}

/**
 * Applies ranked provider orders to theoses's models.json, which keys routing
 * overrides as providers.openrouter.modelOverrides[modelId].compat.openRouterRouting
 * (see packages/coding-agent/src/core/model-config.ts and provider-composer.ts
 * in theoses2 — a different shape than the old providers.openrouter.models[]
 * array this extension originally targeted).
 */
export function applyModelOverrideOrders(
  document: Record<string, unknown>,
  orders: Record<string, string[]>,
): { document: Record<string, unknown>; changed: boolean } {
  const next = structuredClone(document) as Record<string, any>;
  next.providers = next.providers && typeof next.providers === "object" ? next.providers : {};
  next.providers.openrouter = next.providers.openrouter && typeof next.providers.openrouter === "object"
    ? next.providers.openrouter
    : {};
  next.providers.openrouter.modelOverrides = next.providers.openrouter.modelOverrides
    && typeof next.providers.openrouter.modelOverrides === "object"
    ? next.providers.openrouter.modelOverrides
    : {};
  const modelOverrides = next.providers.openrouter.modelOverrides as Record<string, any>;
  let changed = false;

  for (const [modelId, order] of Object.entries(orders)) {
    if (!order.length) continue;
    const existing = modelOverrides[modelId] && typeof modelOverrides[modelId] === "object" ? modelOverrides[modelId] : {};
    const compat = existing.compat && typeof existing.compat === "object" ? existing.compat : {};
    const routing = compat.openRouterRouting && typeof compat.openRouterRouting === "object" ? compat.openRouterRouting : {};
    if (JSON.stringify(routing.order) === JSON.stringify(order)) continue;
    modelOverrides[modelId] = {
      ...existing,
      compat: {
        ...compat,
        // Hard whitelist: if none of the ranked/fp8+ providers are reachable, the
        // request should fail cleanly rather than silently falling through to an
        // unvetted OpenRouter default (the exact "Nexbit" incident this exists to prevent).
        openRouterRouting: { ...routing, order, allow_fallbacks: false },
      },
    };
    changed = true;
  }

  return { document: next, changed };
}

export function applyRoutingPayload(
  payload: unknown,
  order: string[],
): unknown {
  if (!order.length || !payload || typeof payload !== "object" || Array.isArray(payload)) return payload;
  const request = payload as Record<string, unknown>;
  const provider = request.provider && typeof request.provider === "object" && !Array.isArray(request.provider)
    ? request.provider as Record<string, unknown>
    : {};
  return {
    ...request,
    provider: { ...provider, order, allow_fallbacks: false },
  };
}
