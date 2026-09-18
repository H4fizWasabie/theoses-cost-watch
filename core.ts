export type DataHandling = "zdr" | "cache_only" | "trains" | "unknown";

// Prompt-cache reliability cannot be read from OpenRouter's own pricing data: every backend
// for a given model reports the same cache_read/prompt ratio (confirmed live against
// /api/v1/models/.../endpoints for z-ai/glm-5.3-flash — DeepInfra, Morph, GMICloud, etc. all
// show the identical ~0.2x rate), which is OpenRouter's standard billing convention, not a
// signal that a backend's own KV-cache/prefix-caching actually delivers hits. The only way to
// tell providers apart here is by observing real generation logs over time (theoses2's own
// dashboard showed GMICloud landing occasional cache credits for GLM 5.3 Flash while Morph
// never did, across many consecutive calls). Rather than guess or add unproven telemetry,
// providers a human has actually confirmed don't cache well for a given model go in this
// explicit, per-model exclude list (cost-watch.json's "unreliable_cache" key) — the same
// "explicit config change required" philosophy already used for dataHandling and the
// quantization floor below.

export interface CatalogueEntry {
  model: string;
  provider: string;
  input: number;
  output: number;
  cache: number;
  // cache_read / prompt price ratio. Higher is better at equal cache price:
  // the same cache price measured against a cheaper input. Uniform across
  // almost all OpenRouter backends (~0.2, OpenRouter's billing convention),
  // so it only discriminates in the rare cases that deviate (CoreWeave 0.333,
  // Makora 0.171 for GLM as of 2026-09-13).
  cacheRate?: number;
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

// Quantizations abah accepts as first-class (2026-09-13 plan: fp8 as the floor for
// first-class pins; anything explicitly listed here ranks tier 0). "unknown"/missing
// quantization is NOT here — it's handled by the per-model quantization policy below.
const fp8OrBetter = new Set(["fp8", "fp16", "bf16", "fp32"]);

// "require": only fp8+ endpoints are eligible as primary pins; everything else
//   (unknown/missing quantization) may only appear as trailing fallback pins.
// "prefer": fp8+ endpoints rank ahead of unknown-quantization ones, but unknown
//   is still eligible for primary pins (pre-2026-09-13 behaviour, kept for
//   models like DeepSeek where the official endpoint doesn't report quantization).
// "any": no quantization gate beyond the fp4/int4 precision floor above.
export type QuantPolicy = "require" | "prefer" | "any";

function isQuantUnknown(quantization: string | undefined): boolean {
  return !quantization || !fp8OrBetter.has(quantization.toLowerCase());
}

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
    const input = numberValue(pricing.prompt);
    const cache = numberValue(pricing.input_cache_read);
    return [{
      model,
      provider,
      input,
      output: numberValue(pricing.completion),
      cache,
      cacheRate: input > 0 ? cache / input : undefined,
      discount: typeof pricing.discount === "number" ? pricing.discount : 0,
      dataHandling: dataHandling[provider] ?? "unknown",
      quantization: typeof endpoint.quantization === "string" ? endpoint.quantization : undefined,
      status: metricValue(endpoint.status),
      uptime: metricValue(endpoint.uptime_last_30m),
      latency: metricValue(endpoint.latency_last_30m?.p50),
    }];
  });
}

// Lexicographic comparator (abah's 2026-09-13 spec): cache price asc -> cache
// rate desc -> input asc -> output asc -> uptime bucket desc -> latency asc ->
// provider name asc. Replaces the old min-normalized score, whose denominators
// (set by even providers that were filtered out of the ranking) rescaled the
// whole board on every price change, and whose uptime/latency tie-breaks on raw
// 30-minute metrics reordered near-tied groups every refresh — the two churn
// sources that made the pin list jump between providers.
export function rankCatalogueEntries(entries: CatalogueEntry[]): CatalogueEntry[] {
  // Free/zero cache reads are the BEST case, not incomplete — the old
  // cache > 0 requirement demoted DeepSeek's official endpoint (cache 0.00)
  // below paid-cache providers.
  const complete = entries.filter((entry) => entry.input > 0 && entry.output > 0);
  const incomplete = entries.filter((entry) => !complete.includes(entry));
  if (complete.length === 0) return entries;

  // Bucketed uptime instead of raw values: raw 30-minute metrics (99.68 vs
  // 99.75) churned tie groups on every refresh.
  const uptimeBucket = (entry: CatalogueEntry) =>
    entry.uptime === undefined ? -1 : entry.uptime >= 99.9 ? 3 : entry.uptime >= 99 ? 2 : entry.uptime >= 95 ? 1 : 0;
  const latency = (entry: CatalogueEntry) => entry.latency ?? Infinity;

  return [...complete].sort((a, b) =>
    a.cache - b.cache
    || (b.cacheRate ?? 0) - (a.cacheRate ?? 0)
    || a.input - b.input
    || a.output - b.output
    || uptimeBucket(b) - uptimeBucket(a)
    || latency(a) - latency(b)
    || a.provider.localeCompare(b.provider)
  ).concat(incomplete);
}

export function chooseProviderOrder(
  entries: CatalogueEntry[],
  maxPins = 3,
  unreliableCacheProviders: ReadonlySet<string> = new Set(),
  quantPolicy: QuantPolicy = "prefer",
  fallbackPins = 2,
  preferredProviders: ReadonlySet<string> = new Set(),
): string[] {
  // Precise cheapest-cache ranker. "Cheapest first" here means the primary pin
  // list is stable and cost-tiered: pins 2..N are only reached on failure, so a
  // tighter maxPins limits KV-cache fragmentation without losing redundancy.
  const eligible = entries.filter((entry) =>
    entry.dataHandling !== "trains"
    && !unreliableCacheProviders.has(entry.provider)
    && (entry.status === undefined || entry.status === 0)
    && !precisionWorseThanFP8.has(entry.quantization?.toLowerCase() ?? ""));
  const verified = eligible.filter((entry) => !isQuantUnknown(entry.quantization));
  const unverified = eligible.filter((entry) => isQuantUnknown(entry.quantization));

  // Explicit human pins (abah, 2026-09-13): providers named in preferredProviders
  // rank TIER 0 — ahead of the cost ranking — but only if they clear the SAME
  // eligibility gate above (fp4 precision floor, data-handling, status). An
  // explicit pin expresses a judgement the ranker cannot see: OpenRouter publishes
  // per-provider throughput on its model PAGES but not in the endpoints API
  // payload (throughput_last_30m is null for every GLM 5.3 Flash endpoint), so
  // cache price is the only machine-readable key and it cannot express
  // "this provider is ~2x faster". See cost-watch.json's preferred_providers.
  const forced = rankCatalogueEntries(eligible.filter((entry) => preferredProviders.has(entry.provider)))
    .map((entry) => entry.provider)
    .slice(0, maxPins);
  const restSlots = Math.max(0, maxPins - forced.length);
  const notForced = (provider: string) => !forced.includes(provider);

  if (quantPolicy === "any") {
    const rest = rankCatalogueEntries(eligible).map((e) => e.provider).filter(notForced).slice(0, restSlots);
    return [...forced, ...rest];
  }

  const primary = quantPolicy === "require" ? verified : verified.concat(unverified);
  const pinned = rankCatalogueEntries(primary).map((e) => e.provider).filter(notForced).slice(0, restSlots);

  // "require": unknown-quantization providers (e.g. Relace/Wafer/Makora for GLM)
  // are never primary pins, but may appear as trailing fallbacks — reached only
  // when every fp8 pin fails, so fp8 discipline is kept without hard-failing if
  // all fp8 endpoints go down at once.
  const fallback = quantPolicy === "require"
    ? rankCatalogueEntries(unverified).map((e) => e.provider).filter((p) => notForced(p) && !pinned.includes(p)).slice(0, fallbackPins)
    : [];
  return [...forced, ...pinned, ...fallback];
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

// Status codes that describe a problem with THIS specific request (malformed body, payload
// too large, content rejected by moderation) rather than the serving provider's health.
// `after_provider_response` only carries `status`/`headers` — no response body — so a status
// code is the only signal available to tell these apart; counting a request-shape failure
// against a provider would eventually demote a perfectly healthy, cheap provider purely
// because one call happened to be malformed on that pin, the same "hoisted a bad signal into
// the ranking" failure class the fp8/Relace-Wafer incident was about. Deliberately narrow:
// only codes that are unambiguously about the request, not about capacity or availability —
// 429 (rate limit) and every 5xx (including OpenRouter's own 529 "overloaded") stay in scope,
// since those genuinely do indicate the provider is the problem.
const REQUEST_SHAPE_STATUS_CODES = new Set([400, 413, 422]);

/** Whether a failed response's status code should count against the serving provider's sticky
 * reliability score (recordOutcome in index.ts), vs. being a request-shape problem that would
 * fail identically on any provider. */
export function isProviderHealthFailure(status: number): boolean {
  return status >= 400 && !REQUEST_SHAPE_STATUS_CODES.has(status);
}
