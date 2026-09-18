import { appendFileSync } from "node:fs";
import { readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { Type } from "typebox";
import type { ExtensionAPI } from "theoses-coding-agent";
import {
  applyModelOverrideOrders,
  applyRoutingPayload,
  chooseProviderOrder,
  isProviderHealthFailure,
  parseEndpointPayload,
  type DataHandling,
  type EndpointPayload,
  type QuantPolicy,
} from "./core.ts";

const agentDir = process.env.THEOSES_CODING_AGENT_DIR || join(homedir(), ".theoses", "agent");
const modelsPath = join(agentDir, "models.json");
const configPath = join(agentDir, "cost-watch.json");
const statePath = join(agentDir, "cost-watch-state.json");
const defaultRefreshMinutes = 60;

// Load diagnostics: proves whether the agent host actually imports this module and runs
// the factory. Journal capture of extension console output has proven unreliable, so we
// write a marker file instead.
const markerPath = join(agentDir, "cost-watch-load.log");
function mark(msg: string): void {
  try {
    appendFileSync(markerPath, `${new Date().toISOString()} [pid=${process.pid}] ${msg}\n`);
  } catch {
    // Diagnostics must never break the extension.
  }
}
mark("module imported");

interface WatchConfig {
  catalogue_refresh_minutes?: number;
  data_handling?: Record<string, DataHandling>;
  models?: string[];
  // Keyed by "default" and/or model id (model id wins). See core.ts QuantPolicy:
  // "require" = fp8+ primary pins only (unknown-quant providers demoted to trailing
  // fallbacks), "prefer" = fp8+ ranked ahead but unknown still eligible, "any".
  quantization_policy?: Record<string, QuantPolicy>;
  max_pins?: number;
  fallback_pins?: number;
  // Per-model list of provider names a human has confirmed don't cache reliably for that
  // model (see core.ts's comment on DataHandling for why this can't be inferred from pricing).
  unreliable_cache?: Record<string, string[]>;
  // Per-model list of provider names to pin ahead of the cost ranking (tier 0). See
  // core.ts's preferredProviders comment: OpenRouter doesn't expose per-provider
  // throughput in the endpoints API payload, so a speed preference has to be stated
  // by hand rather than inferred from cache price.
  preferred_providers?: Record<string, string[]>;
  // Per-model fixed provider order that bypasses cost ranking entirely (2026-09-13,
  // abah's GLM 5.3 Flash decision: Relace/StreamLake/Parasail chosen by hand after
  // reviewing real cache-hit and throughput data, not by chooseProviderOrder's price
  // ranking). A model listed here skips fetching/ranking OpenRouter's endpoint catalogue
  // altogether — the array is sent as-is. Sticky failover (before_provider_request /
  // recordOutcome) still runs on top of this fixed list, so it can only ever rotate
  // among these named providers: allow_fallbacks stays false, so OpenRouter never serves
  // a provider outside this array, which keeps servedIdx = base.indexOf(served) inside
  // the fixed list too.
  manual_provider_order?: Record<string, string[]>;
  // Sticky provider (2026-09-13): OpenRouter flaps between pins on any top-pin hiccup
  // (it applies its own short cooldowns), so each request can land on a different
  // provider. The extension now observes who ACTUALLY serves each request — OpenRouter
  // echoes an x-generation-id response header, and GET /api/v1/generation?id=... names
  // the serving provider — and only demotes the ranked primary after repeated observed
  // failures, probing back after a TTL. These knobs tune that.
  sticky_demote_threshold?: number; // consecutive observed failed primaries before demotion (default 2)
  sticky_probe_minutes?: number; // minutes on a demoted pin before probing the ranked primary (default 5)
  sticky_price_hysteresis_pct?: number; // refresh-time: keep incumbent pin 1 unless new pin is this % cheaper on cache price (default 5)
}

interface WatchState {
  refreshedAt?: string;
  orders?: Record<string, string[]>;
  errors?: Record<string, string>;
}

async function readJson<T>(path: string, fallback: T): Promise<T> {
  try {
    return JSON.parse(await readFile(path, "utf8")) as T;
  } catch {
    return fallback;
  }
}

async function atomicJsonWrite(path: string, value: unknown): Promise<void> {
  const data = `${JSON.stringify(value, null, 2)}\n`;
  const tempPath = `${path}.tmp-${process.pid}`;
  try {
    await writeFile(`${path}.bak-cost-watch`, await readFile(path));
  } catch {
    // The first write has no previous file to back up.
  }
  await writeFile(tempPath, data, "utf8");
  await rename(tempPath, path);
}

// Models to watch come from cost-watch.json's explicit "models" list (the
// brief's rule: adding a model requires an explicit config change). If that's
// missing, fall back to whatever OpenRouter models already have a routing
// override in models.json, so an existing manual pin doesn't go unmanaged.
function configuredModels(config: WatchConfig, document: Record<string, unknown>): string[] {
  if (config.models?.length) return config.models;
  const providers = document.providers as Record<string, unknown> | undefined;
  const openrouter = providers?.openrouter as { modelOverrides?: Record<string, unknown> } | undefined;
  return Object.keys(openrouter?.modelOverrides ?? {});
}

// The initial load-time refresh and each before_provider_request-triggered
// refresh can overlap; without this guard, two concurrent atomicJsonWrite
// calls to the same path share a temp filename (path.tmp-<pid>) and race on
// rename, so only one refresh runs at a time and the rest await its result.
// --- Sticky provider runtime (in-memory only: a restart deliberately resumes the
// ranked primary, which is the correct resting state) ------------------------------

interface StickyEntry {
  index: number; // position in the refreshed order currently leading
  since: number; // Date.now() when the demotion happened (drives probe-back)
  failedPrimaries: number; // consecutive requests where the leading pin did NOT serve
}
const sticky = new Map<string, StickyEntry>();
// What we actually sent for the most recent request per model (gen-endpoint attribution
// compares the serving provider against the primary we led with).
const lastBase = new Map<string, string[]>();
const lastPrimary = new Map<string, string>();
// x-generation-id captured at after_provider_response time (headers arrive as soon as the
// stream starts, well before a stream-duration timeout could ever fire) — kept so a later
// message_end stream-timeout can still identify which provider was actually serving the
// request that never finished. See the message_end handler below.
const lastGenId = new Map<string, string>();
const inFlightGenPolls = new Set<string>();

const DEFAULT_DEMOTE_THRESHOLD = 2;
const DEFAULT_PROBE_MINUTES = 5;
const DEFAULT_PRICE_HYSTERESIS_PCT = 5;
const GEN_POLL_ATTEMPTS = 5;
const GEN_POLL_DELAY_MS = 2000;

function headerValue(headers: unknown, name: string): string | undefined {
  if (!headers) return undefined;
  if (typeof (headers as Headers).get === "function") return (headers as Headers).get(name) ?? undefined;
  const record = headers as Record<string, unknown>;
  const key = Object.keys(record).find((k) => k.toLowerCase() === name.toLowerCase());
  const value = key ? record[key] : undefined;
  return typeof value === "string" ? value : undefined;
}

async function openRouterKey(): Promise<string | undefined> {
  if (process.env.OPENROUTER_API_KEY) return process.env.OPENROUTER_API_KEY;
  try {
    const env = await readFile(join(agentDir, "theoses.env"), "utf8");
    const line = env.split("\n").map((l) => l.trim()).find((l) => l.startsWith("OPENROUTER_API_KEY="));
    return line?.split("=").slice(1).join("=").replace(/^["']|["']$/g, "") || undefined;
  } catch {
    return undefined;
  }
}

// Demote/promote decision from an observed outcome. `served` is the provider that
// actually handled the request ("" = request failed outright, i.e. all pins failed).
async function recordOutcome(modelId: string, base: string[], primary: string, served: string): Promise<void> {
  const config = await readJson<WatchConfig>(configPath, {});
  const entry = sticky.get(modelId) ?? { index: 0, since: 0, failedPrimaries: 0 };
  if (served && served === primary) {
    entry.failedPrimaries = 0;
  } else {
    entry.failedPrimaries += 1;
    const threshold = config.sticky_demote_threshold ?? DEFAULT_DEMOTE_THRESHOLD;
    if (entry.failedPrimaries >= threshold) {
      // Lead with the provider that proved itself; if nothing served, just step down.
      const servedIdx = base.indexOf(served);
      const nextIdx = servedIdx >= 0 ? servedIdx : Math.min(entry.index + 1, base.length - 1);
      if (nextIdx !== entry.index) {
        entry.index = nextIdx;
        entry.since = Date.now();
      }
      entry.failedPrimaries = 0;
    }
  }
  sticky.set(modelId, entry);
}

/**
 * Resolves which provider actually served a generation, via OpenRouter's generation-lookup
 * endpoint. Returns undefined if the generation never gets indexed within the retry budget, the
 * lookup fails, or (guarding against cross-model mis-attribution) the indexed model doesn't
 * match. Shared by the success-path attribution below and the stream-timeout handler, which both
 * need the same "who actually served this" answer for a genId captured earlier.
 */
async function resolveServedProvider(genId: string, modelId: string): Promise<string | undefined> {
  for (let attempt = 0; attempt < GEN_POLL_ATTEMPTS; attempt++) {
    await new Promise((r) => setTimeout(r, GEN_POLL_DELAY_MS));
    try {
      const key = await openRouterKey();
      if (!key) return undefined;
      const res = await fetch(`https://openrouter.ai/api/v1/generation?id=${encodeURIComponent(genId)}`, {
        headers: { Authorization: `Bearer ${key}` },
      });
      if (res.status === 404) continue; // not indexed yet
      if (!res.ok) return undefined;
      const data = ((await res.json()) as { data?: { provider_name?: string; model?: string } }).data;
      const served = data?.provider_name;
      if (!served) return undefined;
      if (data?.model && !data.model.startsWith(modelId)) return undefined;
      return served;
    } catch {
      // transient network error — next attempt
    }
  }
  return undefined;
}

/** Fire-and-forget attribution for a normal (status < 400) response: never blocks or fails the
 * request itself. */
function pollServedProvider(genId: string, modelId: string, base: string[], primary: string): void {
  void resolveServedProvider(genId, modelId).then((served) => {
    if (served) void recordOutcome(modelId, base, primary, served);
  });
}

let inFlightRefresh: Promise<WatchState> | undefined;

function refresh(force = false): Promise<WatchState> {
  if (inFlightRefresh) return inFlightRefresh;
  inFlightRefresh = performRefresh(force).finally(() => {
    inFlightRefresh = undefined;
  });
  return inFlightRefresh;
}

async function performRefresh(force: boolean): Promise<WatchState> {
  const config = await readJson<WatchConfig>(configPath, {});
  const current = await readJson<WatchState>(statePath, {});
  const refreshMinutes = config.catalogue_refresh_minutes || defaultRefreshMinutes;
  if (!force && current.refreshedAt && Date.now() - Date.parse(current.refreshedAt) < refreshMinutes * 60_000) {
    return current;
  }

  const document = await readJson<Record<string, unknown>>(modelsPath, {});
  const configured = new Set(configuredModels(config, document));
  const orders: Record<string, string[]> = {};
  const errors: Record<string, string> = {};

  for (const model of configured) {
    try {
      const manualOrder = config.manual_provider_order?.[model];
      if (manualOrder?.length) {
        orders[model] = manualOrder;
        continue;
      }
      const response = await fetch(`https://openrouter.ai/api/v1/models/${model}/endpoints`);
      if (!response.ok) throw new Error(`OpenRouter returned HTTP ${response.status}`);
      const entries = parseEndpointPayload(model, (await response.json()) as EndpointPayload, config.data_handling);
      const unreliableCacheProviders = new Set(config.unreliable_cache?.[model] ?? []);
      const preferredProviders = new Set(config.preferred_providers?.[model] ?? []);
      const quantPolicy = config.quantization_policy?.[model] ?? config.quantization_policy?.default ?? "prefer";
      const order = chooseProviderOrder(
        entries,
        config.max_pins ?? 3,
        unreliableCacheProviders,
        quantPolicy,
        config.fallback_pins ?? 2,
        preferredProviders,
      );
      if (!order.length) throw new Error("No eligible complete provider pricing found");
      // Sticky refresh hysteresis: price jitter on 30-minute metrics shouldn't swap pin 1
      // back and forth. If the incumbent top pin is still ranked and the new leader isn't
      // meaningfully cheaper on cache price (the ranker's primary key), keep the incumbent.
      // SAFETY (2026-09-13: Relace/Wafer incident): the incumbent may only be hoisted if it
      // sits in the PINNED (fp8-eligible) portion of the fresh ranking. OpenRouter's
      // quantization metadata drifts — an incumbent that became a primary while mislabeled
      // fp8 drops into the fallback section when the label reverts, and hoisting it then
      // would perpetuate unknown-quant providers as primary pins (exactly the fp8-floor
      // violation this policy exists to prevent).
      const oldTop = current.orders?.[model]?.[0];
      if (oldTop && order[0] !== oldTop && order.includes(oldTop)) {
        const hysteresisPct = config.sticky_price_hysteresis_pct ?? DEFAULT_PRICE_HYSTERESIS_PCT;
        const fallbackPins = quantPolicy === "require" ? (config.fallback_pins ?? 2) : 0;
        const pinnedCount = quantPolicy === "require" ? Math.max(0, order.length - fallbackPins) : order.length;
        const incumbentIdx = order.indexOf(oldTop);
        if (incumbentIdx < pinnedCount) {
          const newTopEntry = entries.find((e) => e.provider === order[0]);
          const oldTopEntry = entries.find((e) => e.provider === oldTop);
          const meaningfullyCheaper = newTopEntry && oldTopEntry
            && newTopEntry.cache < oldTopEntry.cache * (1 - hysteresisPct / 100);
          if (!meaningfullyCheaper) order = [oldTop, ...order.filter((p) => p !== oldTop)];
        }
      }
      orders[model] = order;
    } catch (error) {
      errors[model] = error instanceof Error ? error.message : String(error);
    }
  }

  const update = applyModelOverrideOrders(document, orders);
  if (update.changed) await atomicJsonWrite(modelsPath, update.document);
  const next = { refreshedAt: new Date().toISOString(), orders, errors };
  await atomicJsonWrite(statePath, next);
  console.error(`[cost-watch] refresh written: orders=[${Object.entries(orders).map(([m, o]) => `${m}=[${o.join(",")}]`).join(" ")}] errors=[${Object.entries(errors).map(([m, e]) => `${m}: ${e}`).join("; ") || "none"}]`);
  return next;
}

function formatState(state: WatchState): string {
  const orders = Object.entries(state.orders ?? {})
    .map(([model, providers]) => {
      const entry = sticky.get(model);
      const active = entry && entry.index > 0 && providers[entry.index]
        ? ` [sticky: ${providers[entry.index]}]`
        : "";
      return `${model}: ${providers.join(" -> ")}${active}`;
    })
    .join("\n");
  const errors = Object.entries(state.errors ?? {})
    .map(([model, error]) => `${model}: ${error}`)
    .join("\n");
  return [`last refresh: ${state.refreshedAt ?? "never"}`, orders, errors ? `errors:\n${errors}` : ""]
    .filter(Boolean)
    .join("\n");
}

export default function theosesCostWatch(theoses: ExtensionAPI) {
  const runtime = globalThis as typeof globalThis & {
    __theosesCostWatchTimer?: ReturnType<typeof setInterval>;
  };
  console.error("[cost-watch] factory invoked");
  mark("factory invoked");
  void refresh()
    .then((s) => mark(`startup refresh ok: refreshedAt=${s.refreshedAt ?? "?"}`))
    .catch((e) => mark(`startup refresh FAILED: ${e instanceof Error ? e.message : String(e)}`));
  if (!runtime.__theosesCostWatchTimer) {
    const timer = setInterval(() => {
      console.error("[cost-watch] hourly force-refresh firing");
      void refresh(true).catch((e) => console.error("[cost-watch] timer refresh FAILED:", e));
    }, defaultRefreshMinutes * 60_000);
    timer.unref?.();
    runtime.__theosesCostWatchTimer = timer;
  }

  theoses.registerTool({
    name: "cost_watch_status",
    label: "Cost Watch Status",
    description: "Show the latest OpenRouter provider ranking used by theoses.",
    parameters: Type.Object({}),
    async execute() {
      const state = await readJson<WatchState>(statePath, {});
      return { content: [{ type: "text", text: formatState(state) }], details: state };
    },
  });

  theoses.registerTool({
    name: "cost_watch_refresh",
    label: "Refresh Cost Watch",
    description: "Refresh OpenRouter pricing and update theoses's provider routing.",
    parameters: Type.Object({}),
    async execute() {
      const state = await refresh(true);
      return { content: [{ type: "text", text: formatState(state) }], details: state };
    },
  });

  theoses.on("before_provider_request", (event, ctx) => {
    const model = ctx.model;
    console.error(`[cost-watch] before_provider_request: model=${model ? `${model.provider}/${model.id}` : "UNDEFINED"}`);
    if (!model || model.provider !== "openrouter" || !event.payload || typeof event.payload !== "object") return;
    return refresh().then(async (state) => {
      const config = await readJson<WatchConfig>(configPath, {});
      const order = state.orders?.[model.id];
      if (!order?.length) return;
      // A refresh that changed the ranking invalidates index-based sticky state.
      const prevBase = lastBase.get(model.id);
      if (prevBase && JSON.stringify(prevBase) !== JSON.stringify(order)) sticky.delete(model.id);
      lastBase.set(model.id, order);

      let index = sticky.get(model.id)?.index ?? 0;
      const stickyEntry = sticky.get(model.id);
      // Probe-back: after the TTL on a demoted pin, give the ranked primary another
      // chance. Worst case is one wasted request (OR serves the next pin in-request).
      if (index > 0 && stickyEntry) {
        const probeMs = (config.sticky_probe_minutes ?? DEFAULT_PROBE_MINUTES) * 60_000;
        if (Date.now() - stickyEntry.since > probeMs) {
          sticky.delete(model.id);
          index = 0;
        }
      }
      if (index >= order.length) {
        sticky.delete(model.id);
        index = 0;
      }
      const effective = index === 0
        ? order
        : [order[index], ...order.slice(0, index), ...order.slice(index + 1)];
      lastPrimary.set(model.id, effective[0]);
      console.error(`[cost-watch] routing ${model.id}: stickyIndex=${index} order=[${effective.join(" -> ")}]`);
      return applyRoutingPayload(event.payload, effective);
    });
  });

  theoses.on("after_provider_response", (event, ctx) => {
    const model = ctx.model;
    console.error(`[cost-watch] after_provider_response: model=${model ? model.id : "UNDEFINED"} status=${event.status}`);
    if (!model || model.provider !== "openrouter") return;
    const base = lastBase.get(model.id);
    const primary = lastPrimary.get(model.id);
    if (!base?.length || !primary) return;
    if (event.status >= 400) {
      // Only count status codes that actually indicate a provider-side problem (see core.ts's
      // isProviderHealthFailure) — a request-shape failure (malformed body, oversized payload,
      // moderation rejection) would fail identically on every provider and shouldn't push the
      // sticky ranking away from a perfectly healthy pin.
      if (isProviderHealthFailure(event.status)) void recordOutcome(model.id, base, primary, "");
      return;
    }
    const genId = headerValue(event.headers, "x-generation-id");
    if (!genId) return;
    lastGenId.set(model.id, genId);
    if (inFlightGenPolls.has(genId)) return;
    inFlightGenPolls.add(genId);
    try {
      pollServedProvider(genId, model.id, base, primary);
    } finally {
      setTimeout(() => inFlightGenPolls.delete(genId), 30_000).unref?.();
    }
  });

  /**
   * Catches theoses2's stream-duration watchdog failure ("Stream exceeded the Ns max
   * duration..." — packages/ai/src/api/openai-completions.ts) — a failure mode
   * after_provider_response structurally cannot see, because that hook fires as soon as
   * response headers arrive (the stream has already returned 200 and "succeeded" by
   * after_provider_response's own signal) — long before a stream that never finishes trickling
   * would trip the client-side timeout. Without this, a provider that reliably starts responding
   * but never finishes looks perfectly healthy to cost-watch forever, and sticky demotion never
   * triggers no matter how often it happens (2026-09-18 incident: near-exclusively hitting one
   * pinned provider for z-ai/glm-5.3-flash with no visibility into which one).
   *
   * Uses the genId captured by after_provider_response for this model to identify which
   * provider was actually serving the request that never completed, then records it as a real
   * failure (recordOutcome's demotion counter) instead of leaving it invisible.
   */
  theoses.on("message_end", (event) => {
    const message = event.message as { role?: string; provider?: string; model?: string; stopReason?: string; errorMessage?: string };
    if (message.role !== "assistant" || message.provider !== "openrouter") return;
    if (message.stopReason !== "error" || !message.errorMessage?.includes("Stream exceeded")) return;
    const modelId = message.model;
    if (!modelId) return;

    const base = lastBase.get(modelId);
    const primary = lastPrimary.get(modelId);
    const genId = lastGenId.get(modelId);

    void (async () => {
      const served = genId ? await resolveServedProvider(genId, modelId) : undefined;
      console.error(`[cost-watch] STREAM_TIMEOUT model=${modelId} servedBy=${served ?? "unknown"} genId=${genId ?? "none"}`);
      if (base?.length && primary) {
        // Always a failure regardless of who served it — the point of this event is that the
        // request never completed, independent of which provider was leading.
        await recordOutcome(modelId, base, primary, "");
      }
    })();
  });
}
