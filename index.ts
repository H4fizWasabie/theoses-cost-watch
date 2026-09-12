import { readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { Type } from "typebox";
import type { ExtensionAPI } from "theoses-coding-agent";
import {
  applyModelOverrideOrders,
  applyRoutingPayload,
  chooseProviderOrder,
  parseEndpointPayload,
  type DataHandling,
  type EndpointPayload,
} from "./core.ts";

const agentDir = process.env.THEOSES_CODING_AGENT_DIR || join(homedir(), ".theoses", "agent");
const modelsPath = join(agentDir, "models.json");
const configPath = join(agentDir, "cost-watch.json");
const statePath = join(agentDir, "cost-watch-state.json");
const defaultRefreshMinutes = 60;

interface WatchConfig {
  catalogue_refresh_minutes?: number;
  data_handling?: Record<string, DataHandling>;
  models?: string[];
  // Per-model list of provider names a human has confirmed don't cache reliably for that
  // model (see core.ts's comment on DataHandling for why this can't be inferred from pricing).
  unreliable_cache?: Record<string, string[]>;
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
      const response = await fetch(`https://openrouter.ai/api/v1/models/${model}/endpoints`);
      if (!response.ok) throw new Error(`OpenRouter returned HTTP ${response.status}`);
      const entries = parseEndpointPayload(model, (await response.json()) as EndpointPayload, config.data_handling);
      const unreliableCacheProviders = new Set(config.unreliable_cache?.[model] ?? []);
      const order = chooseProviderOrder(entries, 5, unreliableCacheProviders);
      if (!order.length) throw new Error("No eligible complete provider pricing found");
      orders[model] = order;
    } catch (error) {
      errors[model] = error instanceof Error ? error.message : String(error);
    }
  }

  const update = applyModelOverrideOrders(document, orders);
  if (update.changed) await atomicJsonWrite(modelsPath, update.document);
  const next = { refreshedAt: new Date().toISOString(), orders, errors };
  await atomicJsonWrite(statePath, next);
  return next;
}

function formatState(state: WatchState): string {
  const orders = Object.entries(state.orders ?? {})
    .map(([model, providers]) => `${model}: ${providers.join(" -> ")}`)
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
  void refresh().catch(() => undefined);
  if (!runtime.__theosesCostWatchTimer) {
    const timer = setInterval(() => void refresh(true).catch(() => undefined), defaultRefreshMinutes * 60_000);
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
    if (!model || model.provider !== "openrouter" || !event.payload || typeof event.payload !== "object") return;
    return refresh().then((state) => {
      const order = state.orders?.[model.id];
      if (!order?.length) return;
      return applyRoutingPayload(event.payload, order);
    });
  });
}
