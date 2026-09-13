import fs from "node:fs/promises";
const agentDir = "/home/theoses/.theoses/agent";
import("/home/theoses/.theoses/agent/extensions/theoses-cost-watch/core.ts").then(async (core) => {
  const cfg = JSON.parse(await fs.readFile(agentDir + "/cost-watch.json", "utf8"));
  const orders = {};
  for (const model of cfg.models) {
    const res = await fetch(`https://openrouter.ai/api/v1/models/${model}/endpoints`);
    const entries = core.parseEndpointPayload(model, await res.json(), cfg.data_handling);
    const policy = cfg.quantization_policy?.[model] ?? cfg.quantization_policy?.default ?? "prefer";
    orders[model] = core.chooseProviderOrder(
      entries,
      cfg.max_pins ?? 3,
      new Set(cfg.unreliable_cache?.[model] ?? []),
      policy,
      cfg.fallback_pins ?? 2,
      new Set(cfg.preferred_providers?.[model] ?? []),
    );
  }
  const doc = JSON.parse(await fs.readFile(agentDir + "/models.json", "utf8"));
  const update = core.applyModelOverrideOrders(doc, orders);
  if (update.changed) {
    await fs.writeFile(agentDir + "/models.json", JSON.stringify(update.document, null, 2) + "\n");
    console.log("models.json updated");
  } else console.log("models.json already matches");
  const state = { refreshedAt: new Date().toISOString(), orders, errors: {} };
  await fs.writeFile(agentDir + "/cost-watch-state.json", JSON.stringify(state, null, 2) + "\n");
  console.log(JSON.stringify(orders, null, 2));
});
