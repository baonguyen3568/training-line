// Minimal CDP driver: node shot.mjs <url> <outdir> ; steps via JS strings in STEPS env (JSON array of [name, js, waitMs])
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
const [url, out, hs] = process.argv.slice(2);
const steps = JSON.parse(process.env.STEPS || "[]");
const proc = spawn(hs, ["--no-sandbox","--disable-gpu","--hide-scrollbars","--remote-debugging-port=9333","--window-size=1440,900","about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let ws;
for (let i = 0; i < 50; i++) { try { const r = await fetch("http://127.0.0.1:9333/json"); const t = (await r.json()).find(x => x.type === "page"); ws = new WebSocket(t.webSocketDebuggerUrl); break; } catch { await sleep(200); } }
await new Promise(r => ws.onopen = r);
let id = 0; const pending = new Map();
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
const send = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
if (process.env.DARK) await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "dark" }] });
await send("Page.enable"); await send("Page.navigate", { url }); await sleep(4000);
for (const [name, js, wait] of steps) {
  if (js) { const r = await send("Runtime.evaluate", { expression: js, awaitPromise: true, returnByValue: true }); console.log(name, JSON.stringify(r.result?.result?.value ?? r.result?.exceptionDetails?.text ?? "")); }
  await sleep(wait ?? 1500);
  const s = await send("Page.captureScreenshot", { format: "png" }); writeFileSync(`${out}/${name}.png`, Buffer.from(s.result.data, "base64"));
}
ws.close(); proc.kill();
