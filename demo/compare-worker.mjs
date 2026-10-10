// Background comparison for the oracle-free demo, in its own thread so the visible run never waits for it.
// Same engine, model, scene seed and noise seed as the page; one fresh engine per row (the engine builds its camera
// once with the first setting's options). Message in: { job, items: [{ i, cfg, mem }], seed, scene }.
// Messages out: { job, type: 'busy', i } · { job, type: 'row', i, row } · { job, type: 'done' } · { type: 'error', msg }.
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.15.0/mujoco.js';
import { makeController } from './mj/pickplace_ctrl.mjs';
import { makeEngine } from './mj/scenarios.mjs';
import { makeLiveMemory } from './mj/recall.mjs';

const ready = (async () => {
  const mj = await loadMujoco();
  mj.FS.writeFile('/pp.mjb', new Uint8Array(await (await fetch('mj/pickplace_l3.mjb')).arrayBuffer()));
  const model = mj.MjModel.from_binary_path('/pp.mjb', new mj.MjVFS()), data = new mj.MjData(model);
  const C = makeController(mj, model, data); C.reset();
  const g = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, 'goal'), goal = Array.from(data.geom_xpos.slice(3 * g, 3 * g + 3));
  const BANK = await (await fetch('mj/bank_v6_qwen_mid.json')).json();
  return { mj, model, data, C, goal, BANK };
})();

let current = 0;   // the newest job; older jobs stop between episodes
const yieldNow = () => new Promise(r => setTimeout(r, 0));

async function run({ job, items, seed, scene }) {
  const { mj, model, data, C, goal, BANK } = await ready;
  for (const { i, cfg, mem } of items) {
    if (job !== current) return;
    postMessage({ job, type: 'busy', i });
    const E = makeEngine(mj, model, data, C, goal);
    const M = mem === 'none' ? null : makeLiveMemory(mem === 'bank' ? BANK.candidates : [], 0.5);
    const c = { ...cfg, ...(M ? { recall: M.recall } : {}) };
    const st = E.newStats(), memo = { scene: new Map(), act: new Map() }, pv = [], pm = [], okv = [];
    let looks = 0;
    for (const task of E.tasks(false, scene)) {
      const c0 = st.calls, m0 = st.memSteps, s0 = st.standing;
      const gen = E.episode(task, c, memo, st, seed, (k, x) => { if (M) M.onEvent(k, x); if (k === 'active-sense') looks += x.probes; });
      while (!gen.next().done);
      pv.push(st.calls - c0); pm.push(st.memSteps - m0); okv.push(st.standing > s0);
      await yieldNow(); if (job !== current) return;
    }
    const row = { ...st, looks, bank: M ? M.size - (mem === 'bank' ? BANK.candidates.length : 0) : 0,
      rounds: [0, 1, 2].map(r => pv.slice(6 * r, 6 * r + 6).reduce((a, b) => a + b, 0)), pv, pm, okv };
    postMessage({ job, type: 'row', i, row });
  }
  if (job === current) postMessage({ job, type: 'done' });
}

onmessage = e => { current = e.data.job; run(e.data).catch(err => postMessage({ type: 'error', msg: String(err && err.stack || err) })); };
