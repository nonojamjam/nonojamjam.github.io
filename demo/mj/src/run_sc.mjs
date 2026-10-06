// 7개 시나리오 일괄 실행 (Node) → 설정별 성공·잘못 옮김·놓침·플래너 호출 등
import loadMujoco from '@mujoco/mujoco'; import fs from 'fs';
import { makeController } from './pickplace_ctrl.mjs';
import { makeEngine, SCENARIOS } from './scenarios.mjs';
const mj = await loadMujoco(); mj.FS.writeFile('/pp.mjb', fs.readFileSync('pickplace.mjb'));
const model = mj.MjModel.from_binary_path('/pp.mjb', new mj.MjVFS()); const data = new mj.MjData(model);
const C = makeController(mj, model, data); C.reset();
const gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, 'goal'); const goal = Array.from(data.geom_xpos.slice(3 * gid, 3 * gid + 3));
const E = makeEngine(mj, model, data, C, goal);
const only = process.argv[2] ? process.argv[2].split(',').map(Number) : null;
for (const sc of SCENARIOS) {
  if (only && !only.includes(sc.id)) continue;
  const t0 = Date.now();
  const rows = sc.cfgs.map((cfg, i) => { const s = E.runBatch(cfg); return `${sc.labels[i].padEnd(28)} success ${s.success}/${s.n} wrong ${s.wrong} miss ${s.miss} | calls ${s.calls} steps ${s.steps} memSteps ${s.memSteps} rej ${s.rejected} unsafe ${s.unsafe} memDel ${s.memDeleted} notReached ${s.notReached} blocked ${s.blocked} caught ${s.caught} dropped ${s.dropped}`; });
  console.log(`\n[${sc.id}] ${sc.title}  (${((Date.now() - t0) / 1000).toFixed(1)} s)\n  ` + rows.join('\n  '));
}
