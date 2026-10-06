// Node 에서 집어 옮기기 5회 (상자 위치 조금씩 흔듦) → 목적 구역 안에 놓였는지
import loadMujoco from '@mujoco/mujoco';
import fs from 'fs';
import { makeController, pickPlacePlan } from './pickplace_ctrl.mjs';
const mj = await loadMujoco();
mj.FS.writeFile('/pp.mjb', fs.readFileSync('pickplace.mjb'));
const vfs = new mj.MjVFS();
const model = mj.MjModel.from_binary_path('/pp.mjb', vfs);
const data = new mj.MjData(model);
const C = makeController(mj, model, data);
const gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, 'goal');
C.reset(); const goal = Array.from(data.geom_xpos.slice(3 * gid, 3 * gid + 3)); console.log('goal', goal.map(v=>v.toFixed(3)));
const SUB = 5;
let ok = 0;
for (const [t, jit] of [[0, [0, 0]], [1, [0.02, -0.01]], [2, [-0.02, 0.015]], [3, [0.01, 0.02]], [4, [-0.015, -0.02]]]) {
  C.reset(jit);
  const obj = C.bodyPos(C.bTarget), log = [];
  for (const ph of pickPlacePlan(obj, goal)) {
    data.ctrl[7] = ph.grip;
    let n = 0;
    const qT = ph.pos ? C.solveIK(ph.pos).q : null;
    while (n < 1500) {
      const done = qT ? C.moveToward(qT) : true;
      for (let s = 0; s < SUB; s++) { C.gravComp(); mj.mj_step(model, data); }
      n++;
      if (ph.hold) { if (n >= ph.hold / SUB) break; continue; }
      if (done && Math.hypot(...C.sitePos().map((v, i) => v - ph.pos[i])) < ph.tol) break;
    }
    { const q = Array.from(data.qpos.slice(12, 16)); const tilt = 2*Math.acos(Math.min(1,Math.abs(q[0])))*180/Math.PI; log.push(`${ph.name}:${n}(tilt ${tilt.toFixed(0)}°)`); }
  }
  for (let s = 0; s < 300; s++) mj.mj_step(model, data);
  const p = C.bodyPos(C.bTarget);
  const inGoal = Math.abs(p[0] - goal[0]) < 0.06 && Math.abs(p[1] - goal[1]) < 0.06 && p[2] < 0.05;
  if (inGoal) ok++;
  console.log(`trial ${t} ${inGoal ? 'PLACED' : 'MISS'} target=(${p.map(v => v.toFixed(3))}) ${log.join(' ')}`);
}
console.log(`placed ${ok}/5`);
