// Panda 집어 옮기기 컨트롤러 — Node 와 브라우저가 같은 파일을 쓴다.
// 손끝 사이트('gripper')를 6자유도 감쇠 최소제곱 IK 로 웨이포인트에 보내고, 손가락은 위치 명령으로 열고 닫는다.
// 플래너는 정해진 스크립트(접근 → 내려가기 → 쥐기 → 들기 → 옮기기 → 내려놓기 → 펴기 → 물러나기)이고, 물리는 MuJoCo 가 계산한다.

export const HOME = [0, 0.3, 0, -1.57079, 0, 2.0, -0.7853];
export const OPEN = 0.04, CLOSED = 0.0;
const NARM = 7, LAMBDA = 0.02, MAXDQ = 0.01;

export function makeController(mj, model, data) {
  const site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, 'gripper');
  const body = n => mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, n);
  const bTarget = body('target'), bLook = body('lookalike');
  const jacp = new mj.DoubleBuffer(3 * model.nv);
  const jacr = new mj.DoubleBuffer(3 * model.nv);
  let Rdes = null;   // 목표 손 방향 (reset 에서 정함)
  // 인덱스 가정 확인: 팔 7 + 손가락 2 + 상자 freejoint 7×2 = 23, 액추에이터 8 (손가락 1개가 두 손가락을 함께 구동)
  // 레벨 3(10/10): 상자 freejoint 가 3 개 더(방해 2 + 2차 목표) = 44. 뒤에 붙어 있어 위 번호(9..22)는 그대로
  if (![23, 44].includes(model.nq) || model.nu !== 8 || site < 0 || bTarget < 0) throw new Error(`unexpected model: nq=${model.nq} nu=${model.nu}`);

  function reset(jitter = [0, 0], place = null) {
    mj.mj_resetData(model, data);
    for (let i = 0; i < NARM; i++) { data.qpos[i] = HOME[i]; data.ctrl[i] = HOME[i]; }
    data.qpos[7] = data.qpos[8] = OPEN; data.ctrl[7] = OPEN;
    // 목표 상자 자유관절: qpos 9..15, 닮은 상자: 16..22
    data.qpos[9] += jitter[0]; data.qpos[10] += jitter[1];
    if (place) { data.qpos[9] = place.target[0]; data.qpos[10] = place.target[1]; data.qpos[16] = place.look[0]; data.qpos[17] = place.look[1]; }
    mj.mj_forward(model, data);
    // 손을 정확히 아래로(집게 z축 = -세계 z), 손가락은 세계 x 방향으로 벌어지게 고정. 기본 자세는 7° 기울어 있어 상자를 밀었다
    Rdes = [0, 1, 0, 1, 0, 0, 0, 0, -1];
  }
  // V3(10/9): 손을 세계 z 축으로 ψ 만큼 돌린 방향을 목표로 (Rdes = Rz(ψ)·R0). 손가락이 벌어지는 축이 (cos ψ, sin ψ) 가 된다. reset 이 다시 0 으로
  function setYaw(psi) {
    const c = Math.cos(psi), s = Math.sin(psi), R0 = [0, 1, 0, 1, 0, 0, 0, 0, -1], Rz = [c, -s, 0, s, c, 0, 0, 0, 1];
    Rdes = [0, 1, 2].flatMap(i => [0, 1, 2].map(j => Rz[3 * i] * R0[j] + Rz[3 * i + 1] * R0[3 + j] + Rz[3 * i + 2] * R0[6 + j]));
  }

  const sitePos = () => Array.from(data.site_xpos.slice(3 * site, 3 * site + 3));
  const bodyPos = b => Array.from(data.xpos.slice(3 * b, 3 * b + 3));

  // 회전 오차: 0.5 * Σ (현재 축 × 목표 축)
  function rotErrOf(R) {
    const e = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const a = [R[c], R[3 + c], R[6 + c]], b = [Rdes[c], Rdes[3 + c], Rdes[6 + c]];
      e[0] += 0.5 * (a[1] * b[2] - a[2] * b[1]); e[1] += 0.5 * (a[2] * b[0] - a[0] * b[2]); e[2] += 0.5 * (a[0] * b[1] - a[1] * b[0]);
    }
    return e;
  }

  // 별도 MjData 에서 기구학만 돌려 goal 에 맞는 관절각을 푼다 (감쇠 최소제곱, 실제 로봇 상태는 건드리지 않음)
  const scratch = new mj.MjData(model);
  function solveIK(goal, iters = 150, seed = null) {
    for (let i = 0; i < model.nq; i++) scratch.qpos[i] = data.qpos[i];
    for (let i = 0; i < NARM; i++) scratch.qpos[i] = seed ? seed[i] : data.ctrl[i];   // 직전 명령(또는 주어진 자세)에서 출발 → 자세가 매끄럽게 이어짐
    let err = 1;
    for (let it = 0; it < iters; it++) {
      mj.mj_kinematics(model, scratch); mj.mj_comPos(model, scratch);
      const p = Array.from(scratch.site_xpos.slice(3 * site, 3 * site + 3));
      const er = rotErrOf(scratch.site_xmat.slice(9 * site, 9 * site + 9));
      const e = [goal[0] - p[0], goal[1] - p[1], goal[2] - p[2], er[0], er[1], er[2]];
      err = Math.hypot(e[0], e[1], e[2]);
      if (err < 1e-4 && Math.hypot(er[0], er[1], er[2]) < 1e-3) break;
      mj.mj_jacSite(model, scratch, jacp, jacr, site);
      const P = jacp.GetView(), Rr = jacr.GetView(), nv = model.nv, J = [];
      for (let r = 0; r < 3; r++) J.push(Array.from({ length: NARM }, (_, c) => P[r * nv + c]));
      for (let r = 0; r < 3; r++) J.push(Array.from({ length: NARM }, (_, c) => Rr[r * nv + c]));
      const A = J.map((ri, a) => J.map((rj, b) => ri.reduce((sum, v, k) => sum + v * rj[k], 0) + (a === b ? LAMBDA * LAMBDA : 0)));
      const y = solve(A, e);
      for (let c = 0; c < NARM; c++) {
        const dq = J.reduce((sum, row, r) => sum + row[c] * y[r], 0);
        const lo = model.jnt_range[2 * c], hi = model.jnt_range[2 * c + 1];
        scratch.qpos[c] = Math.min(hi, Math.max(lo, scratch.qpos[c] + Math.max(-0.2, Math.min(0.2, dq))));
      }
    }
    if (err > 0.005) console.warn(`IK did not converge: ${err.toFixed(4)} m`);
    return { q: Array.from(scratch.qpos.slice(0, NARM)), err };
  }

  // 관절 명령을 qT 쪽으로 한 걸음(최대 MAXDQ) 옮긴다. 다 왔으면 true
  function moveToward(qT) {
    let done = true;
    for (let i = 0; i < NARM; i++) {
      const d = qT[i] - data.ctrl[i];
      if (Math.abs(d) > MAXDQ) done = false;
      data.ctrl[i] += Math.max(-MAXDQ, Math.min(MAXDQ, d));
    }
    return done;
  }

  // 중력·코리올리 보상: 위치 서보가 처지지 않게 (실제 Panda 제어기도 중력 보상을 한다)
  function gravComp() { for (let i = 0; i < NARM + 2; i++) data.qfrc_applied[i] = data.qfrc_bias[i]; }

  // 관절각 q(팔 7)에서의 손끝 위치 — 순기구학 검증기(게이트)가 쓴다
  function fkTip(q) {
    for (let i = 0; i < model.nq; i++) scratch.qpos[i] = data.qpos[i];
    for (let i = 0; i < NARM; i++) scratch.qpos[i] = q[i];
    mj.mj_kinematics(model, scratch);
    return Array.from(scratch.site_xpos.slice(3 * site, 3 * site + 3));
  }

  // 관절각 q 에서 손 방향이 목표 방향(아래, 손가락 x)에서 벗어난 정도 (작은 각에서 ≈ 라디안)
  function handErr(q) {
    for (let i = 0; i < model.nq; i++) scratch.qpos[i] = data.qpos[i];
    for (let i = 0; i < NARM; i++) scratch.qpos[i] = q[i];
    mj.mj_kinematics(model, scratch);
    return Math.hypot(...rotErrOf(scratch.site_xmat.slice(9 * site, 9 * site + 9)));
  }

  function free() { scratch.delete(); jacp.delete(); jacr.delete(); }

  return { reset, setYaw, solveIK, moveToward, gravComp, sitePos, bodyPos, fkTip, handErr, bTarget, bLook, site, free };
}

function solve(A, b) {   // 가우스 소거 (6×6)
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < n; i++) {
    let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
    [M[i], M[p]] = [M[p], M[i]];
    for (let r = i + 1; r < n; r++) { const f = M[r][i] / M[i][i]; for (let c = i; c <= n; c++) M[r][c] -= f * M[i][c]; }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let c = i + 1; c < n; c++) s -= M[i][c] * x[c]; x[i] = s / M[i][i]; }
  return x;
}

// 집어 옮기기 단계. 각 단계: 손끝 목표(물체 위치 기준), 손가락 명령, 다음 단계로 넘어갈 조건
export function pickPlacePlan(objPos, goalPos) {
  const above = 0.15, grasp = 0.005;
  return [
    { name: 'approach', pos: [objPos[0], objPos[1], objPos[2] + above], grip: OPEN, tol: 0.01 },
    { name: 'descend', pos: [objPos[0], objPos[1], objPos[2] + grasp], grip: OPEN, tol: 0.006 },
    { name: 'close', hold: 250, grip: CLOSED },
    { name: 'lift', pos: [objPos[0], objPos[1], objPos[2] + above], grip: CLOSED, tol: 0.015 },
    { name: 'carry', pos: [goalPos[0], goalPos[1], objPos[2] + above], grip: CLOSED, tol: 0.015 },
    { name: 'lower', pos: [goalPos[0], goalPos[1], objPos[2] + grasp + 0.002], grip: CLOSED, tol: 0.008 },
    { name: 'release', hold: 200, grip: OPEN },
    { name: 'retreat', pos: [goalPos[0], goalPos[1], objPos[2] + above], grip: OPEN, tol: 0.015 },
  ];
}
