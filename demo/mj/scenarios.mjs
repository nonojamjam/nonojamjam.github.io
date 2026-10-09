// 7개 시나리오를 MuJoCo 물리 위에서 돌리는 에피소드 엔진 (걸음 단위 판단). Node(일괄 통계)와 브라우저(재생)가 같은 코드를 쓴다.
// 원래 데모와 같은 구조: 손끝을 잡을 지점 위로 보내는 동안 걸음마다
//   상황 키로 기억 조회 → 없으면 플래너(LLM 대역)에게 다음 관절 이동을 묻고 → 순기구학 게이트가 검사(막히면 다시 생성)
//   → MuJoCo 가 실제로 그 걸음을 실행 → 나아갔으면 기억에 저장, 기억에서 꺼낸 걸음이 안 나아갔으면 그 기억을 지운다.
// 다 오면 내려가 실제 손가락으로 쥐고(촉각), 들어 옮겨 놓는다. 에피소드는 제너레이터: 제어 한 틱마다 yield.
import { OPEN, CLOSED } from './pickplace_ctrl.mjs';

export const TARGET_W = 0.032, LOOK_W = 0.040;     // 손가락이 닫힌 폭: 목표 상자 32 mm, 닮은 상자 40 mm
const SUB = 5, PRE = 0.08, ABOVE = 0.15, GRASP_Z = 0.035, PLACE_Z = 0.037;
const BIAS_JOINT = 1, TOUCH_TOL = 0.006, CAM_NOISE = 0.003, SENSOR_NOISE = 0.002;
const STEP = 0.06, SUCC = 0.025, PROG = 0.005, JERK = 0.4, MAX_STEPS = 12, MAX_REPLAN = 3, UNSAFE = 0.005, TILT = 0.25;
const FLOOR_EPS = 0.003;   // P1(10/9): 쥔 상자를 들고 걸을 때 손끝이 PLACE_Z(상자가 탁자에 닿는 높이)보다 이만큼 아래로 내려가는 제안은 거절

export const SCENES = [
  { target: [0.55, 0.10], look: [0.45, 0.20] }, { target: [0.50, 0.18], look: [0.60, 0.05] },
  { target: [0.60, 0.00], look: [0.48, 0.12] }, { target: [0.45, 0.08], look: [0.58, 0.18] },
  { target: [0.57, 0.16], look: [0.44, 0.02] }, { target: [0.48, 0.00], look: [0.60, 0.12] },
];

// 장면 seed → 장면 목록. seed 0(또는 없음) = 위 고정 6장면 (데모·기준선 전용, 통계 제외).
// 양의 정수 seed = 무작위 6장면 (S1b-1: 위치만. 회전은 손목 정렬이 필요해 S1b-2).
// 범위는 고정 장면이 이미 검증한 영역 그대로, 닮은 상자는 목표와 SCENE_MIN_SEP 이상 떨어뜨린다.
// 장면 난수는 잡음 seed(runBatch 의 seed)와 별도 스트림이다.
const SCENE_X = [0.45, 0.60], SCENE_Y = [0.00, 0.18], SCENE_MIN_SEP = 0.10, SCENE_N = 6;
export function scenesFor(sceneSeed = 0) {
  if (!sceneSeed) return SCENES;
  if (!Number.isInteger(sceneSeed) || sceneSeed < 0) throw new Error(`scenesFor: sceneSeed 는 0 이상 정수여야 함 (${sceneSeed})`);
  // LCG 첫 출력은 seed 와 거의 선형이라(첫 장면 x 가 seed 순으로 0.494→0.507 [실측]) 해시로 먼저 섞는다
  let h = (sceneSeed ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b); h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35); h = (h ^ (h >>> 16)) >>> 0;
  const rng = makeRng(h);
  const pick = () => [SCENE_X[0] + (SCENE_X[1] - SCENE_X[0]) * rng(), SCENE_Y[0] + (SCENE_Y[1] - SCENE_Y[0]) * rng()];
  const out = [];
  for (let k = 0; k < SCENE_N; k++) {
    const target = pick(); let look = null;
    for (let tries = 0; tries < 200 && !look; tries++) { const c = pick(); if (Math.hypot(c[0] - target[0], c[1] - target[1]) >= SCENE_MIN_SEP) look = c; }
    if (!look) throw new Error(`scenesFor: seed ${sceneSeed} 장면 ${k} 닮은 상자 배치 실패`);
    out.push({ target, look });
  }
  return out;
}

// 쓰러짐 판정 (S2a): 상자 긴 축(몸체 z, 반높이 0.03 m) 과 세계 수직의 각도. xmat 은 행 우선 3×3 → z 축의 세계 z 성분 = m[8].
// 평지에 멈춘 상자는 0°(섬) 아니면 90°(누움) 이라 45° 에서 가른다. 자세는 시뮬 참값 = 오라클.
export const FALL_DEG = 45;
export function tiltDegFromXmat(m) { return Math.acos(Math.min(1, Math.abs(m[8]))) * 180 / Math.PI; }

export function makeRng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
const gauss = rng => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
// O1 (10/9): 오라클 손끝 센서 대신 '거리 기반 외부 추적' 모형 (UWB 같은 거리 측정 방식, 하드웨어 없이 센서 모형만).
//   앵커 4개(높이가 달라 한 평면에 있지 않음)까지 거리 = 참 거리 + N(0, σ) + 확률 pNlos 로 양의 바이어스 U(0, nlosMax)(가려짐),
//   믿는 손끝에서 시작하는 가우스-뉴턴 삼변측량으로 위치를 추정한다. cfg.tracker 가 있을 때만 (없으면 예전 참값+잡음 센서)
export const TRACKER_DEFAULT = { anchors: [[0.2, -0.6, 0.8], [1.0, -0.6, 0.1], [0.2, 0.6, 0.1], [1.0, 0.6, 0.8]], sigma: 0.02, pNlos: 0.1, nlosMax: 0.15 };
function solve3(A, b) {   // 3×3 선형계 (크래머), 거의 특이면 감쇠
  const M = A.map((r, i) => r.map((v, j) => v + (i === j ? 1e-9 : 0)));
  const det = m => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(M);
  return [0, 1, 2].map(k => det(M.map((r, i) => r.map((v, j) => (j === k ? b[i] : v)))) / D);
}
export function rangeTrack(p, rng, tr, x0) {
  const z = tr.anchors.map(a => Math.hypot(p[0] - a[0], p[1] - a[1], p[2] - a[2]) + tr.sigma * gauss(rng) + (rng() < tr.pNlos ? tr.nlosMax * rng() : 0));
  let x = x0.slice();
  for (let it = 0; it < 10; it++) {
    const JTJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], JTr = [0, 0, 0];
    tr.anchors.forEach((a, i) => {
      const d = [x[0] - a[0], x[1] - a[1], x[2] - a[2]], n = Math.hypot(...d) || 1e-9, g = d.map(v => v / n), r = n - z[i];
      for (let u = 0; u < 3; u++) { JTr[u] += g[u] * r; for (let v = 0; v < 3; v++) JTJ[u][v] += g[u] * g[v]; }
    });
    const dx = solve3(JTJ, JTr); x = x.map((v, i) => v - dx[i]);
    if (Math.hypot(...dx) < 1e-6) break;
  }
  return x.every(Number.isFinite) ? x : x0.slice();   // 검수 반영: 발산(NaN)이면 믿는 손끝으로 되돌린다
}
const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

export function makeEngine(mj, model, data, C, goal) {
  const physicsTick = () => { for (let s = 0; s < SUB; s++) { C.gravComp(); mj.mj_step(model, data); } };
  const jr = j => [model.jnt_range[2 * j], model.jnt_range[2 * j + 1]];

  // 관절 명령: 로봇은 "읽은 값"이 qBelief 가 되도록 서보한다. 읽기에 bias 가 있으면 실제 관절은 qBelief - bias 로 간다
  function* moveJoints(qBelief, bias, maxTicks = 500) {
    const qTrue = qBelief.map((v, i) => v - (i === BIAS_JOINT ? bias : 0));
    let settle = 0;
    for (let t = 0; t < maxTicks; t++) { const done = C.moveToward(qTrue); physicsTick(); yield; if (done && ++settle > 20) return; }
  }
  function* hold(ticks) { for (let t = 0; t < ticks; t++) { physicsTick(); yield; } }
  const readJoints = bias => Array.from(data.qpos.slice(0, 7)).map((v, i) => v + (i === BIAS_JOINT ? bias : 0));

  // 상황 키: 목표까지 방향(x·y·z 부호, 1 cm 불감대) + 거리 구간 + 큰 관절 3개(1·2·4)의 구간
  function situationKey(tip, target, q) {
    const sg = v => (Math.abs(v) < 0.01 ? 0 : Math.sign(v));
    const dd = d3(tip, target), db = dd < 0.05 ? 0 : dd < 0.15 ? 1 : 2;
    const jb = [1, 2, 3].map(j => { const [lo, hi] = jr(j); return Math.min(2, Math.floor(3 * (q[j] - lo) / (hi - lo))); });
    return [sg(target[0] - tip[0]), sg(target[1] - tip[1]), sg(target[2] - tip[2]), db, ...jb].join(',');
  }

  // 플래너(대역): 믿는 손끝에서 목표 쪽으로 최대 6 cm 가는 관절 이동을 낸다. 확률 pm 으로 관절 두 개를 크게 틀리게 낸다
  function planStep(qB, tipB, target, cfg, st, rng, rej = null) {   // L10: rej = 이 걸음에서 마지막으로 거절된 제안 {reason, src, dtip} (진짜 LLM 만 쓴다, 대역은 무시)
    st.calls++;
    if (cfg.planner) {   // L1: 진짜 LLM(파일럿) — 손끝 이동량 [m] 배열이나 {dq} 를 받는다. 대역의 실수 주입·난수는 쓰지 않는다
      const d = cfg.planner({ tip: tipB, target, q: qB, reject: rej });
      if (!d) return qB.map(() => 0);   // 못 읽은 응답 = 움직이지 않음 (게이트가 전진 없음으로 거절)
      if (d.dq) return d.dq.slice();
      const f = C.fkTip(qB);
      return C.solveIK([f[0] + d[0], f[1] + d[1], f[2] + d[2]], 60, qB).q.map((x, i) => x - qB[i]);
    }
    const fk = C.fkTip(qB), v = [target[0] - tipB[0], target[1] - tipB[1], target[2] - tipB[2]];
    const n = Math.hypot(...v), k = n > STEP ? STEP / n : 1;
    const way = [fk[0] + k * v[0], fk[1] + k * v[1], fk[2] + k * v[2]];   // 센서가 있으면 tipB 는 잰 손끝 → 오차만큼 목표를 옮겨 낸다
    const q = C.solveIK(way, 60, qB).q, dq = q.map((x, i) => x - qB[i]);
    const pm = cfg.pre ? cfg.pMistake / 4 : cfg.pMistake;   // 계획 전 검사: 같은 상태를 보고 실수율만 낮춘다
    if (rng() < pm) for (const j of [Math.floor(rng() * 4), 4 + Math.floor(rng() * 3)]) dq[j] += (rng() < 0.5 ? -1 : 1) * (0.25 + 0.15 * rng());
    return dq;
  }

  // 게이트: 이 이동 뒤 손끝(읽은 관절값 기준, 센서가 있으면 잰 손끝 기준)이 목표에 다가가는지, 급하지 않은지, 관절 한계 안인지, 손이 계속 아래를 향하는지
  // S5a: 같은 순서로 검사하되 처음 걸린 이유와 수치를 함께 돌려준다 (로그 스키마 proposal.gate). 난수를 쓰지 않는다
  // P1: floorZ 를 주면(쥔 채 걷는 단계) 예측 손끝 높이가 그보다 낮은 제안을 'floor' 로 거절 — 게이트가 쥔 물체·탁자 충돌을 안 보던 구멍(10/9 패널)
  function gateCheck(qB, tipB, dq, target, floorZ = null) {
    const q1 = qB.map((x, i) => x + dq[i]);
    const maxDq = Math.max(...dq.map(Math.abs)), handErr = C.handErr(q1);
    let margin = Infinity; for (let j = 0; j < 7; j++) { const [lo, hi] = jr(j); margin = Math.min(margin, q1[j] - lo, hi - q1[j]); }
    const f0 = C.fkTip(qB), f1 = C.fkTip(q1), dtip = [f1[0] - f0[0], f1[1] - f0[1], f1[2] - f0[2]];
    const pred = [tipB[0] + dtip[0], tipB[1] + dtip[1], tipB[2] + dtip[2]], progress = d3(tipB, target) - d3(pred, target);
    const floorHit = floorZ !== null && pred[2] < floorZ;
    const reason = maxDq > JERK ? 'jerk' : handErr > TILT ? 'tilt' : margin < 0 ? 'limit' : floorHit ? 'floor' : progress > PROG ? null : 'progress';
    const values = { max_dq: maxDq, hand_err: handErr, limit_margin: margin, progress };
    if (floorZ !== null) values.pred_z = pred[2];   // 바닥 검사를 켰을 때만 (끄면 기록이 예전과 같다)
    return { ok: reason === null, reason, dtip, values };
  }
  const gateOk = (qB, tipB, dq, target) => gateCheck(qB, tipB, dq, target).ok;

  function* episode(task, cfg, mem, st, seed, ev = () => {}) {
    const base = (seed * 7919 + task.idx * 104729) >>> 0;
    const rCam = makeRng(base + 1), rPlan = makeRng(base + 2), rSen = makeRng(base + 3);
    const bias = cfg.bias ? 0.15 : 0;
    C.reset([0, 0], task.place);
    for (let t = 0; t < 40; t++) { physicsTick(); yield; }
    const truth = C.bodyPos(C.bTarget);
    const cam = () => [truth[0] + CAM_NOISE * gauss(rCam), truth[1] + CAM_NOISE * gauss(rCam)];
    let belief, source;
    if (cfg.mem && mem.scene.has(task.key)) {
      belief = mem.scene.get(task.key); source = 'memory';
      if (cfg.camCheck) { const c = cam(); if (d2(belief, c) > 0.02) { mem.scene.delete(task.key); belief = c; source = 'camera'; st.dropped++; ev('drop'); } }
    } else { belief = cam(); source = 'camera'; }
    ev('belief', { belief, source, task });
    const tipNow = () => C.sitePos();
    const TR = cfg.tracker ? (cfg.tracker === true ? TRACKER_DEFAULT : { ...TRACKER_DEFAULT, ...cfg.tracker }) : null;
    const sense = qB => {   // O1: 손끝 '센서' 한 번 읽기 — 추적 모형이면 거리 삼변측량, 아니면 예전 참값+잡음
      if (!TR) return tipNow().map(v => v + SENSOR_NOISE * gauss(rSen));
      const t = tipNow();
      if (!cfg.trustRoute) {
        const est = rangeTrack(t, rSen, TR, C.fkTip(qB));
        ev('track', { err: Math.hypot(est[0] - t[0], est[1] - t[1], est[2] - t[2]) });
        return est;
      }
      // G3: 인식 = 라우팅. 추적을 k 번 읽어 좌표별 중앙값(가려짐 바이어스에 강함)을 내고, 믿는 관절의 FK 와 δ 넘게 어긋날 때만 추적을 믿는다.
      //   어긋나지 않으면 FK(오독이 없으면 정밀)를 쓴다. 비용 = 읽기 k 번
      const { k = 5, delta = 0.05 } = cfg.trustRoute, fk = C.fkTip(qB), reads = [];
      for (let i = 0; i < k; i++) reads.push(rangeTrack(t, rSen, TR, fk));
      const med = [0, 1, 2].map(j => reads.map(r => r[j]).sort((a, b) => a - b)[k >> 1]);
      const dis = Math.hypot(med[0] - fk[0], med[1] - fk[1], med[2] - fk[2]), useTrack = dis > delta, out = useTrack ? med : fk;
      ev('track', { err: Math.hypot(out[0] - t[0], out[1] - t[1], out[2] - t[2]), dis, src: useTrack ? 'tracker' : 'fk', reads: k });
      return out;
    };
    const believedTip = qB => cfg.sensor ? sense(qB) : C.fkTip(qB);
    data.ctrl[7] = OPEN;
    // S4: phase 태그 — 추론하지 않고 제어 코드가 지금 하는 일을 그대로 적는다 (approach·grasp·lift·carry·place·retreat)
    let phase = null;
    const setPhase = p => { phase = p; ev('phase', { phase: p }); };

    // L5(E1): 경유점까지 걸음 단위로 가기 — 예전 접근 루프를 그대로 함수로 뺐다 (동작·난수 불변). 걸음 번호는 에피소드 전체에서 이어진다
    //   (재시도마다 0 으로 돌아가면 기록의 (ep, step, attempt) 짝이 겹친다)
    let stepNo = 0;
    function* stepTo(target) {
      let reached = false;
      for (let n = 0; n < MAX_STEPS; n++) {
        const qB = readJoints(bias), tipB = believedTip(qB);
        if (d3(tipB, target) < SUCC) { reached = true; break; }
        const step = stepNo++;
        const key = (phase === 'approach' ? '' : phase + ':') + situationKey(tipB, target, qB);   // E2: 접근 밖 단계는 키에 단계를 붙인다 (접근 기억을 운반에서 꺼내지 않게)
        // 타임라인용: 플래너 호출마다 번호(call)를 붙이고, 기억에는 그 걸음을 낸 호출 번호(from)를 함께 저장한다
        let lastRej = null;   // L10: 게이트가 마지막으로 거절한 제안 — 다음 플래너 호출에 이유를 넘긴다
        const ask = () => { const d = planStep(qB, tipB, target, cfg, st, rPlan, lastRej); ev('call', { id: st.calls, vars: { q: qB, tip: tipB, target: target }, dq: d.slice() }); return d; };
        let dq = null, src = 'plan', call = 0, from = 0;
        // L6c: cfg.recall = 실행 밖 기억 묶음(DB). 재생 후보 [{id, dtip}] 를 점수순으로 게이트에 넣고, 없거나 다 거절되면 플래너.
        //   손끝 변위 → 지금 자세에서 IK 로 관절 이동 (R2: 관절 증분 그대로보다 쓸 수 있는 거리가 두 배). 엔진 Map 기억(cfg.mem)과 함께 쓰지 않는다
        const memQ = cfg.recall ? cfg.recall({ phase, tip: tipB, target, q: qB }).slice() : [];
        const nextMem = () => {
          const m = memQ.shift(), f = C.fkTip(qB);
          dq = C.solveIK([f[0] + m.dtip[0], f[1] + m.dtip[1], f[2] + m.dtip[2]], 60, qB).q.map((x, i) => x - qB[i]);
          from = m.id; src = 'mem'; call = 0; st.replays++;
        };
        if (cfg.mem && mem.act.has(key)) { const m = mem.act.get(key); dq = m.dq.slice(); from = m.from; src = 'mem'; st.replays++; }
        else if (memQ.length) nextMem();
        else { dq = ask(); call = st.calls; }
        let attemptNo = 0, gateUs = 0;
        const judge = () => {   // 제안 하나를 게이트에 넣고 proposal 이벤트를 남긴다 (게이트 꺼짐이면 판정 없이 기록만)
          const floorZ = cfg.stepPhases && ['lift', 'carry', 'place'].includes(phase) ? PLACE_Z - FLOOR_EPS : null;   // P1: 걸음 단계 옵션일 때 쥔 채 걷는 단계만
          const t0 = performance.now(), g = cfg.gate ? gateCheck(qB, tipB, dq, target, floorZ) : null; gateUs += performance.now() - t0;
          const dtip = g ? g.dtip : (() => { const a = C.fkTip(qB), b = C.fkTip(qB.map((x, i) => x + dq[i])); return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]; })();
          ev('proposal', { step, attempt: attemptNo++, phase, src, call, from, dq: dq.slice(), dtipPred: dtip, gate: g && { ok: g.ok, reason: g.reason, values: g.values } });
          if (g && !g.ok) lastRej = { reason: g.reason, src, dtip };
          return g ? g.ok : true;
        };
        if (!cfg.gate) judge();
        if (cfg.gate) {
          let tries = 0, lastOk = false;   // 루프를 빠져나온 마지막 판정 = 예전 코드의 재검사 결과와 같다 (같은 dq 를 두 번 기록하지 않으려고 재사용)
          // L6c: DB 기억 후보가 남아 있으면 거절돼도 재계획 횟수(tries)를 쓰지 않고 다음 후보로. 후보가 없으면 예전과 똑같은 순서·횟수
          while (!(lastOk = judge())) {
            if (!memQ.length && tries++ >= MAX_REPLAN) break;
            st.rejected++; ev('reject', { q: qB.map((x, i) => x + dq[i] - (i === BIAS_JOINT ? bias : 0)), src, call, from });
            if (src === 'mem' && cfg.mem) { mem.act.delete(key); st.memDeleted++; ev('forget', { from }); }
            if (memQ.length) nextMem();
            else { dq = ask(); call = st.calls; from = 0; src = 'plan'; }
          }
          if (!lastOk) { st.blocked++; ev('blocked'); continue; }   // 끝까지 막히면 움직이지 않고 다음 걸음에서 다시 묻는다
        }
        const realBefore = d3(tipNow(), target), qTrue0 = Array.from(data.qpos.slice(0, 7)), tip0 = tipNow();
        yield* moveJoints(qB.map((x, i) => x + dq[i]), bias);
        const qB2 = readJoints(bias), tipB2 = believedTip(qB2);
        const unsafe = d3(tipNow(), target) > realBefore + UNSAFE;   // 실제 손끝이 향하던 지점에서 멀어졌다
        if (unsafe) st.unsafe++;
        st.steps++; if (src === 'mem') st.memSteps++;
        const progressed = d3(tipB2, target) < d3(tipB, target) - PROG;
        if (cfg.mem) {
          if (src === 'plan' && progressed) { mem.act.set(key, { dq, from: call }); ev('store', { call }); }
          if (src === 'mem' && !progressed) { mem.act.delete(key); st.memDeleted++; ev('forget', { from }); }
        }
        const tip1 = tipNow(), qTrue1 = Array.from(data.qpos.slice(0, 7));
        ev('step', { src: unsafe ? 'unsafe' : src, mem: src === 'mem', tip: tip1, belief: tipB2, key, call, from, phase,
          step, attempt: attemptNo - 1, qRead: qB, qTrue: qTrue0, tipToTarget: [target[0] - tipB[0], target[1] - tipB[1], target[2] - tipB[2]],
          dqCmd: dq.slice(), dqActual: qTrue1.map((v, i) => v - qTrue0[i]), dtipActual: [tip1[0] - tip0[0], tip1[1] - tip0[1], tip1[2] - tip0[2]],
          distDelta: d3(tip1, target) - realBefore, progressed, unsafe, gateUs });
      }
      if (!reached) { const qB = readJoints(bias); reached = d3(believedTip(qB), target) < SUCC; }
      return reached;
    }

    for (let attempt = 0; attempt < 3; attempt++) {
      setPhase('approach');
      const pre = [belief[0], belief[1], GRASP_Z + PRE];
      ev('reach-start', { tip: tipNow(), belief: believedTip(readJoints(bias)) });
      // ---- 걸음 단위로 잡을 지점 위까지 ----
      const reached = yield* stepTo(pre);
      if (!reached) { st.notReached++; ev('not-reached'); break; }

      // ---- 정렬해서 내려가 쥐기 (읽은 관절값 기준 IK). 센서가 있으면 잰 손끝 오차를 매번 다시 재서 고친다 ----
      let off = [0, 0, 0];
      const corrected = function* (pt) {
        yield* moveJoints(C.solveIK([pt[0] + off[0], pt[1] + off[1], pt[2] + off[2]], 150, readJoints(bias)).q, bias);
        if (!cfg.sensor) return;
        for (let k = 0; k < 3; k++) {
          const m = sense(readJoints(bias)), e = [pt[0] - m[0], pt[1] - m[1], pt[2] - m[2]];
          if (Math.hypot(...e) < 0.004) return;
          off = [off[0] + e[0], off[1] + e[1], off[2] + e[2]]; ev('sensor-fix');
          yield* moveJoints(C.solveIK([pt[0] + off[0], pt[1] + off[1], pt[2] + off[2]], 150, readJoints(bias)).q, bias);
        }
      };
      const g = [belief[0], belief[1], GRASP_Z];
      setPhase('grasp');
      yield* corrected([g[0], g[1], g[2] + PRE]);
      yield* corrected([g[0], g[1], g[2] + 0.03]);
      yield* corrected(g);
      data.ctrl[7] = CLOSED; yield* hold(60);
      const width = data.qpos[7] + data.qpos[8];
      ev('touch', { width, tip: tipNow(), belief: C.fkTip(readJoints(bias)) });   // belief = 읽은 관절값으로 계산한 손끝 (센서와 무관)
      if (cfg.touch && Math.abs(width - TARGET_W) > TOUCH_TOL) {
        st.caught++; ev('caught', { width });
        data.ctrl[7] = OPEN; yield* hold(40);
        yield* corrected([g[0], g[1], g[2] + ABOVE]);
        mem.scene.delete(task.key); belief = cam(); continue;
      }
      // ---- 들어서 옮겨 놓기 ----
      // L5(E2): cfg.stepPhases 에 든 단계는 경유점까지 걸음 단위(플래너·게이트·기억)로 간 뒤 IK 로 마무리 정렬한다. 없으면 예전처럼 IK 한 번.
      //   단계 표지를 먼저 세운 뒤 걷는다 (setPhase 누락 시 이전 단계 이름이 찍힌다 — E1 검수 지적)
      const go = function* (p, pt) {
        setPhase(p);
        if (cfg.stepPhases?.includes(p) && !(yield* stepTo(pt))) ev('phase-not-reached', { phase: p });
        yield* corrected(pt);
      };
      yield* go('lift', [g[0], g[1], GRASP_Z + ABOVE]);
      const over = [goal[0], goal[1], GRASP_Z + ABOVE];
      yield* go('carry', over);
      yield* go('place', [goal[0], goal[1], PLACE_Z]);
      data.ctrl[7] = OPEN; yield* hold(40);
      yield* go('retreat', over);
      for (let t = 0; t < 40; t++) { physicsTick(); yield; }
      if (cfg.mem) mem.scene.set(task.key, belief);
      break;
    }
    const inGoal = b => { const p = C.bodyPos(b); return Math.abs(p[0] - goal[0]) < 0.06 && Math.abs(p[1] - goal[1]) < 0.06; };
    const result = inGoal(C.bTarget) ? 'success' : inGoal(C.bLook) ? 'wrong' : 'miss';
    st[result]++; st.n++;
    ev('result', { result, source, task });
    // S2a: 판정 '뒤에' 상자가 멈출 때까지 기다려 기울기만 기록한다 (성공 판정·통계 불변, 다음 에피소드는 reset).
    // 목표 상자 자유관절 속도 = qvel[9..14] [실측 jnt_dofadr]. 0.01 미만이 10틱 이어지면 정지, 최대 200틱.
    let still = 0, ticks = 0;
    for (; ticks < 200 && still < 10; ticks++) {
      physicsTick(); yield;
      let v = 0; for (let i = 9; i < 15; i++) v = Math.max(v, Math.abs(data.qvel[i]));
      still = v < 0.01 ? still + 1 : 0;
    }
    const tiltOf = b => tiltDegFromXmat(data.xmat.slice(9 * b, 9 * b + 9));
    const tilt = tiltOf(C.bTarget);
    ev('settle', { result, tilt, tiltLook: tiltOf(C.bLook), fallen: tilt >= FALL_DEG, settled: still >= 10, ticks });
    // S2b: 지표 S1 = 목표 칸 + 서 있음. success(S0) 는 그대로 두고 병기한다. fallen = 목표 상자가 45° 이상 (결과와 무관하게 셈)
    if (tilt >= FALL_DEG) st.fallen++;
    else if (result === 'success') st.standing++;
    return result;
  }

  function tasks(swap, sceneSeed = 0) {
    const out = [];
    for (let v = 0; v < 3; v++) scenesFor(sceneSeed).forEach((sc, i) => out.push({ idx: out.length, key: i, visit: v, place: swap && v > 0 ? { target: sc.look, look: sc.target } : sc }));
    return out;
  }
  const newStats = () => ({ n: 0, success: 0, wrong: 0, miss: 0, calls: 0, replays: 0, rejected: 0, caught: 0, dropped: 0, unsafe: 0, steps: 0, memSteps: 0, memDeleted: 0, notReached: 0, blocked: 0, standing: 0, fallen: 0 });
  function runBatch(cfg, seed = 7) {
    const st = newStats(), mem = { scene: new Map(), act: new Map() };
    for (const t of tasks(cfg.swap, cfg.sceneSeed)) { const g = episode(t, cfg, mem, st, seed); while (!g.next().done); }
    return st;
  }
  return { episode, tasks, runBatch, newStats, gateCheck };   // gateCheck: 오프라인 재생 분석용 (R2, 상태·난수 안 건드림)
}

export const SCENARIOS = [
  { id: 1, title: 'Memory cuts planner calls', sub: '18 visits without vs with step replay', cfgs: [{ gate: true, pMistake: 0.3 }, { gate: true, mem: true, pMistake: 0.3 }], labels: ['no memory', 'memory + gate'] },
  { id: 2, title: 'Memory without a gate', sub: 'what happens when nothing checks each step first', cfgs: [{ mem: true, pMistake: 0.5 }, { mem: true, gate: true, pMistake: 0.5 }], labels: ['memory, no gate', 'memory + gate'] },
  { id: 3, title: 'When the joint reading is wrong', sub: 'the gate checks the misread state and is fooled', cfgs: [{ gate: true, pMistake: 0.3 }, { gate: true, bias: true, pMistake: 0.3 }], labels: ['correct reading', 'shoulder misread +0.15 rad'] },
  { id: 4, title: 'An independent sensor', sub: 'check against the fingertip sensor instead', cfgs: [{ gate: true, bias: true, pMistake: 0.3 }, { gate: true, bias: true, sensor: true, pMistake: 0.3 }], labels: ['misread, gate only', 'misread + fingertip sensor'] },
  { id: 5, title: 'Objects get swapped', sub: 'memory points at the look-alike; a touch check catches it', cfgs: [{ gate: true, mem: true, swap: true, pMistake: 0.3 }, { gate: true, mem: true, swap: true, touch: true, pMistake: 0.3 }], labels: ['memory, no touch', 'memory + touch check'] },
  { id: 6, title: 'Memory vs camera', sub: 'drop the memory when the camera disagrees', cfgs: [{ gate: true, mem: true, swap: true, pMistake: 0.3 }, { gate: true, mem: true, swap: true, camCheck: true, pMistake: 0.3 }], labels: ['memory, no camera check', 'memory + camera check'] },
  { id: 7, title: 'Check before or after planning?', sub: 'with a wrong joint reading, both are fooled', cfgs: [{ pre: true, bias: true, pMistake: 0.3 }, { gate: true, bias: true, pMistake: 0.3 }, { gate: true, bias: true, sensor: true, pMistake: 0.3 }], labels: ['check before planning', 'check after planning', 'after planning + sensor'] },
];
