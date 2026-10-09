// 7개 시나리오를 MuJoCo 물리 위에서 돌리는 에피소드 엔진 (걸음 단위 판단). Node(일괄 통계)와 브라우저(재생)가 같은 코드를 쓴다.
// 원래 데모와 같은 구조: 손끝을 잡을 지점 위로 보내는 동안 걸음마다
//   상황 키로 기억 조회 → 없으면 플래너(LLM 대역)에게 다음 관절 이동을 묻고 → 순기구학 게이트가 검사(막히면 다시 생성)
//   → MuJoCo 가 실제로 그 걸음을 실행 → 나아갔으면 기억에 저장, 기억에서 꺼낸 걸음이 안 나아갔으면 그 기억을 지운다.
// 다 오면 내려가 실제 손가락으로 쥐고(촉각), 들어 옮겨 놓는다. 에피소드는 제너레이터: 제어 한 틱마다 yield.
import { OPEN, CLOSED } from './pickplace_ctrl.mjs';
import { estimateOffset } from './calib.mjs';
import { makeVision, makeSceneVision } from './vision.mjs';

export const TARGET_W = 0.032, LOOK_W = 0.040;     // 손가락이 닫힌 폭: 목표 상자 32 mm, 닮은 상자 40 mm
const SUB = 5, PRE = 0.08, ABOVE = 0.15, GRASP_Z = 0.035, PLACE_Z = 0.037;
const BIAS_JOINT = 1, TOUCH_TOL = 0.006, CAM_NOISE = 0.003, SENSOR_NOISE = 0.002;
const STEP = 0.06, SUCC = 0.025, PROG = 0.005, JERK = 0.4, MAX_STEPS = 12, MAX_REPLAN = 3, UNSAFE = 0.005, TILT = 0.25;
const CAL = { k: 5, window: 20, min: 10, every: 5, sigI: Math.PI / 180, prior: 0.3 };   // V7: cfg.calib = { maxStd } 이면 관측 안 되는 방향은 고치지 않는다   // B′2: 온라인 보정 — 추적 k 회 중앙값 + IMU σ 1°, 최근 20 관측, 10개부터 5개마다 재추정
const FLOOR_EPS = 0.003;   // P1(10/9): 쥔 상자를 들고 걸을 때 손끝이 PLACE_Z(상자가 탁자에 닿는 높이)보다 이만큼 아래로 내려가는 제안은 거절
// L2-자유(10/10): 손 충돌 캡슐(모델 실측: 반지름 0.04, 반길이 0.06, 손끝 7 cm 위, 축 = 손가락 여는 방향, 수평)이 벽에 닿는 수평 거리.
//   게이트가 손끝 한 점만 보면 축이 벽을 향할 때 손끝이 벽 면에서 8 cm 인데도 캡슐이 걸려 멈춘다 (대역 시험 실측: 명령 관절 변화의 ~10 % 만 실행, 30 걸음 넘게 제자리).
//   벽 축마다 반폭 = 반길이·|축·그 축| + 반지름 + 여유 5 mm. 손 방향은 지금 읽은 관절로 (한 걸음 안에서 손목은 거의 안 돈다)
const HAND_CAP = { half: 0.06, r: 0.04, pad: 0.005 };

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
  let vis = null;   // V2: 이미지 기반 비전 (cfg.vision 일 때만 만든다)
  let sceneVis = null;   // L2 S2: 놓을 곳 위 장면 카메라 (cfg.l2.measure 일 때만)
  // L2(10/9): 레벨 2 장면 — 장애물 벽(cfg.l2.wall) · 받침 상자(cfg.l2.support). pickplace_l2.mjb 의 mocap 몸체를 장면마다 옮긴다.
  //   위치·높이는 장면 seed·장면 번호로만 정한다(조건이 달라도 같은 세계). 둘 다 늘 뽑아서 l2a·l2b·l2ab 가 같은 수를 공유한다.
  //   ⚠️ 여기서 정한 참값은 채점·기록 전용이다. 결정 경로(계획·게이트·기억)는 측정값만 쓴다 — 오라클 회귀 방지 (패널 Kimi).
  const WALL_HALF = [0.18, 0.01, 0.15], SUP_HALF = [0.03, 0.03, 0.05];
  const L2_CLEAR = 0.035 + 0.04;   // 벽 넘을 때 손끝 높이 = 벽 윗면 + (쥔 점→상자 바닥 3.5 cm) + 여유 4 cm
  const mocapOf = n => { const b = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, n), m = b >= 0 ? model.body_mocapid[b] : -1; if (m < 0) throw new Error(`L2: 모델에 mocap '${n}' 없음 — pickplace_l2.mjb 를 쓰세요`); return m; };
  function placeL2(cfg, task) {
    // 씨앗을 해시로 섞는다 — 섞지 않으면 장면 번호에 따라 위치·높이가 거의 직선으로 변한다 (실측: 벽 윗면 0.245→0.233→0.222, scenesFor 와 같은 문제)
    let h = (((cfg.sceneSeed || 0) * 7919 + task.key * 104729 + 29) ^ 0x9e3779b9) >>> 0;
    h = Math.imul(h ^ (h >>> 16), 0x85ebca6b); h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35); h = (h ^ (h >>> 16)) >>> 0;
    const r = makeRng(h), u = (lo, hi) => lo + (hi - lo) * r();
    // 벽 y 는 카메라 시선 밖으로 (−0.14~−0.08 은 가까운 상자를 가렸다: 대조군 s514 비전 44회 중 14 실패·21 오차 >1 cm [실측]. 가림은 레벨 3 재료).
    //   시선 높이 ≈ 0.72 − 0.66·(y+0.42)/0.42 → 벽 앞면 y ≤ −0.16 이면 시선이 0.31 이상으로 윗면 0.26 위를 지난다. 받침은 벽과 3 cm 이상 띄운다.
    const wall = { xy: [u(0.50, 0.54), u(-0.20, -0.17)], top: u(0.20, 0.26) }, support = { xy: [u(0.40, 0.55), u(-0.38, -0.27)], top: u(0.04, 0.08) };
    const put = (n, xy, z) => { const m = mocapOf(n); data.mocap_pos[3 * m] = xy[0]; data.mocap_pos[3 * m + 1] = xy[1]; data.mocap_pos[3 * m + 2] = z; };
    if (cfg.l2.wall) put('wall', wall.xy, wall.top - WALL_HALF[2]);
    if (cfg.l2.support) put('support', support.xy, support.top - SUP_HALF[2]);
    mj.mj_forward(model, data);
    return { wall: cfg.l2.wall ? wall : null, support: cfg.l2.support ? support : null };
  }
  let kd = null;   // B′2: 보정용 운동학 전용 MjData (cfg.calib 일 때만 만든다)
  function handPose(q) {   // 손끝 위치 + 손 좌표계에서 본 중력 방향 (R 행 우선: Rᵀ·[0,0,-1] = −(셋째 행))
    kd ??= new mj.MjData(model);
    for (let i = 0; i < 7; i++) kd.qpos[i] = q[i];
    mj.mj_kinematics(model, kd);
    const R = kd.site_xmat.slice(9 * C.site, 9 * C.site + 9);
    return { p: Array.from(kd.site_xpos.slice(3 * C.site, 3 * C.site + 3)), g: [-R[6], -R[7], -R[8]] };
  }

  let capG = null;   // L2-자유: 손 충돌 캡슐 geom 번호 (처음 쓸 때 찾는다)
  function handReach(q) {   // 지금 관절에서 손 캡슐의 벽 축별 수평 반폭 [짧은 변 축, 긴 변 축] (잰 벽 yaw 기준)
    if (capG === null) { const hb = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, 'hand'); capG = -1;
      for (let g = 0; g < model.ngeom; g++) if (model.geom_bodyid[g] === hb && model.geom_type[g] === 3 && model.geom_conaffinity[g]) capG = g;
      if (capG < 0) throw new Error('L2-자유: 손 충돌 캡슐을 모델에서 못 찾음'); }
    handPose(q);   // kd 에 운동학 (geom 자세 포함)
    const R = kd.geom_xmat.slice(9 * capG, 9 * capG + 9), ax = [R[2], R[5], R[8]];
    const ca = Math.cos(handReach.yaw), sa = Math.sin(handReach.yaw);
    return [ax[0] * ca + ax[1] * sa, -ax[0] * sa + ax[1] * ca].map(c => HAND_CAP.half * Math.abs(c) + HAND_CAP.r + HAND_CAP.pad);
  }
  handReach.yaw = 0;

  // 상황 키: 목표까지 방향(x·y·z 부호, 1 cm 불감대) + 거리 구간 + 큰 관절 3개(1·2·4)의 구간
  function situationKey(tip, target, q) {
    const sg = v => (Math.abs(v) < 0.01 ? 0 : Math.sign(v));
    const dd = d3(tip, target), db = dd < 0.05 ? 0 : dd < 0.15 ? 1 : 2;
    const jb = [1, 2, 3].map(j => { const [lo, hi] = jr(j); return Math.min(2, Math.floor(3 * (q[j] - lo) / (hi - lo))); });
    return [sg(target[0] - tip[0]), sg(target[1] - tip[1]), sg(target[2] - tip[2]), db, ...jb].join(',');
  }

  // 플래너(대역): 믿는 손끝에서 목표 쪽으로 최대 6 cm 가는 관절 이동을 낸다. 확률 pm 으로 관절 두 개를 크게 틀리게 낸다
  function planStep(qB, tipB, target, cfg, st, rng, rej = null, scene = null) {   // L2 S4: scene = 잰 장애물 (진짜 LLM 프롬프트에만, 대역은 무시)   // L10: rej = 이 걸음에서 마지막으로 거절된 제안 {reason, src, dtip} (진짜 LLM 만 쓴다, 대역은 무시)
    st.calls++;
    if (cfg.planner) {   // L1: 진짜 LLM(파일럿) — 손끝 이동량 [m] 배열이나 {dq} 를 받는다. 대역의 실수 주입·난수는 쓰지 않는다
      const d = cfg.planner(scene ? { tip: tipB, target, q: qB, reject: rej, scene } : { tip: tipB, target, q: qB, reject: rej });
      if (!d) return qB.map(() => 0);   // 못 읽은 응답 = 움직이지 않음 (게이트가 전진 없음으로 거절)
      if (d.dq) return d.dq.slice();
      const f = C.fkTip(qB);
      return C.solveIK([f[0] + d[0], f[1] + d[1], f[2] + d[2]], 60, qB).q.map((x, i) => x - qB[i]);
    }
    const fk = C.fkTip(qB);
    // L2-자유 시험 전용: cfg.l2.free.standinClimb 이면 대역이 'wall' 거절 직후 — 거절된 걸음이 내려가던 것이면 높이를 지킨 수평 걸음, 아니면 위로 5 cm
    //   (면제·빠져나오기 경로 확인용. 진짜 LLM 실행에는 쓰지 않는다)
    if (cfg.l2?.free?.standinClimb && rej?.reason === 'wall') {
      const h = [target[0] - tipB[0], target[1] - tipB[1]], hn = Math.hypot(...h) || 1;
      const d = rej.dtip[2] < -0.005 ? [STEP * h[0] / hn, STEP * h[1] / hn, 0] : [0, 0, 0.05];
      return C.solveIK([fk[0] + d[0], fk[1] + d[1], fk[2] + d[2]], 60, qB).q.map((x, i) => x - qB[i]);
    }
    const v = [target[0] - tipB[0], target[1] - tipB[1], target[2] - tipB[2]];
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
  // L2 S3: 잰 벽(사각형 c·long·short·yaw=짧은 변 축)과의 여유. 지금 손끝 → 예측 손끝 선분 위 점들에서, 벽 발자국(+여유) 안이면
  //   (손끝 − below) 가 벽 윗면 + WALL_MARGIN 보다 높아야 한다. below = 쥔 상자 바닥까지(쥔 단계) 또는 손가락 끝까지.
  //   선분은 5 mm 간격 이하로 짚는다 (검수 Flash: 5점 고정이면 긴 걸음이 모서리를 건너뛸 수 있다 — 재현은 안 됐지만 비용 0)
  const WALL_MARGIN = 0.02;
  const segPts = (a, b) => { const n = Math.max(4, Math.ceil(d3(a, b) / 0.005)); return Array.from({ length: n + 1 }, (_, k) => { const t = k / n; return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])]; }); };
  function wallClear(a, b, w, below, reach = null) {   // reach = [짧은 변 축, 긴 변 축] 손 반폭 (L2-자유만, 없으면 예전 그대로)
    const ca = Math.cos(w.yaw), sa = Math.sin(w.yaw); let worst = Infinity;
    for (const p of segPts(a, b)) {
      const dx = p[0] - w.c[0], dy = p[1] - w.c[1], us = dx * ca + dy * sa, ul = -dx * sa + dy * ca;   // 짧은 변 축·긴 변 축 좌표
      if (Math.abs(us) < w.short / 2 + WALL_MARGIN + (reach ? reach[0] : 0) && Math.abs(ul) < w.long / 2 + WALL_MARGIN + (reach ? reach[1] : 0)) worst = Math.min(worst, p[2] - below - w.top);
    }
    return worst;   // Infinity = 발자국 밖
  }
  function gateCheck(qB, tipB, dq, target, floorZ = null, obst = null) {
    const q1 = qB.map((x, i) => x + dq[i]);
    const maxDq = Math.max(...dq.map(Math.abs)), handErr = C.handErr(q1);
    let margin = Infinity; for (let j = 0; j < 7; j++) { const [lo, hi] = jr(j); margin = Math.min(margin, q1[j] - lo, hi - q1[j]); }
    const f0 = C.fkTip(qB), f1 = C.fkTip(q1), dtip = [f1[0] - f0[0], f1[1] - f0[1], f1[2] - f0[2]];
    const pred = [tipB[0] + dtip[0], tipB[1] + dtip[1], tipB[2] + dtip[2]], progress = d3(tipB, target) - d3(pred, target);
    // L2: 받침 발자국(+2 cm) 위를 지나는 점은 받침 윗면이 바닥이다 (밖에서는 탁자 — 들어 올릴 때 막히지 않게).
    //   끝점만이 아니라 선분 위 점 전부 (검수 Flash: 받침 위를 낮게 지나 밖에서 끝나는 걸음)
    const overSup = p => Math.abs(p[0] - obst.sup.xy[0]) < obst.sup.half && Math.abs(p[1] - obst.sup.xy[1]) < obst.sup.half;
    const supHit = floorZ !== null && !!obst?.sup && segPts(tipB, pred).some(p => overSup(p) && p[2] < obst.sup.z);
    const floorHit = floorZ !== null && (pred[2] < floorZ || supHit);
    // 검수 Flash(2차): 손목이 돌면 반폭이 바뀐다 → 걸음 앞뒤 관절의 반폭 중 큰 쪽으로 선분을 보고, 빠져나오기 판정은 앞뒤 각각으로
    const r0 = obst?.wall && obst.reach ? obst.reach(qB) : null, r1 = r0 ? obst.reach(q1) : null;
    const reachNow = r0 ? [Math.max(r0[0], r1[0]), Math.max(r0[1], r1[1])] : null;
    const wallC = obst?.wall ? wallClear(tipB, pred, obst.wall, obst.below, reachNow) : Infinity;
    let wallHit = wallC < WALL_MARGIN;
    // L2-자유: 출발점이 이미 (넓힌) 발자국 안 여유 부족이면 — 추적 잡음·손 방향 변화로 들어올 수 있다 — 더 나빠지지 않는 걸음은 통과.
    //   더 나빠짐 = 벽 면 쪽으로 다가감(짧은 변 축 |us| 감소) 또는 내려감. 대역 시험 실측: 이 규칙 없이 바로 위 5 cm 걸음까지 'wall' 로 거절돼 갇힘
    if (wallHit && reachNow) {
      const startC = wallClear(tipB, tipB, obst.wall, obst.below, r0);
      if (startC < WALL_MARGIN) {
        // 검수 Astra(2차): 절댓값 비교는 벽 면을 가로지르는 걸음(us +2 cm → −2 cm)도 통과시킨다 → 출발한 쪽 기준 부호 있는 거리로
        const w = obst.wall, side = Math.sign((tipB[0] - w.c[0]) * Math.cos(w.yaw) + (tipB[1] - w.c[1]) * Math.sin(w.yaw)) || 1;
        const away = p => side * ((p[0] - w.c[0]) * Math.cos(w.yaw) + (p[1] - w.c[1]) * Math.sin(w.yaw));
        wallHit = away(pred) - r1[0] < away(tipB) - r0[0] - 0.001 || pred[2] < tipB[2] - 0.001;   // 손 가장자리가 벽 면에 다가가거나(넘어가거나) 내려가면 거절
      }
    }
    const reason = maxDq > JERK ? 'jerk' : handErr > TILT ? 'tilt' : margin < 0 ? 'limit' : floorHit ? 'floor' : wallHit ? 'wall' : progress > PROG ? null : 'progress';
    const values = { max_dq: maxDq, hand_err: handErr, limit_margin: margin, progress };
    if (obst?.wall && wallC < Infinity) values.wall_clear = wallC;   // L2 S3: 벽 발자국 위일 때만 기록 (L2 끄면 기록이 예전과 같다)
    if (floorZ !== null) values.pred_z = pred[2];   // 바닥 검사를 켰을 때만 (끄면 기록이 예전과 같다)
    return { ok: reason === null, reason, dtip, values };
  }
  const gateOk = (qB, tipB, dq, target) => gateCheck(qB, tipB, dq, target).ok;

  function* episode(task, cfg, mem, st, seed, ev = () => {}) {
    const base = (seed * 7919 + task.idx * 104729) >>> 0;
    const rCam = makeRng(base + 1), rPlan = makeRng(base + 2), rSen = makeRng(base + 3);
    const bias = cfg.bias ? 0.15 : 0;
    C.reset([0, 0], task.place);
    // V3: cfg.yaw = 상자 회전 범위(±도). 회전값은 장면 seed·장면 번호로만 정한다 (조건이 달라도 같은 세계). 끄면 회전 없음
    let trueYaw = 0;
    if (cfg.yaw) {
      const ry = makeRng(((cfg.sceneSeed || 0) * 7919 + task.key * 104729 + 13) >>> 0), yaws = [0, 1].map(() => (2 * ry() - 1) * cfg.yaw * Math.PI / 180);
      const [yT, yL] = task.place === null ? yaws : (cfg.swap && task.visit > 0 ? [yaws[1], yaws[0]] : yaws);
      for (const [a, t] of [[12, yT], [19, yL]]) { data.qpos[a] = Math.cos(t / 2); data.qpos[a + 1] = 0; data.qpos[a + 2] = 0; data.qpos[a + 3] = Math.sin(t / 2); }
      mj.mj_forward(model, data); trueYaw = yT;
    }
    if (cfg.l2?.oracle && cfg.l2?.measure) throw new Error('L2: oracle 과 measure 는 함께 쓸 수 없다');
    // L2-자유(10/10, 사전등록 '레벨 2-자유'): 들기·운반 경유점 없이 쥔 뒤 곧장 놓을 곳 위(놓는 높이 + 3 cm) 한 점으로 걷는다.
    //   벽은 게이트만 안다(플래너 프롬프트에 벽 없음, 거절은 제약 진술만). 측정판 + 운반 걸음 단위에서만
    const FREE = cfg.l2?.free ? { maxSteps: 40, ...(cfg.l2.free === true ? {} : cfg.l2.free) } : null;
    if (FREE && !(cfg.l2.measure && cfg.stepPhases?.includes('carry'))) throw new Error('L2-자유는 measure 와 걸음 단계 carry 가 필요하다');
    const l2Truth = cfg.l2 ? placeL2(cfg, task) : null;   // L2: 채점·기록 전용 (결정 경로에서 읽지 않는다)
    if (l2Truth) ev('l2-scene', l2Truth);
    for (let t = 0; t < 40; t++) { physicsTick(); yield; }
    // L2 S2: 측정판이면 시작할 때 장면 카메라로 한 번 본다 (관측 → 검증 → 행동). 참값은 오차 기록에만
    let l2Meas = null;
    if (cfg.l2?.measure) {
      sceneVis ??= makeSceneVision(mj, model, data, cfg.l2.measure === true ? {} : cfg.l2.measure);
      l2Meas = sceneVis.measure(makeRng(base + 7));
      const e = (m, t, f) => m && t ? f(m, t) : m ? 'false-positive' : t ? 'missed' : null;
      ev('scene-vision', { meas: l2Meas, err: {
        wallTop: e(l2Meas.wall, l2Truth.wall, (m, t) => m.top - t.top),
        supXY: e(l2Meas.support, l2Truth.support, (m, t) => Math.hypot(m.xy[0] - t.xy[0], m.xy[1] - t.xy[1])),
        supTop: e(l2Meas.support, l2Truth.support, (m, t) => m.top - t.top) } });
    }
    // L2: 운반 높이·놓을 점·놓는 높이 + 게이트·플래너에 줄 벽. 기본은 레벨 1 그대로(고정 놓을 곳, 탁자 높이, 벽 없음).
    //   cfg.l2.oracle = 의도적 대조군: 벽·받침 참값 (풀 수 있는 과제인지 + 측정판의 상한).
    //   cfg.l2.measure = 측정판: 장면 카메라가 잰 값(l2Meas). 못 찾으면 레벨 1 기본값(실패로 이어지게 — 참값으로 되돌리지 않는다)
    let carryZ = GRASP_Z + ABOVE, dst = goal, placeZ = PLACE_Z, l2Wall = null, l2Sup = null;
    const l2Src = cfg.l2?.oracle ? l2Truth : cfg.l2?.measure ? l2Meas : null;
    if (l2Src) {
      if (l2Src.wall) { carryZ = Math.max(carryZ, l2Src.wall.top + L2_CLEAR);
        l2Wall = cfg.l2.oracle ? { c: l2Src.wall.xy, top: l2Src.wall.top, long: 2 * WALL_HALF[0], short: 2 * WALL_HALF[1], yaw: Math.PI / 2 } : l2Src.wall; }
      if (l2Src.support) { dst = l2Src.support.xy; placeZ = PLACE_Z + l2Src.support.top; carryZ = Math.max(carryZ, placeZ + ABOVE);
        l2Sup = { xy: l2Src.support.xy, half: (cfg.l2.oracle ? SUP_HALF[0] : l2Src.support.long / 2) + 0.02 }; }
    }
    if (l2Wall) handReach.yaw = l2Wall.yaw;   // L2-자유: 손 반폭을 잰 벽 축으로 투영
    const truth = C.bodyPos(C.bTarget);
    const cam = () => [truth[0] + CAM_NOISE * gauss(rCam), truth[1] + CAM_NOISE * gauss(rCam)];
    // V2: cfg.vision 이면 상자 위치를 참값 + 잡음 대신 ray 렌더 이미지에서 찾는다 (vision.mjs — 목표 판별은 측정 폭·모양으로만).
    //   참값은 기록(err)에만 쓴다. 못 찾으면 작업 영역 가운데로 간다 (실패로 이어지게 — 참값으로 되돌리지 않는다). 끄면 see = cam 그대로
    const rVis = cfg.vision ? makeRng(base + 5) : null;
    let beliefYaw = 0;
    const see = () => {
      if (!cfg.vision) return cam();
      vis ??= makeVision(mj, model, data, cfg.vision === true ? {} : cfg.vision);
      const L = vis.look(rVis), now = C.bodyPos(C.bTarget);
      ev('vision', { ok: L.ok, xy: L.xy, yaw: L.yaw, short: L.short ?? null, cands: L.cands.length, err: L.ok ? Math.hypot(L.xy[0] - now[0], L.xy[1] - now[1]) : null, ...(L.frame ? { frame: L.frame, pick: L.pick } : {}) });   // frame: 데모가 cfg.vision={frame:true} 일 때만
      if (!L.ok) { st.visionMiss = (st.visionMiss || 0) + 1; return [0.525, 0.09]; }
      beliefYaw = L.yaw; return L.xy;
    };
    // V3: 비전이 없으면(가짜 카메라 대조군) 방향도 참값 + 2° 잡음 (오라클 상한). yawAlign 이 꺼져 있으면 손목은 늘 0
    if (cfg.yawAlign && !cfg.vision) beliefYaw = trueYaw + (2 * Math.PI / 180) * gauss(makeRng(base + 6));
    let belief, source;
    if (cfg.mem && mem.scene.has(task.key)) {
      belief = mem.scene.get(task.key); source = 'memory';
      if (cfg.yawAlign && belief.length > 2) { beliefYaw = belief[2]; belief = belief.slice(0, 2); }   // V8b: 기억한 상자 방향도 꺼낸다 (아무 방향 세계에서 기억을 쓰면 손목이 0° 로 돌아가던 결함)
      if (cfg.camCheck) { const c = see(); if (d2(belief, c) > 0.02) { mem.scene.delete(task.key); belief = c; source = 'camera'; st.dropped++; ev('drop'); } }
    } else { belief = see(); source = 'camera'; }
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
    // B′2: 온라인 보정. Δ̂ 는 실행(mem) 전체에 유지한다 (고정 고장). 보정 관절 = 읽은 값 − Δ̂ 를 FK·게이트·기억·명령에 쓴다.
    //   끄면 rd = readJoints(bias), mv = moveJoints(·, bias) 그대로 (S0 불변)
    if (cfg.calib && !TR) throw new Error('cfg.calib 은 cfg.tracker 가 있어야 한다');
    const cal = cfg.calib ? (mem.cal ??= { D: new Array(7).fill(0), obs: [], n: 0 }) : null;
    const rCal = cal ? makeRng(base + 4) : null;
    const rd = () => { const q = readJoints(bias); return cal ? q.map((v, i) => v - cal.D[i]) : q; };
    const mv = qB => moveJoints(cal ? qB.map((v, i) => v + cal.D[i]) : qB, bias);
    const calObserve = (force = false) => {   // V8: force = 관측 수가 차 있으면 이번에 바로 재추정 (능동 감지용)   // 한 걸음의 관측(추적 k 회 중앙값 + IMU)을 쌓고 때가 되면 Δ̂ 를 다시 추정. 반환 = 추적 중앙값과 FK(보정 관절)의 거리
      const qr = readJoints(bias), t = tipNow(), fk = C.fkTip(qr.map((v, i) => v - cal.D[i])), reads = [];
      for (let i = 0; i < CAL.k; i++) reads.push(rangeTrack(t, rCal, TR, fk));
      const pm = [0, 1, 2].map(j => reads.map(r => r[j]).sort((a, b) => a - b)[CAL.k >> 1]);
      const R = data.site_xmat.slice(9 * C.site, 9 * C.site + 9), gn = [-R[6], -R[7], -R[8]].map(v => v + CAL.sigI * gauss(rCal)), nn = Math.hypot(...gn);
      cal.obs.push({ qr, pm, gm: gn.map(v => v / nn) }); if (cal.obs.length > CAL.window) cal.obs.shift();
      cal.n++;
      if (cal.obs.length >= CAL.min && (force || cal.n % CAL.every === 0)) {
        cal.D = estimateOffset(cal.obs, handPose, { sigP: TR.sigma * 1.5, sigI: CAL.sigI, prior: CAL.prior, maxStd: cfg.calib?.maxStd ?? null });
        const truth = cal.D.map((v, i) => v - (i === BIAS_JOINT ? bias : 0));   // 시뮬이라 참 오프셋을 안다 — 기록 전용
        ev('calib', { n: cal.n, D: cal.D.slice(), err: Math.hypot(...truth) });
      }
      const fkNow = C.fkTip(qr.map((v, i) => v - cal.D[i]));   // 검수(Flash) 반영: 격리 판단은 방금 갱신된 Δ̂ 기준 잔차로
      return Math.hypot(pm[0] - fkNow[0], pm[1] - fkNow[1], pm[2] - fkNow[2]);
    };
    // V4: 보정 없이도 독립 감각 불일치를 '감시만' 한다 (cfg.monitor). 추적 k 회 중앙값과 FK(읽은 관절)의 거리. 에피소드 최대값을 기록해
    //   고장 기간 결과를 자동으로 무효 처리하는 데 쓴다 (derive_bank --auto-invalid). 보정이 켜져 있으면 calObserve 의 잔차를 쓴다
    if (cfg.monitor && !TR) throw new Error('cfg.monitor 는 cfg.tracker 가 있어야 한다');
    const rMon = cfg.monitor && !cal ? makeRng(base + 7) : null;
    const monitorDis = () => {
      const qr = readJoints(bias), t = tipNow(), fk = C.fkTip(qr), reads = [];
      for (let i = 0; i < CAL.k; i++) reads.push(rangeTrack(t, rMon, TR, fk));
      const pm = [0, 1, 2].map(j => reads.map(r => r[j]).sort((a, b) => a - b)[CAL.k >> 1]);
      return Math.hypot(pm[0] - fk[0], pm[1] - fk[1], pm[2] - fk[2]);
    };
    const monAgg = { max: 0, n: 0, over: 0 };
    let lastDis = Infinity;   // V8: 마지막 걸음의 추적-FK 불일치 (쥐기 관문이 본다)
    data.ctrl[7] = OPEN;
    // S4: phase 태그 — 추론하지 않고 제어 코드가 지금 하는 일을 그대로 적는다 (approach·grasp·lift·carry·place·retreat)
    let phase = null;
    const setPhase = p => { phase = p; ev('phase', { phase: p }); };

    // L5(E1): 경유점까지 걸음 단위로 가기 — 예전 접근 루프를 그대로 함수로 뺐다 (동작·난수 불변). 걸음 번호는 에피소드 전체에서 이어진다
    //   (재시도마다 0 으로 돌아가면 기록의 (ep, step, attempt) 짝이 겹친다)
    let stepNo = 0;
    // L2-자유: 벽 여유(수직) = 쥔 상자 바닥 − 잰 벽 윗면. 직전 거절이 'wall' 이면 이 여유를 늘리는 걸음은 진척 없어도 통과(면제),
    //   여유가 WALL_MARGIN(+2 cm) 이상이 되면 면제 소멸. 벽 기준 특징 wallRel = [벽 면까지 수평 거리(+ = 아직 넘지 않은 쪽), 여유]
    const freeHold = () => ['lift', 'carry', 'place'].includes(phase);
    const clearOf = z => z - (freeHold() ? 0.035 : 0.01) - l2Wall.top;
    const wallRel = tip => {
      if (!FREE || !l2Wall) return null;
      const ca = Math.cos(l2Wall.yaw), sa = Math.sin(l2Wall.yaw), us = p => (p[0] - l2Wall.c[0]) * ca + (p[1] - l2Wall.c[1]) * sa;
      const side = Math.sign(us(dst)) || 1;
      return [-us(tip) * side, clearOf(tip[2])];
    };
    let wallEx = false;
    function* stepTo(target, maxSteps = MAX_STEPS) {
      let reached = false;
      wallEx = false;
      for (let n = 0; n < maxSteps; n++) {
        const dis = cal ? calObserve() : rMon ? monitorDis() : 0; lastDis = dis;   // B′2: 관측은 읽기 전에 (방금 갱신된 Δ̂ 로 이 걸음을 읽는다) · V4: 감시만
        if (cfg.monitor) { monAgg.n++; monAgg.max = Math.max(monAgg.max, dis); if (dis > (cfg.monitor.delta ?? 0.05)) monAgg.over++; }
        const qB = rd(), tipB = believedTip(qB);
        if (d3(tipB, target) < SUCC) { reached = true; break; }
        if (wallEx && clearOf(tipB[2]) >= WALL_MARGIN) wallEx = false;   // L2-자유: 여유 +2 cm 에 닿으면 면제 소멸
        const step = stepNo++;
        const key = (phase === 'approach' ? '' : phase + ':') + situationKey(tipB, target, qB);   // E2: 접근 밖 단계는 키에 단계를 붙인다 (접근 기억을 운반에서 꺼내지 않게)
        // 타임라인용: 플래너 호출마다 번호(call)를 붙이고, 기억에는 그 걸음을 낸 호출 번호(from)를 함께 저장한다
        let lastRej = null;   // L10: 게이트가 마지막으로 거절한 제안 — 다음 플래너 호출에 이유를 넘긴다
        const ask = () => { const d = planStep(qB, tipB, target, cfg, st, rPlan, lastRej, l2Wall && !FREE ? { wall: l2Wall } : null); ev('call', { id: st.calls, vars: { q: qB, tip: tipB, target: target }, dq: d.slice() }); return d; };
        let dq = null, src = 'plan', call = 0, from = 0;
        // L6c: cfg.recall = 실행 밖 기억 묶음(DB). 재생 후보 [{id, dtip}] 를 점수순으로 게이트에 넣고, 없거나 다 거절되면 플래너.
        //   손끝 변위 → 지금 자세에서 IK 로 관절 이동 (R2: 관절 증분 그대로보다 쓸 수 있는 거리가 두 배). 엔진 Map 기억(cfg.mem)과 함께 쓰지 않는다
        const quarantined = !!cfg.quarantine && dis > cfg.quarantine.delta;   // B′2(D2): 독립 감각과 어긋나는 동안은 기억을 꺼내지 않는다
        if (quarantined) ev('quarantine', { dis });
        const memQ = cfg.recall && !quarantined ? cfg.recall({ phase, tip: tipB, target, q: qB, ...(FREE ? { wallRel: wallRel(tipB) } : {}) }).slice() : [];
        const nextMem = () => {
          const m = memQ.shift(), f = C.fkTip(qB);
          dq = C.solveIK([f[0] + m.dtip[0], f[1] + m.dtip[1], f[2] + m.dtip[2]], 60, qB).q.map((x, i) => x - qB[i]);
          from = m.id; src = 'mem'; call = 0; st.replays++;
        };
        if (cfg.mem && mem.act.has(key)) { const m = mem.act.get(key); dq = m.dq.slice(); from = m.from; src = 'mem'; st.replays++; }
        else if (memQ.length) nextMem();
        else { dq = ask(); call = st.calls; }
        let attemptNo = 0, gateUs = 0, lastExempt = false;   // lastExempt: 마지막 판정(= 실행할 제안)이 진척 면제로 통과했나
        const judge = () => {   // 제안 하나를 게이트에 넣고 proposal 이벤트를 남긴다 (게이트 꺼짐이면 판정 없이 기록만)
          const holding = ['lift', 'carry', 'place'].includes(phase);
          const floorZ = cfg.stepPhases && holding ? PLACE_Z - FLOOR_EPS : null;   // P1: 걸음 단계 옵션일 때 쥔 채 걷는 단계만
          const obst = l2Wall || l2Sup ? { wall: l2Wall, below: holding ? 0.035 : 0.01, sup: l2Sup && { xy: l2Sup.xy, half: l2Sup.half, z: placeZ - FLOOR_EPS }, ...(FREE ? { reach: handReach } : {}) } : null;   // L2 S3: 쥔 상자 바닥(3.5 cm 아래) 또는 손가락 끝 · 받침 위 바닥
          const t0 = performance.now(), g = cfg.gate ? gateCheck(qB, tipB, dq, target, floorZ, obst) : null; gateUs += performance.now() - t0;
          if (FREE && g && l2Wall) {   // L2-자유: 면제 판단 (다른 검사는 그대로 — 진척만 면제)
            if (g.reason === 'wall' && clearOf(tipB[2]) < WALL_MARGIN) wallEx = true;   // 검수 Astra(2차): 이미 여유 ≥ 2 cm 면 켜지 않는다 (사전등록 '여유 +2 cm 면 소멸')
            else if (g.reason === 'progress' && wallEx && clearOf(tipB[2] + g.dtip[2]) > clearOf(tipB[2]) + PROG) { g.ok = true; g.reason = null; g.values.exempt = 1; }
          }
          lastExempt = !!g?.values?.exempt;
          const dtip = g ? g.dtip : (() => { const a = C.fkTip(qB), b = C.fkTip(qB.map((x, i) => x + dq[i])); return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]; })();
          ev('proposal', { step, attempt: attemptNo++, phase, src, call, from, dq: dq.slice(), dtipPred: dtip, gate: g && { ok: g.ok, reason: g.reason, values: g.values } });
          if (g && !g.ok) lastRej = { reason: g.reason, src, dtip, ...(FREE ? { free: true, values: g.values } : {}) };   // L2-자유: 플래너는 제약 진술(여유 수치)만 받는다
          return g ? g.ok : true;
        };
        if (!cfg.gate) judge();
        if (cfg.gate) {
          let tries = 0, lastOk = false;   // 루프를 빠져나온 마지막 판정 = 예전 코드의 재검사 결과와 같다 (같은 dq 를 두 번 기록하지 않으려고 재사용)
          // L6c: DB 기억 후보가 남아 있으면 거절돼도 재계획 횟수(tries)를 쓰지 않고 다음 후보로. 후보가 없으면 예전과 똑같은 순서·횟수
          while (!(lastOk = judge())) {
            if (!memQ.length && tries++ >= MAX_REPLAN) break;
            st.rejected++; ev('reject', { q: qB.map((x, i) => x + dq[i] + (cal ? cal.D[i] : 0) - (i === BIAS_JOINT ? bias : 0)), src, call, from });
            if (src === 'mem' && cfg.mem) { mem.act.delete(key); st.memDeleted++; ev('forget', { from }); }
            if (memQ.length) nextMem();
            else { dq = ask(); call = st.calls; from = 0; src = 'plan'; }
          }
          if (!lastOk) { st.blocked++; ev('blocked'); continue; }   // 끝까지 막히면 움직이지 않고 다음 걸음에서 다시 묻는다
        }
        const realBefore = d3(tipNow(), target), qTrue0 = Array.from(data.qpos.slice(0, 7)), tip0 = tipNow();
        yield* mv(qB.map((x, i) => x + dq[i]));
        const qB2 = rd(), tipB2 = believedTip(qB2);
        const unsafe = d3(tipNow(), target) > realBefore + UNSAFE && !lastExempt;   // L2-자유: 면제 걸음(위로 비켜 감)은 멀어지는 게 정상   // 실제 손끝이 향하던 지점에서 멀어졌다
        if (unsafe) st.unsafe++;
        st.steps++; if (src === 'mem') st.memSteps++;
        const progressed = d3(tipB2, target) < d3(tipB, target) - PROG || (lastExempt && clearOf(tipB2[2]) > clearOf(tipB[2]) + PROG);   // L2-자유: 면제 걸음은 여유가 늘면 전진으로 친다
        if (cfg.mem) {
          if (src === 'plan' && progressed) { mem.act.set(key, { dq, from: call }); ev('store', { call }); }
          if (src === 'mem' && !progressed) { mem.act.delete(key); st.memDeleted++; ev('forget', { from }); }
        }
        const tip1 = tipNow(), qTrue1 = Array.from(data.qpos.slice(0, 7));
        ev('step', { src: unsafe ? 'unsafe' : src, mem: src === 'mem', tip: tip1, belief: tipB2, key, call, from, phase,
          step, attempt: attemptNo - 1, qRead: qB, qTrue: qTrue0, tipToTarget: [target[0] - tipB[0], target[1] - tipB[1], target[2] - tipB[2]],
          dqCmd: dq.slice(), dqActual: qTrue1.map((v, i) => v - qTrue0[i]), dtipActual: [tip1[0] - tip0[0], tip1[1] - tip0[1], tip1[2] - tip0[2]],
          distDelta: d3(tip1, target) - realBefore, progressed, unsafe, gateUs, ...(FREE ? { exempt: lastExempt, wallRel: wallRel(tipB) } : {}) });
      }
      if (!reached) { const qB = rd(); reached = d3(believedTip(qB), target) < SUCC; }
      return reached;
    }

    for (let attempt = 0; attempt < 3; attempt++) {
      setPhase('approach');
      const pre = [belief[0], belief[1], GRASP_Z + PRE];
      ev('reach-start', { tip: tipNow(), belief: believedTip(rd()) });
      // ---- 걸음 단위로 잡을 지점 위까지 ----
      const reached = yield* stepTo(pre);
      if (!reached) { st.notReached++; ev('not-reached'); break; }
      // V8(10/9): 관측 → 검증 → 행동. 쥐기(되돌리기 어려움) 전에 보정이 준비됐고(관측 ≥ CAL.min) 독립 감각과 어긋나지 않는지(불일치 ≤ δ) 본다.
      //   아니면 쥐기 대신 잡을 점 주변 몇 곳으로 움직여 관측을 쌓고 그때마다 다시 추정한다 (능동 감지). 실측 동기: 고장 + 보정 실패의 대부분이 첫 판(관측 10개 전 쥐기 확정)
      if (cal && cfg.calibGate) {
        const delta = cfg.calibGate.delta ?? 0.05, ready = () => cal.obs.length >= CAL.min && lastDis <= delta;
        const OFF = [[0.04, 0, 0], [-0.04, 0, 0], [0, 0.04, 0], [0, -0.04, 0], [0, 0, 0.04], [0.03, 0.03, 0.02], [-0.03, -0.03, 0.02], [0, 0, 0]];
        const before = { n: cal.obs.length, dis: lastDis }; let probes = 0;
        while (!ready() && probes < OFF.length) {
          const o = OFF[probes++];
          yield* mv(C.solveIK([pre[0] + o[0], pre[1] + o[1], pre[2] + o[2]], 150, rd()).q);
          lastDis = calObserve(true);
        }
        if (probes) { st.probes = (st.probes || 0) + probes; ev('active-sense', { probes, ready: ready(), before, n: cal.obs.length, dis: lastDis }); }
      }

      // ---- 정렬해서 내려가 쥐기 (읽은 관절값 기준 IK). 센서가 있으면 잰 손끝 오차를 매번 다시 재서 고친다 ----
      let off = [0, 0, 0];
      const corrected = function* (pt) {
        yield* mv(C.solveIK([pt[0] + off[0], pt[1] + off[1], pt[2] + off[2]], 150, rd()).q);
        if (!cfg.sensor) return;
        for (let k = 0; k < 3; k++) {
          const m = sense(rd()), e = [pt[0] - m[0], pt[1] - m[1], pt[2] - m[2]];
          if (Math.hypot(...e) < 0.004) return;
          off = [off[0] + e[0], off[1] + e[1], off[2] + e[2]]; ev('sensor-fix');
          yield* mv(C.solveIK([pt[0] + off[0], pt[1] + off[1], pt[2] + off[2]], 150, rd()).q);
        }
      };
      const g = [belief[0], belief[1], GRASP_Z];
      setPhase('grasp');
      if (cfg.yawAlign) { C.setYaw(beliefYaw); ev('yaw-align', { belief: beliefYaw, truth: trueYaw }); }   // V3: 쥐기 전에 손목을 짧은 변 축에 맞춘다
      yield* corrected([g[0], g[1], g[2] + PRE]);
      yield* corrected([g[0], g[1], g[2] + 0.03]);
      yield* corrected(g);
      data.ctrl[7] = CLOSED; yield* hold(60);
      const width = data.qpos[7] + data.qpos[8];
      ev('touch', { width, tip: tipNow(), belief: C.fkTip(rd()) });   // belief = 읽은 관절값으로 계산한 손끝 (센서와 무관)
      if (cfg.touch && Math.abs(width - TARGET_W) > TOUCH_TOL) {
        st.caught++; ev('caught', { width });
        data.ctrl[7] = OPEN; yield* hold(40);
        yield* corrected([g[0], g[1], g[2] + ABOVE]);
        mem.scene.delete(task.key); belief = see(); continue;
      }
      // ---- 들어서 옮겨 놓기 ----
      // L5(E2): cfg.stepPhases 에 든 단계는 경유점까지 걸음 단위(플래너·게이트·기억)로 간 뒤 IK 로 마무리 정렬한다. 없으면 예전처럼 IK 한 번.
      //   단계 표지를 먼저 세운 뒤 걷는다 (setPhase 누락 시 이전 단계 이름이 찍힌다 — E1 검수 지적)
      const go = function* (p, pt) {
        setPhase(p);
        if (cfg.stepPhases?.includes(p) && !(yield* stepTo(pt))) ev('phase-not-reached', { phase: p });
        yield* corrected(pt);
      };
      if (FREE) {
        // L2-자유: 들기·운반 경유점 없이 놓을 곳 위 한 점으로. 못 닿으면 IK 마무리를 하지 않는다 — 직선 IK 가 벽을 뚫거나
        //   '코드가 대신 넘겨 주는' 길이 되지 않게. 그 자리에서 놓고 끝(실패로 채점된다)
        setPhase('carry');
        const above = [dst[0], dst[1], placeZ + 0.03];
        const ok = yield* stepTo(above, FREE.maxSteps);
        if (!ok) {   // 못 닿았어도 손이 이미 벽을 넘어(벽 발자국 + 손 반폭 바깥, 놓을 곳 쪽) 있으면 IK 마무리 — 남은 길에 벽이 없다. 아니면 그 자리에서 놓고 끝
          ev('phase-not-reached', { phase: 'carry' });
          const qN = rd(), crossed = wallRel(believedTip(qN))[0] < -(l2Wall.short / 2 + WALL_MARGIN + handReach(qN)[0]);
          ev('free-finish', { crossed });
          if (!crossed) { data.ctrl[7] = OPEN; yield* hold(40); break; }
        }
        yield* corrected(above);
        yield* go('place', [dst[0], dst[1], placeZ]);
        data.ctrl[7] = OPEN; yield* hold(40);
        yield* go('retreat', [dst[0], dst[1], placeZ + ABOVE]);
      } else {
      yield* go('lift', [g[0], g[1], carryZ]);
      const over = [dst[0], dst[1], carryZ];
      yield* go('carry', over);
      yield* go('place', [dst[0], dst[1], placeZ]);
      data.ctrl[7] = OPEN; yield* hold(40);
      yield* go('retreat', over);
      }
      for (let t = 0; t < 40; t++) { physicsTick(); yield; }
      if (cfg.mem) mem.scene.set(task.key, cfg.yawAlign ? [belief[0], belief[1], beliefYaw] : belief);   // V8b: 방향도 함께 (정렬 끔이면 예전 그대로)
      break;
    }
    // L2: 받침이 있으면 성공 = 상자 중심이 받침 윗면 안쪽 위 (참값 채점). 없으면 예전 그대로 놓을 곳 12 cm 칸
    const onSupport = p => { const s = l2Truth.support; return Math.abs(p[0] - s.xy[0]) < SUP_HALF[0] && Math.abs(p[1] - s.xy[1]) < SUP_HALF[1] && p[2] > s.top + 0.02; };
    const inGoal = b => { const p = C.bodyPos(b); return l2Truth?.support ? onSupport(p) : Math.abs(p[0] - goal[0]) < 0.06 && Math.abs(p[1] - goal[1]) < 0.06; };
    const result = inGoal(C.bTarget) ? 'success' : inGoal(C.bLook) ? 'wrong' : 'miss';
    st[result]++; st.n++;
    if (cfg.monitor) ev('monitor', { ...monAgg });   // V4: 이 에피소드의 독립 감각 불일치 요약
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
    else if (result === 'success' && (!l2Truth?.support || onSupport(C.bodyPos(C.bTarget)))) st.standing++;   // L2: 받침 세계는 멈춘 뒤에도 받침 위여야
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
