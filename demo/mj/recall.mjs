// 브라우저용 기억 조회 (10/9): 실험의 memory_recall.mjs(JUHYEOK projects/mj_memory) 에서 SQLite 를 뺀 순수 함수만 같은 식으로 옮겼다.
//   pr = (p+1)/(p+r+2) · d = ‖Δ손끝→목표‖ (λ = 0) · s = pr·exp(-d/σ), σ = d_max/2 · 재생 ⇔ pr ≥ θ_R ∧ d ≤ d_R · 동점은 id 순
//   parity: node 에서 memory_recall.mjs 의 rank 와 같은 결과인지 대조했다 (parity_recall_browser.mjs).
export const POLICY = { theta_R: 0.6, theta_H: 0.4, d_R: 0.10, d_max: 0.16, lam: 0, k_max: 3 };
const passRate = (p, r) => (p + 1) / (p + r + 2);
const score = (pr, d, pol) => pr * Math.exp(-d / (pol.d_max / 2));
const modeOf = (pr, d, pol) => (pr >= pol.theta_R && d <= pol.d_R ? 'replay' : pr >= pol.theta_H && d <= pol.d_max ? 'hint' : 'plan');
const cmpId = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
export function rank(cands, now, pol = POLICY) {
  const out = [];
  for (const c of cands) {
    const t = c.ctx.tip_to_target, d = Math.hypot(now.tip_to_target[0] - t[0], now.tip_to_target[1] - t[1], now.tip_to_target[2] - t[2]);
    if (d > pol.d_max) continue;
    const pr = passRate(c.stats.p, c.stats.r);
    out.push({ id: c.id, d, pr, s: score(pr, d, pol), mode: modeOf(pr, d, pol), cand: c });
  }
  out.sort((a, b) => b.s - a.s || cmpId(a.id, b.id));
  return out.slice(0, pol.k_max);
}
// 묶음(JSON: 통과 후보 목록) → 엔진의 cfg.recall. κ: 기억 거리 d 가 남은 거리 r 의 κ 배를 넘으면 재생하지 않는다 (G0 에서 데이터가 찾은 규칙)
export function makeRecall(cands, kappa = 0.5, pol = POLICY) {
  const byPhase = new Map();
  for (const c of cands) { if (!byPhase.has(c.phase)) byPhase.set(c.phase, []); byPhase.get(c.phase).push(c); }
  return ({ phase, tip, target }) => {
    const v = [target[0] - tip[0], target[1] - tip[1], target[2] - tip[2]], r = Math.hypot(...v);
    return rank(byPhase.get(phase) ?? [], { tip_to_target: v }, pol).filter(o => o.mode === 'replay' && o.d <= kappa * r).map(o => ({ id: o.id, dtip: o.cand.act.dtip }));
  };
}

// 10/9 데모: 실행하면서 쌓이는 기억. 게이트를 통과하고 전진한 '플래너' 걸음을 그 자리에서 후보로 저장한다
//   (상황 = 손끝→목표 벡터, 행동 = 믿는 관절로 예측한 손끝 이동 — 실험의 --dtip pred 와 같은 정의). 처음 pr = 2/3 (재생 문턱 0.6 위).
//   그 기억을 재생했는데 게이트가 거절하거나 전진이 없으면 실패 1 을 더해 pr 이 0.5 로 내려가 다시 꺼내지 않는다. 시작은 빈 기억 또는 미리 준 묶음.
export function makeLiveMemory(seedCands = [], kappa = 0.5, pol = POLICY) {
  const cands = seedCands.map(c => ({ ...c, stats: { ...c.stats }, origin: 'bank' })), byId = new Map(cands.map(c => [c.id, c])), pred = new Map();
  const recall = ({ phase, tip, target }) => {
    const v = [target[0] - tip[0], target[1] - tip[1], target[2] - tip[2]], r = Math.hypot(...v);
    return rank(cands.filter(c => c.phase === phase), { tip_to_target: v }, pol).filter(o => o.mode === 'replay' && o.d <= kappa * r).map(o => ({ id: o.id, dtip: o.cand.act.dtip }));
  };
  const onEvent = (k, x) => {   // 엔진 이벤트를 받아 기억을 갱신한다 (proposal 의 예측 손끝 이동을 걸음 저장에 쓴다)
    if (k === 'proposal') pred.set(`${x.step}:${x.attempt}`, x.dtipPred);
    else if (k === 'reject' && x.src === 'mem' && byId.has(x.from)) byId.get(x.from).stats.r++;
    else if (k === 'step') {
      if (x.mem) { const c = byId.get(x.from); if (c && !(x.progressed && !x.unsafe)) c.stats.r++; else if (c) c.stats.p++; }
      else if (x.progressed && !x.unsafe) {
        const d = pred.get(`${x.step}:${x.attempt}`); if (!d) return;
        const c = { id: `call${x.call}`, phase: x.phase, ctx: { tip_to_target: x.tipToTarget }, stats: { p: 1, r: 0 }, act: { dtip: d }, origin: 'live', call: x.call };
        cands.push(c); byId.set(c.id, c);
      }
    }
  };
  return { recall, onEvent, cands, get size() { return cands.length; } };
}
