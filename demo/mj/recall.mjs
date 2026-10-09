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
