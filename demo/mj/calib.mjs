// B′(10/9): 관절 오프셋 Δ(7) 추정 — 위치 추적(P) · IMU 중력 방향(I) · 둘 다(P+I). 엔진(온라인 B′2)과 오프라인 검증(B′1)이 같은 코드를 쓴다.
//   obs: [{ qr: 읽은 관절, pm: 잰 손끝 위치(P), gm: 잰 손 좌표계 중력 단위벡터(I) }]
//   pose(q) → { p: 손끝 위치, g: 손 좌표계 중력 방향 } (호출 쪽이 넘긴다 — 이 모듈은 MuJoCo 를 모른다)
//   보정 관절 = qr − Δ. 가우스-뉴턴(수치 야코비안) + 능선 정규화(사전 σ prior rad): 관측되지 않는 방향은 0 쪽으로
function solve(A, b) {   // 작은 대칭 양정치 선형계 (부분 피벗 가우스 소거)
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}
export function estimateOffset(obs, pose, { P = true, I = true, sigP, sigI, prior = 0.3, iters = 15 }) {
  let D = new Array(7).fill(0);
  const resid = Dv => {
    const r = [];
    for (const o of obs) {
      const f = pose(o.qr.map((v, i) => v - Dv[i]));
      if (P) for (let k = 0; k < 3; k++) r.push((o.pm[k] - f.p[k]) / sigP);
      if (I) for (let k = 0; k < 3; k++) r.push((o.gm[k] - f.g[k]) / sigI);
    }
    for (let i = 0; i < 7; i++) r.push(Dv[i] / prior);
    return r;
  };
  for (let it = 0; it < iters; it++) {
    const r0 = resid(D), J = [];
    for (let i = 0; i < 7; i++) { const Dh = D.slice(); Dh[i] += 1e-5; const r1 = resid(Dh); J.push(r1.map((v, k) => (v - r0[k]) / 1e-5)); }
    const JTJ = [...Array(7)].map((_, a) => [...Array(7)].map((_, b) => J[a].reduce((s, v, k) => s + v * J[b][k], 0)));
    const JTr = [...Array(7)].map((_, a) => J[a].reduce((s, v, k) => s + v * r0[k], 0));
    const dx = solve(JTJ, JTr); D = D.map((v, i) => v - dx[i]);
    if (Math.hypot(...dx) < 1e-7) break;
  }
  return D;
}
