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
// V7(10/9): 관측되지 않는 방향은 고치지 않는다. 손이 아래를 향하면 손목 회전(관절 7)은 손끝 위치도 중력 방향도 바꾸지 않아
//   능선 정규화만으로는 Δ 가 지어낸 값(실측 0.137 rad)으로 흐른다. 데이터만의 정보 행렬 JᵀJ 를 고유분해해,
//   그 방향의 사후 σ(= 1/√λ)가 maxStd 보다 큰 고유벡터 성분을 Δ 에서 뺀다. maxStd 를 주지 않으면 예전과 같다 (B′1·B′2 재현)
function eigSym(A) {   // 작은 대칭 행렬의 야코비 고유분해 → { vals, vecs(열 = 고유벡터) }
  const n = A.length, a = A.map(r => r.slice()), v = A.map((_, i) => A.map((_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
    }
  }
  return { vals: a.map((r, i) => r[i]), vecs: v };
}
export function estimateOffset(obs, pose, { P = true, I = true, sigP, sigI, prior = 0.3, iters = 15, maxStd = null }) {
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
  if (maxStd) {
    const r0 = resid(D), nData = r0.length - 7, J = [];
    for (let i = 0; i < 7; i++) { const Dh = D.slice(); Dh[i] += 1e-5; const r1 = resid(Dh); J.push(r1.slice(0, nData).map((v, k) => (v - r0[k]) / 1e-5)); }
    const H = [...Array(7)].map((_, a) => [...Array(7)].map((_, b) => J[a].reduce((s, v, k) => s + v * J[b][k], 0)));
    const { vals, vecs } = eigSym(H), lamMin = 1 / (maxStd * maxStd);
    let Dk = new Array(7).fill(0);
    for (let k = 0; k < 7; k++) {
      if (vals[k] < lamMin) continue;   // 정보가 없는 방향 = 고치지 않는다
      const u = vecs.map(r => r[k]), c = u.reduce((s, v, i) => s + v * D[i], 0);
      Dk = Dk.map((v, i) => v + c * u[i]);
    }
    return Dk;
  }
  return D;
}
