// V1(10/9): 상자 '카메라' 오라클(참 위치 + 3 mm 잡음)을 대신하는 이미지 기반 비전. Node 와 브라우저가 같은 파일을 쓴다.
//   GL 렌더러 없이 mj_ray 로 고정 카메라의 깊이 + 색 이미지를 만들고(로봇 팔 가림 포함), 잡음을 넣은 뒤 이미지만으로 상자를 찾는다.
//   ⚠️ 렌더러가 돌려주는 geom 번호는 그 화소의 '겉색'(geom_rgba)을 칠하는 데만 쓴다. 어느 상자가 목표인지는 번호가 아니라
//      측정한 윗면의 짧은 변 폭(목표 32 mm · 닮은 상자 40 mm)으로만 가른다 — 번호로 가르면 새 오라클이 된다 (Kimi 지적, 10/9 패널).
//   처리: 화소 → 3D 점 → 상자 후보(탁자 위 높이 + 채도 높은 색) → 4-이웃 연결 덩어리 → 윗면 점의 최소 넓이 외접 사각형 → 짧은/긴 변 폭 · 중심 · 방향(짧은 변 축)
export const VISION_DEFAULT = {
  pos: [0.53, -0.42, 0.72], look: [0.53, 0.09, 0.03],   // 카메라 위치·바라보는 점 (작업 영역 x 0.45~0.60 · y 0~0.18 을 비스듬히 위에서)
  fovy: 25, W: 128, H: 128,                             // 화소 하나 ≈ 2.6 mm (거리 0.75 m)
  sigD: 0.002, pDrop: 0.03, sigC: 0.05,                 // 깊이 가우스 σ [m] · 화소 탈락 확률 · 색 잡음 σ
  zMin: 0.012, zMax: 0.09, satMin: 0.25, minPx: 12, topBand: 0.006, shortMax: 0.036, rectMin: 0.004,
};
const gauss = rng => Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], norm = a => Math.hypot(a[0], a[1], a[2]);
const unit = a => { const n = norm(a); return [a[0] / n, a[1] / n, a[2] / n]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function makeVision(mj, model, data, opt = {}) {
  const V = { ...VISION_DEFAULT, ...opt };
  const fwd = unit(sub(V.look, V.pos)), right = unit(cross(fwd, [0, 0, 1])), up = cross(right, fwd);
  const th = Math.tan((V.fovy * Math.PI / 180) / 2), gid = new mj.IntBuffer(1), nrm = new mj.DoubleBuffer(3);
  const dirs = [];
  for (let r = 0; r < V.H; r++) for (let c = 0; c < V.W; c++) {
    const u = ((c + 0.5) / V.W * 2 - 1) * th, v = (1 - (r + 0.5) / V.H * 2) * th;
    dirs.push(unit([fwd[0] + u * right[0] + v * up[0], fwd[1] + u * right[1] + v * up[1], fwd[2] + u * right[2] + v * up[2]]));
  }
  function capture(rng) {   // → 화소마다 { p: 3D 점 | null, rgb } (잡음 포함). 지금 물리 상태(data)를 그대로 본다
    mj.mj_forward(model, data);
    return dirs.map(d => {
      const dist = mj.mj_ray(model, data, V.pos, d, null, 1, -1, gid, nrm), g = gid.GetView()[0];
      if (dist < 0 || g < 0 || rng() < V.pDrop) return { p: null, rgb: null };
      const dn = dist + V.sigD * gauss(rng);
      const rgb = [0, 1, 2].map(k => Math.min(1, Math.max(0, model.geom_rgba[4 * g + k] + V.sigC * gauss(rng))));
      return { p: [V.pos[0] + dn * d[0], V.pos[1] + dn * d[1], V.pos[2] + dn * d[2]], rgb };
    });
  }
  function detect(img) {   // → 상자 후보 목록 [{ c:[x,y], top, short, long, yaw, n }]
    const isBox = img.map(px => px.p && px.p[2] > V.zMin && px.p[2] < V.zMax && Math.max(...px.rgb) - Math.min(...px.rgb) > V.satMin);
    const lab = new Int32Array(img.length).fill(-1), out = [];
    for (let s = 0; s < img.length; s++) {
      if (!isBox[s] || lab[s] >= 0) continue;
      const stack = [s], mem = []; lab[s] = out.length;
      while (stack.length) {
        const i = stack.pop(); mem.push(i);
        const r = (i / V.W) | 0, c = i % V.W;
        for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const rr = r + dr, cc = c + dc, j = rr * V.W + cc;
          if (rr >= 0 && rr < V.H && cc >= 0 && cc < V.W && isBox[j] && lab[j] < 0) { lab[j] = out.length; stack.push(j); }
        }
      }
      if (mem.length < V.minPx) { out.push(null); continue; }
      const zTop = Math.max(...mem.map(i => img[i].p[2]));
      const top = mem.map(i => img[i].p).filter(p => p[2] > zTop - V.topBand);
      const mx = top.reduce((s, p) => s + p[0], 0) / top.length, my = top.reduce((s, p) => s + p[1], 0) / top.length;
      // 최소 넓이 외접 사각형: 각도를 2° 씩 훑어 두 축 폭의 곱이 가장 작은 방향 (주성분은 32×40 처럼 정사각에 가까우면 축이 흔들려 폭이 부푼다 — 실측 36.5 mm)
      const ext = a => { const e = top.map(p => (p[0] - mx) * Math.cos(a) + (p[1] - my) * Math.sin(a)).sort((x, y) => x - y);
        return e[Math.floor(0.98 * (e.length - 1))] - e[Math.floor(0.02 * (e.length - 1))]; };
      let best = null;
      for (let k = 0; k < 45; k++) { const a = k * Math.PI / 90, e1 = ext(a), e2 = ext(a + Math.PI / 2); if (!best || e1 * e2 < best.area) best = { area: e1 * e2, a, e1, e2 }; }
      const a0 = best.a;   // 검수(Flash) 반영: 가장 좋은 각도 ±2° 를 0.5° 간격으로 다시 훑는다 (2° 양자화)
      for (let k = -4; k <= 4; k++) { const a = a0 + k * Math.PI / 360, e1 = ext(a), e2 = ext(a + Math.PI / 2); if (e1 * e2 < best.area) best = { area: e1 * e2, a, e1, e2 }; }
      const [short, long, shortAng] = best.e1 < best.e2 ? [best.e1, best.e2, best.a] : [best.e2, best.e1, best.a + Math.PI / 2];
      let yaw = shortAng % Math.PI; if (yaw > Math.PI / 2) yaw -= Math.PI; if (yaw < -Math.PI / 2) yaw += Math.PI;
      out.push({ c: [mx, my], top: zTop, short, long, yaw, n: mem.length, nTop: top.length });
    }
    return out.filter(Boolean);
  }
  function project(p) {   // 월드 점 → 화소 (열, 행). 데모의 '로봇이 보는 것' 패널이 찾은 상자를 그리는 데 쓴다
    const d = sub(p, V.pos), z = d[0] * fwd[0] + d[1] * fwd[1] + d[2] * fwd[2];
    const u = (d[0] * right[0] + d[1] * right[1] + d[2] * right[2]) / z / th, v = (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / z / th;
    return [(u + 1) / 2 * V.W - 0.5, (1 - v) / 2 * V.H - 0.5];
  }
  function frameOf(img, cands) {   // 화면용 RGBA (잡음 포함 그대로, 미스·탈락 화소는 검정) + 찾은 상자의 꼭짓점(화소)
    const rgba = new Uint8ClampedArray(V.W * V.H * 4);
    img.forEach((px, i) => { if (px.rgb) { rgba[4 * i] = 255 * px.rgb[0]; rgba[4 * i + 1] = 255 * px.rgb[1]; rgba[4 * i + 2] = 255 * px.rgb[2]; } rgba[4 * i + 3] = 255; });
    const boxes = cands.map(b => { const a = b.yaw, ca = Math.cos(a), sa = Math.sin(a), hs = b.short / 2, hl = b.long / 2;
      return { short: b.short, long: b.long, corners: [[hs, hl], [-hs, hl], [-hs, -hl], [hs, -hl]].map(([x, y]) => project([b.c[0] + x * ca - y * sa, b.c[1] + x * sa + y * ca, b.top])) }; });
    return { W: V.W, H: V.H, rgba, boxes };
  }
  function look(rng) {   // → 목표 추정 { ok, xy, yaw, cands }. 목표 = 짧은 변이 가장 짧은 후보, 단 짧은 변 < shortMax 이고 직사각(긴 변 − 짧은 변 > rectMin)일 때만
    //   검수 반영: 문턱 하나(34 mm)는 양쪽 여유가 1.6 mm 뿐이었다 → 폭 + 모양 두 기준으로 (목표 32×40 직사각 · 닮은 40×40 정사각)
    const img = capture(rng), cands = detect(img), tgt = cands.slice().sort((a, b) => a.short - b.short).find(b => b.short < V.shortMax && b.long - b.short > V.rectMin);
    const frame = V.frame ? frameOf(img, cands) : null;   // 데모 전용 (opt.frame) — 실험 경로는 만들지 않는다
    return tgt ? { ok: true, xy: tgt.c, yaw: tgt.yaw, short: tgt.short, cands, frame, pick: cands.indexOf(tgt) } : { ok: false, xy: null, yaw: 0, cands, frame, pick: -1 };
  }
  return { capture, detect, look, project, V };
}

// L2 S2(10/9): 놓을 곳 위 '장면 카메라' — 위에서 거의 수직으로 내려다보는 두 번째 고정 카메라의 깊이만으로 벽 윗면·받침 윗면을 잰다.
//   색·geom 번호는 쓰지 않는다 (geom 번호로 벽/받침을 가르면 새 오라클 — 10/9 패널 Kimi). 가르는 기준은 잰 크기뿐:
//   긴 변 ≥ wallLong 이면 벽, 두 변이 supMin~supMax 이면 받침. 윗면 높이는 덩어리 높이의 상위 백분위(잡음 σ 2 mm 에 최댓값은 위로 치우친다).
//   에피소드 시작 때(팔이 시야 밖 집 자세) 한 번 본다. zMaxScene 위의 점(팔·손)은 버린다.
export const SCENE_VISION_DEFAULT = {
  pos: [0.47, -0.28, 0.95], look: [0.47, -0.26, 0.0],   // 놓을 곳 구역(x 0.22~0.72 · y −0.53~−0.03 남짓)을 위에서
  fovy: 30, W: 128, H: 128,                             // 화소 하나 ≈ 4 mm
  sigD: 0.002, pDrop: 0.03, sigC: 0.05,
  zMin: 0.012, zMaxScene: 0.34, minPx: 12, topQ: 0.9, topBand: 0.008,
  wallLong: 0.15, supMin: 0.04, supMax: 0.085,
};
export function makeSceneVision(mj, model, data, opt = {}) {
  const S = { ...SCENE_VISION_DEFAULT, ...opt };
  const cam = makeVision(mj, model, data, S);   // 같은 레이캐스트 카메라 (위치·시야만 다르다)
  function measure(rng) {   // → { wall: {top, c, short, long, yaw} | null, support: {xy, top, short, long} | null, blobs }
    const img = cam.capture(rng), W = S.W, H = S.H;
    const up = img.map(px => px.p && px.p[2] > S.zMin && px.p[2] < S.zMaxScene);
    const lab = new Int32Array(img.length).fill(-1), blobs = [];
    for (let s = 0; s < img.length; s++) {
      if (!up[s] || lab[s] >= 0) continue;
      const stack = [s], mem = []; lab[s] = blobs.length;
      while (stack.length) {
        const i = stack.pop(); mem.push(i);
        const r = (i / W) | 0, c = i % W;
        for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const rr = r + dr, cc = c + dc, j = rr * W + cc;
          if (rr >= 0 && rr < H && cc >= 0 && cc < W && up[j] && lab[j] < 0) { lab[j] = blobs.length; stack.push(j); }
        }
      }
      if (mem.length < S.minPx) { blobs.push(null); continue; }
      const zs = mem.map(i => img[i].p[2]).sort((a, b) => a - b), zTop = zs[Math.floor(S.topQ * (zs.length - 1))];
      const top = mem.map(i => img[i].p).filter(p => Math.abs(p[2] - zTop) < S.topBand);
      const mx = top.reduce((a, p) => a + p[0], 0) / top.length, my = top.reduce((a, p) => a + p[1], 0) / top.length;
      const ext = a => { const e = top.map(p => (p[0] - mx) * Math.cos(a) + (p[1] - my) * Math.sin(a)).sort((x, y) => x - y);
        return e[Math.floor(0.98 * (e.length - 1))] - e[Math.floor(0.02 * (e.length - 1))]; };
      let best = null;
      for (let k = 0; k < 45; k++) { const a = k * Math.PI / 90, e1 = ext(a), e2 = ext(a + Math.PI / 2); if (!best || e1 * e2 < best.area) best = { area: e1 * e2, a, e1, e2 }; }
      const [short, long, shortAng] = best.e1 < best.e2 ? [best.e1, best.e2, best.a] : [best.e2, best.e1, best.a + Math.PI / 2];
      blobs.push({ c: [mx, my], top: zTop, short, long, yaw: shortAng, n: mem.length });
    }
    const bs = blobs.filter(Boolean);
    const wall = bs.filter(b => b.long >= S.wallLong).sort((a, b) => b.n - a.n)[0] ?? null;
    const sup = bs.filter(b => b.short >= S.supMin && b.long <= S.supMax).sort((a, b) => b.n - a.n)[0] ?? null;
    return { wall: wall && { top: wall.top, c: wall.c, short: wall.short, long: wall.long, yaw: wall.yaw },
             support: sup && { xy: sup.c, top: sup.top, short: sup.short, long: sup.long }, blobs: bs.length };
  }
  return { measure, cam, S };
}
