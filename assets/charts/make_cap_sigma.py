"""cap_sigma.svg — box mAP50 vs evaluation noise σ (thesis Fig. 3, English, site style).
Data = same sources as thesis figures/src/fig3_sigma_curve_v3.py:
  σ0.1–0.4 eval_ci46, σ0 eval_ci_20261027 (seed 0) / eval_ci45/desktop_orig_20261105 (seed 1), BL0 σ0.3 eval_ci_20261027."""
import json, pathlib
R = pathlib.Path.home() / 'JUHYEOK/projects/kd_research/results/v23'
g = lambda p, k: json.loads((R / p).read_text())['summary'][k]['recomputed_map50']
v46 = lambda m, ts, sg: g(f'eval_ci46/{m}_s{ts}_sig{sg}_n0_summary.json', 'noisy')
clean = {('S0', 0): g('eval_ci_20261027/S0_summary.json', 'clean'), ('X1', 0): g('eval_ci_20261027/X1_summary.json', 'clean'),
         ('S0', 1): g('eval_ci45/desktop_orig_20261105/S0_s1_summary.json', 'clean'),
         ('X1', 1): g('eval_ci45/desktop_orig_20261105/X1_s1_summary.json', 'clean')}
SG = ['0.1', '0.2', '0.3', '0.4']; XS = [0.0] + [float(s) for s in SG]
series = [('no noise training', '#c0392b', '', [g('eval_ci_20261027/BL0_summary.json', 'clean')] +
           [v46('BL0', 0, s) if s != '0.3' else g('eval_ci_20261027/BL0_summary.json', 'noisy') for s in SG])]
for m, name, col in (('S0', 'noise training only', '#8a8d94'), ('X1', 'centred spatial KD', '#1a5fb4')):
    for ts in (0, 1):
        series.append((f'{name}, seed {ts}', col, '' if ts == 0 else '5 4', [clean[(m, ts)]] + [v46(m, ts, s) for s in SG]))

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
INK, MUT, LINE = "#1d1d1f", "#6b6b70", "#e3e3e6"
W, H, L, Rm, T, B = 640, 360, 54, 20, 34, 52
px = lambda x: L + x / 0.4 * (W - L - Rm)
py = lambda y: T + (0.7 - y) / 0.7 * (H - T - B)
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}" font-size="12" fill="{INK}">',
     f'<text x="0" y="18" font-weight="650" font-size="14">Box mAP50 vs evaluation noise σ (training noise fixed at 50/255)</text>']
for y in (0, .1, .2, .3, .4, .5, .6, .7):
    o.append(f'<line x1="{L}" x2="{W - Rm}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="{LINE}"/>')
    o.append(f'<text x="{L - 8}" y="{py(y) + 4:.1f}" text-anchor="end" fill="{MUT}" font-size="11">{y:.1f}</text>')
for x in XS:
    o.append(f'<text x="{px(x):.1f}" y="{H - B + 18}" text-anchor="middle" fill="{MUT}" font-size="11">{x:g}</text>')
o.append(f'<text x="{(L + W - Rm) / 2}" y="{H - 14}" text-anchor="middle" fill="{MUT}" font-size="11.5">evaluation noise σ (0 = clean)</text>')
for name, col, dash, ys in series:
    pts = ' '.join(f'{px(x):.1f},{py(y):.1f}' for x, y in zip(XS, ys))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
    for x, y in zip(XS, ys):
        o.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{col}"/>')
lx, ly = px(0.255), py(0.69)
for i, (name, col, dash, _) in enumerate(series):
    yy = ly + i * 17
    o.append(f'<line x1="{lx}" x2="{lx + 24}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="2"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
    o.append(f'<text x="{lx + 30}" y="{yy + 4:.1f}" font-size="11.5">{name}</text>')
o.append('</svg>')
out = pathlib.Path(__file__).with_name('cap_sigma.svg'); out.write_text('\n'.join(o))
for name, _, _, ys in series: print(f'{name:28s}', [round(y, 3) for y in ys])
print(out)
