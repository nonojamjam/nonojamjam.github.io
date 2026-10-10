"""Small SVG charts for the portfolio pages, same look as assets/poc/pipeline.svg.
Every number is copied from the results table on the page named next to it; run: python3 assets/charts/make_charts.py"""
from pathlib import Path

OUT = Path(__file__).parent
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
INK, MUT, LINE, BLUE, RED, GOLD, GREY = "#1d1d1f", "#6b6b70", "#e3e3e6", "#1a5fb4", "#c0392b", "#c9a227", "#b9bcc4"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def hbars(name, title, rows, vmax, unit="", w=680, label_w=230, note=None):
    """rows: (label, value, colour, right-hand text). One bar per row."""
    bh, gap, top = 26, 12, 40
    h = top + len(rows) * (bh + gap) + (34 if note else 14)
    x0, x1 = label_w, w - 170
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}" font-size="13" fill="{INK}">',
           f'<title>{esc(title)}</title><rect width="{w}" height="{h}" fill="#fff"/>',
           f'<text x="0" y="20" font-weight="650" font-size="14">{esc(title)}</text>']
    for i, (lab, v, col, right) in enumerate(rows):
        y = top + i * (bh + gap)
        bw = max(2, (x1 - x0) * v / vmax) if v else 0
        out.append(f'<text x="{x0 - 10}" y="{y + bh / 2 + 5}" text-anchor="end">{esc(lab)}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{bh}" fill="#f6f7f9"/>')
        if bw:
            out.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="{bh}" rx="3" fill="{col}"/>')
        inside = bw > 50
        out.append(f'<text x="{(x0 + bw - 8) if inside else (x0 + bw + 8):.1f}" y="{y + bh / 2 + 5}" font-weight="700" text-anchor="{"end" if inside else "start"}" fill="{"#fff" if inside and col not in (LINE,) else INK}">{esc(v)}{esc(unit)}</text>')
        if right:
            out.append(f'<text x="{x1 + 14}" y="{y + bh / 2 + 5}" fill="{MUT}" font-size="12">{esc(right)}</text>')
    if note:
        out.append(f'<text x="0" y="{h - 10}" fill="{MUT}" font-size="11.5">{esc(note)}</text>')
    out.append("</svg>")
    (OUT / f"{name}.svg").write_text("\n".join(out))


def grouped(name, title, groups, series, vmax, w=640, note=None):
    """groups: [(group label, [values per series], [labels under values or None])]; series: [(name, colour)]."""
    top, gh, bh = 76, 40, 0
    n = len(series)
    bw, inner = 34, 10
    gw = n * bw + (n - 1) * 4
    plot_h = 170
    x_start, step = 70, (w - 90) / len(groups)
    h = top + plot_h + 52 + (20 if note else 0)
    base = top + plot_h
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}" font-size="13" fill="{INK}">',
           f'<title>{esc(title)}</title><rect width="{w}" height="{h}" fill="#fff"/>',
           f'<text x="0" y="20" font-weight="650" font-size="14">{esc(title)}</text>']
    lx = 0
    for sname, col in series:  # legend
        out.append(f'<rect x="{lx}" y="32" width="12" height="12" rx="2" fill="{col}"/><text x="{lx + 17}" y="42" font-size="12">{esc(sname)}</text>')
        lx += 34 + 7.4 * len(sname)
    out.append(f'<line x1="{x_start - 10}" y1="{base}" x2="{w}" y2="{base}" stroke="{LINE}"/>')
    for gi, (glab, vals, subs) in enumerate(groups):
        cx = x_start + step * gi + step / 2
        x = cx - gw / 2
        for si, v in enumerate(vals):
            bh = plot_h * v / vmax
            col = series[si][1]
            out.append(f'<rect x="{x:.1f}" y="{base - max(bh, 1.5):.1f}" width="{bw}" height="{max(bh, 1.5):.1f}" rx="3" fill="{col}"/>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{base - max(bh, 1.5) - 6:.1f}" text-anchor="middle" font-size="12" font-weight="700">{esc(v)}</text>')
            if subs and subs[si]:
                out.append(f'<text x="{x + bw / 2:.1f}" y="{base + 15}" text-anchor="middle" font-size="10.5" fill="{MUT}">{esc(subs[si])}</text>')
            x += bw + 4
        out.append(f'<text x="{cx:.1f}" y="{base + 36}" text-anchor="middle" font-weight="650">{esc(glab)}</text>')
    if note:
        out.append(f'<text x="0" y="{h - 6}" fill="{MUT}" font-size="11.5">{esc(note)}</text>')
    out.append("</svg>")
    (OUT / f"{name}.svg").write_text("\n".join(out))


# research/oracle-free.html, section 1 table (held-out scenes, 36 episodes)
hbars("of_level1_calls", "Large-model (32B) calls, held-out scenes · 36 episodes", [
    ("32B planner alone", 793, GREY, "36/36 placed"),
    ("4B → 32B cascade, no memory", 269, GREY, "33/36 placed"),
    ("cascade + verified memory", 0, BLUE, "36/36 placed"),
], 793, note="The memory came from the 32B planner's own runs on two other scene seeds (644 steps).")

# research/oracle-free.html, section 2 table (six unseen seeds, real LLMs, 108 episodes)
hbars("of_fault_recovery", "Shoulder reading 0.15 rad off · placed out of 108, real LLMs", [
    ("fault, no fix (stand-in planner)", 0, RED, ""),
    ("+ tracker + IMU calibration", 100, GREY, "large-model calls 36"),
    ("+ observe → verify → act", 108, BLUE, "large-model calls 13"),
    ("reference: no fault", 108, LINE, "large-model calls 11"),
], 108)

# research/oracle-free.html, level 2 table (held-out scenes, 36 episodes each)
grouped("of_level2_calls", "Level 2 · large-model calls, held-out scenes (36 episodes each)", [
    ("wall", [1245, 262, 404], ["31/36", "36/36", "34/36"]),
    ("support box", [308, 13, 2], ["36/36", "36/36", "35/36"]),
    ("both", [706, 155, 99], ["31/36", "36/36", "35/36"]),
], [("no memory", GREY), ("level 1 memory", BLUE), ("memory grown on level 2", "#7aa5dc")], 1245,
    note="Numbers under the bars: boxes placed and left standing.")

if __name__ == "__main__":
    print(sorted(p.name for p in OUT.glob("*.svg")))
