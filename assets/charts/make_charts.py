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
        if v is None:   # section header row
            out.append(f'<text x="0" y="{y + bh / 2 + 9}" font-size="12" font-weight="650" fill="{MUT}">{esc(lab)}</text>')
            continue
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

# research/oracle-free.html, level 2 without the waypoint (pre-registered, held-out seeds 523/524, 18 episodes each)
grouped("of_level2_search", "Level 2, no waypoint · over the wall, out of 18 per seed", [
    ("seed 523", [12, 7, 8], None),
    ("seed 524", [9, 8, 17], None),
], [("search on wall rejection", GREY), ("+ level 1 memory", "#8e93a0"), ("+ stored detours", BLUE)], 18,
    note="First visits to new wall placements (12): median search 18,654 → 2,708 gate checks with detours.")

# research/fk-verifier.html, "What was tested" table (2-DOF planar arm, 5 trials each)
hbars("fk_error_mm", "End-effector error in mm (lower is better), 2-DOF arm", [
    ("VLM reads vs draws the arm", None, None, ""),
    ("reads angles from an image", 14.5, BLUE, "5/5 with a grid"),
    ("generates the moved arm", 93.7, GREY, "0/5"),
    ("Giving the VLM more in text", None, None, ""),
    ("without joint angles", 165.7, GREY, "0/5"),
    ("+ joint angles as text", 276.9, RED, "0/5, worse"),
], 276.9, unit=" mm")

# research/capstone.html, Results table (Kaggle T4, 10 epochs, sigma 0.3, box mAP50, seed 0)
grouped("cap_map50", "Box mAP50 · 2 × 2 ablation (σ = 0.3, seed 0)", [
    ("noisy tiles", [0.015, 0.018, 0.178, 0.182], None),
    ("clean tiles", [0.284, 0.297, 0.225, 0.231], None),
], [("baseline", GREY), ("+ distillation", "#8e93a0"), ("+ noise training", "#7aa5dc"), ("+ both (proposed)", BLUE)], 0.30,
    note="Noise training gives 97 % of the noisy-tile gain; distillation adds +0.005.")

# research/llm-wiki.html, KPI boxes (~6,000-note wiki, 56 questions, blind graders)
grouped("wiki_recall", "Hybrid retrieval vs ambient injection · ~6,000 notes, 56 questions", [
    ("retrieval recall@3 (%)", [55, 79], None),
    ("answer correctness (%)", [33, 55], None),
], [("hybrid-retrieval baseline", GREY), ("ambient injection", BLUE)], 100)

if __name__ == "__main__":
    print(sorted(p.name for p in OUT.glob("*.svg")))
