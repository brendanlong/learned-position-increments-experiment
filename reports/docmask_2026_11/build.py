# ruff: noqa: E501, RUF001
"""Build the document-masking / larger-model report from the files in data/.

    aws s3 sync s3://brendanlong-experiments/learned-position-increments/reports/docmask_2026_11/data reports/docmask_2026_11/data
    python reports/docmask_2026_11/build.py OUT_DIR

Published at https://lpi-docmask-report-cc88d3a2.surge.sh (republish there).
data/ is made by dump_char_deltas.py, dump_wandb.py and the analysis scripts'
outputs (see the page's Reproduce section).

writes OUT_DIR/body.html (the page body); render it with the html-doc
template. Every number in the page is read from data/ here.
"""

import html
import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path

DATA = Path(__file__).parent / "data"
A = DATA / "analysis"

# One color per arm, used in every figure (light-theme hex; CSS overrides
# for dark mode are in STYLE).
C_PERLAYER = "#2a78d6"
C_CONTROL = "#eb6834"
C_SHARED = "#1baf7a"
C_BLOG = "#6250d6"
C_ROPE = "#8a8984"
C_GRID = "#e1e0d9"
C_INK2 = "#52514e"

STYLE = """
<style>
.fig { margin: 1.2rem 0 1.6rem; }
.fig svg { display: block; width: 100%; height: auto; }
.fig figcaption { color: var(--ink-2); font-size: 0.9rem; margin-top: 0.4rem; }
.fig .wide { overflow-x: auto; }
.fig .wide svg { width: auto; max-width: none; }
.legend { display: flex; flex-wrap: wrap; gap: 0.3rem 1.1rem; font-size: 0.85rem; color: var(--ink-2); margin: 0.2rem 0 0.4rem; }
.legend span::before { content: ""; display: inline-block; width: 0.75em; height: 0.75em; border-radius: 50%; margin-right: 0.35em; vertical-align: -0.05em; background: var(--c); }
svg text { fill: var(--ink-2); }
svg text.ink { fill: var(--ink); }
svg .grid { stroke: var(--grid); }
svg .link { stroke: var(--grid); }
svg circle[stroke="#f9f9f7"] { stroke: var(--page); }
.num { text-align: right; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) svg [fill="#2a78d6"] { fill: #3987e5; }
  :root:where(:not([data-theme="light"])) svg [stroke="#2a78d6"] { stroke: #3987e5; }
  :root:where(:not([data-theme="light"])) svg [fill="#6250d6"] { fill: #9085e9; }
  :root:where(:not([data-theme="light"])) svg [stroke="#6250d6"] { stroke: #9085e9; }
}
:root[data-theme="dark"] svg [fill="#2a78d6"] { fill: #3987e5; }
:root[data-theme="dark"] svg [stroke="#2a78d6"] { stroke: #3987e5; }
:root[data-theme="dark"] svg [fill="#6250d6"] { fill: #9085e9; }
:root[data-theme="dark"] svg [stroke="#6250d6"] { stroke: #9085e9; }
</style>
"""

# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

MODELS = {
    # key: (label, ablation file stem, boundary file, entity file, color)
    "perlayer_s42": ("Small per-layer, seed 42", "docmask_perlayer_s42"),
    "perlayer_s1": ("Small per-layer, seed 1", "docmask_perlayer_s1"),
    "perlayer_s2": ("Small per-layer, seed 2", "docmask_perlayer_s2"),
    "control": (
        "Small per-layer, no doc mask (control)",
        "nodoc_mask_perlayer_unbounded_s42",
    ),
    "blog_perlayer": ("Blog-post replica, per-layer (capped, no mask)", "old_perlayer"),
    "shared": ("Small shared", "docmask_shared_s42"),
    "blog_shared": ("Blog-post replica, shared (no mask)", "old_shared"),
    "big_s42": ("Large per-layer, seed 42", "big_docmask_perlayer_s42"),
    "big_s1": ("Large per-layer, seed 1", "big_docmask_perlayer_s1"),
}
COLOR = {
    "perlayer_s42": C_PERLAYER,
    "perlayer_s1": C_PERLAYER,
    "perlayer_s2": C_PERLAYER,
    "big_s42": C_PERLAYER,
    "big_s1": C_PERLAYER,
    "control": C_CONTROL,
    "blog_perlayer": C_BLOG,
    "blog_shared": C_BLOG,
    "shared": C_SHARED,
}


def label(key: str) -> str:
    return MODELS[key][0]


def parse_ablation(stem: str) -> dict:
    text = (A / f"ablation_{stem}.txt").read_text()
    speed = [
        float(x) for x in re.search(r"^speed\s+(.*)$", text, re.M).group(1).split()
    ]
    rows = {}
    for m in re.finditer(
        r"^(learned|all layers = \w+|L\d+ = \w+)\s+([\d.]+) \(([+-][\d.]+)\)",
        text,
        re.M,
    ):
        rows[m.group(1)] = float(m.group(3))
    return {"speed": speed, "rows": rows}


def boundary_section(key: str) -> str:
    stem = MODELS[key][1]
    if key == "blog_perlayer":
        text = (A / "boundary_old_and_shared.txt").read_text()
        return text[text.index("PER-LAYER") :]
    if key == "shared":
        text = (A / "boundary_old_and_shared.txt").read_text()
        return text[: text.index("PER-LAYER")]
    return (A / f"boundary_{stem}.txt").read_text()


def parse_boundary(key: str) -> dict:
    text = boundary_section(key)

    def num(name: str) -> float:
        return float(
            re.search(rf"^\s+{re.escape(name)}\s+([\d.]+)", text, re.M).group(1)
        )

    layers = [
        float(x)
        for x in re.findall(r"^\s+(?:L\d+|shared) delta\s+([\d.]+)", text, re.M)
    ]
    return {
        "layers": layers,
        "entropy": num("next-byte entropy"),
        "surprisal": num("next-char surprisal"),
        "uncertainty": num("uncertainty"),
        "delta": num("all-layer log delta"),
        "both": num("uncertainty + log delta"),
        "gaps": int(re.search(r"gaps=(\d+)", text).group(1)),
    }


def parse_entity(key: str) -> dict:
    stem = MODELS[key][1]
    if key == "blog_perlayer":
        text = (A / "entity_old_and_shared.txt").read_text()
        text = text[text.index("PER-LAYER") :]
    else:
        text = (A / f"entity_{stem}.txt").read_text()
    lp = re.search(r"log p\(second word\): real ([-\d.]+), reversed ([-\d.]+)", text)
    rows = []
    for m in re.finditer(
        r"^(L\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.e+-]+)\s+([\d.e+-]+)$",
        text,
        re.M,
    ):
        rows.append((m.group(1), float(m.group(5)), float(m.group(6))))
    return {"lp_real": float(lp.group(1)), "lp_rev": float(lp.group(2)), "rows": rows}


def parse_per_byte_shared() -> dict:
    text = (A / "boundary_s42.txt").read_text()
    text = text[text.index("SHARED:") : text.index("Part 2")]
    out = {}
    for m in re.finditer(r"^(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)$", text, re.M):
        out[m.group(1).strip()] = (
            float(m.group(2)),
            float(m.group(3)),
            float(m.group(4)),
        )
    return out


# --------------------------------------------------------------------------
# SVG helpers
# --------------------------------------------------------------------------


def esc(s: object) -> str:
    return html.escape(str(s))


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def legend(items: list[tuple[str, str]]) -> str:
    spans = "".join(f'<span style="--c:{c}">{esc(t)}</span>' for t, c in items)
    return f'<div class="legend">{spans}</div>'


def figure(
    svg: str,
    caption: str,
    items: list[tuple[str, str]] | None = None,
    wide: bool = False,
) -> str:
    leg = legend(items) if items else ""
    inner = f'<div class="wide">{svg}</div>' if wide else svg
    return (
        f'<figure class="fig">{leg}{inner}<figcaption>{caption}</figcaption></figure>'
    )


def log_ticks(lo: float, hi: float) -> list[float]:
    return [
        10.0**e
        for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
        if lo <= 10.0**e <= hi
    ]


def tick_label(v: float) -> str:
    return f"{v:g}"


def line_panel(
    series: list[dict],
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    xmax: float,
    ylo: float,
    yhi: float,
    log: bool,
    title: str,
    yticks: list[float],
    ylabel_fmt: Callable[[float], str] = tick_label,
) -> str:
    def sy(v: float) -> float:
        if log:
            v = max(v, ylo)
            t = (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo))
        else:
            t = (v - ylo) / (yhi - ylo)
        return y0 + h - t * h

    def sx(v: float) -> float:
        return x0 + v / xmax * w

    out = [
        f'<text class="ink" x="{x0}" y="{y0 - 10}" font-size="13">{esc(title)}</text>'
    ]
    for t in yticks:
        y = sy(t)
        out.append(
            f'<line class="grid" x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{C_GRID}"/>'
        )
        out.append(
            f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{ylabel_fmt(t)}</text>'
        )
    for k in range(0, int(xmax) + 1, 10000):
        out.append(
            f'<text x="{sx(k):.1f}" y="{y0 + h + 16}" font-size="11" text-anchor="middle">{k // 1000}k</text>'
        )
    for s in series:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s["points"])
        dash = ' stroke-dasharray="5 3"' if s.get("dash") else ""
        out.append(
            f'<polyline fill="none" stroke="{s["color"]}" stroke-width="2"{dash} points="{pts}"><title>{esc(s["name"])}</title></polyline>'
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_delta_max(wb: dict) -> str:
    W, H = 720, 300
    panels = [
        (
            "Small model (6 layers)",
            [
                ("control", C_CONTROL, False),
                ("blog_perlayer", C_BLOG, True),
                ("perlayer_s42", C_PERLAYER, False),
                ("perlayer_s1", C_PERLAYER, False),
                ("perlayer_s2", C_PERLAYER, False),
            ],
        ),
        (
            "Large model (30 layers)",
            [("big_s42", C_PERLAYER, False), ("big_s1", C_PERLAYER, False)],
        ),
    ]
    parts = []
    pw = 285
    for i, (title, runs) in enumerate(panels):
        series = [
            {
                "name": label(k),
                "color": c,
                "dash": d,
                "points": [(s, v) for s, v in wb[k]["delta_max"] if v is not None],
            }
            for k, c, d in runs
        ]
        parts.append(
            line_panel(
                series,
                x0=50 + i * (pw + 60),
                y0=30,
                w=pw,
                h=220,
                xmax=50000,
                ylo=1,
                yhi=200,
                log=True,
                title=title,
                yticks=[1, 10, 100],
            )
        )
    svg = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Largest learned increment during training">{"".join(parts)}<text x="{W / 2}" y="{H - 2}" font-size="11" text-anchor="middle">training step</text></svg>'
    items = [
        ("per-layer, document-masked (each line a seed)", C_PERLAYER),
        ("no document mask, uncapped (control)", C_CONTROL),
        ("blog-post replica (capped at 10, no mask)", C_BLOG),
    ]
    return figure(
        svg,
        "Largest increment in a fixed logged batch of training windows, by step (log scale). Hover a line for the run.",
        items,
    )


def eval_points(wb: dict, key: str) -> dict[int, float]:
    return {int(s): v for s, v in wb[key]["val_loss"] if v is not None}


def fig_val_loss(wb: dict) -> tuple[str, dict]:
    panels = [
        (
            "Small model, vs RoPE",
            "rope",
            [
                ("shared", C_SHARED),
                ("perlayer_s42", C_PERLAYER),
                ("perlayer_s1", C_PERLAYER),
                ("perlayer_s2", C_PERLAYER),
            ],
        ),
        (
            "Large model, vs RoPE",
            "big_rope",
            [("big_s42", C_PERLAYER), ("big_s1", C_PERLAYER)],
        ),
    ]
    final = {}
    parts = []
    pw = 285
    for i, (title, base, runs) in enumerate(panels):
        b = eval_points(wb, base)
        series = []
        for k, c in runs:
            p = eval_points(wb, k)
            pts = sorted((s, p[s] - b[s]) for s in p if s in b and s >= 5000)
            final[k] = (p[max(p)], b[max(b)])
            series.append({"name": label(k), "color": c, "points": pts})
        series.append(
            {
                "name": "RoPE",
                "color": C_ROPE,
                "dash": True,
                "points": [(5000, 0), (50000, 0)],
            }
        )
        parts.append(
            line_panel(
                series,
                x0=55 + i * (pw + 60),
                y0=30,
                w=pw,
                h=200,
                xmax=50000,
                ylo=-0.015,
                yhi=0.015,
                log=False,
                title=title,
                yticks=[-0.01, 0, 0.01],
                ylabel_fmt=lambda v: f"{v:+.2f}" if v else "0",
            )
        )
    svg = f'<svg viewBox="0 0 720 270" role="img" aria-label="Validation loss difference from RoPE">{"".join(parts)}<text x="360" y="268" font-size="11" text-anchor="middle">training step</text></svg>'
    items = [
        ("per-layer (each line a seed)", C_PERLAYER),
        ("shared", C_SHARED),
        ("RoPE baseline", C_ROPE),
    ]
    return figure(
        svg,
        "Validation loss (nats per byte) minus the RoPE baseline's at the same step, from step 5k. Below zero is better than RoPE.",
        items,
    ), final


def spacing_svg(
    text: str,
    rows: list[list[float]],
    row_labels: list[str],
    *,
    char_w: float = 11.5,
    row_h: float = 30,
    font: int = 14,
) -> str:
    n = len(text)
    width = n * char_w
    xs_rows = []
    for r in rows:
        mean = sum(r) / len(r)
        pos, acc = [], 0.0
        for d in r:
            acc += d / mean
            pos.append(acc)
        lo, hi = pos[0], pos[-1]
        xs_rows.append([70 + (p - lo) / (hi - lo) * width for p in pos])
    H = len(rows) * row_h + 10
    W = 70 + width + 20
    out = []
    for j in range(n):
        if text[j] == " ":
            continue
        for i in range(len(rows) - 1):
            out.append(
                f'<line class="link" x1="{xs_rows[i][j]:.1f}" y1="{12 + i * row_h + 6}" x2="{xs_rows[i + 1][j]:.1f}" y2="{12 + (i + 1) * row_h - font}" stroke="{C_GRID}" stroke-width="1"/>'
            )
    for i, xs in enumerate(xs_rows):
        y = 12 + i * row_h + 4
        out.append(f'<text x="4" y="{y}" font-size="11">{esc(row_labels[i])}</text>')
        for j, ch in enumerate(text):
            if ch == " ":
                continue
            out.append(
                f'<text class="ink" x="{xs[j]:.1f}" y="{y}" font-size="{font}" text-anchor="middle">{esc(ch)}<title>{esc(ch)}: increment {rows[i][j]:.3g}</title></text>'
            )
    return f'<svg viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" role="img" aria-label="Characters placed at learned positions">{"".join(out)}</svg>'


def fig_speed(abl: dict) -> str:
    keys = [
        "perlayer_s42",
        "perlayer_s1",
        "perlayer_s2",
        "control",
        "blog_perlayer",
        "big_s42",
        "big_s1",
    ]
    row_h, x0, w = 84, 290, 400
    out = []
    lo, hi = 0.001, 10
    for r, k in enumerate(keys):
        sp = abl[k]["speed"]
        y0 = 18 + r * row_h

        def sy(v: float, y0: float = y0) -> float:
            t = (math.log10(max(v, lo)) - math.log10(lo)) / (
                math.log10(hi) - math.log10(lo)
            )
            return y0 + 62 - t * 62

        out.append(
            f'<text class="ink" x="0" y="{y0 + 34}" font-size="12">{esc(label(k))}</text>'
        )
        for t in [0.001, 0.01, 0.1, 1, 10]:
            out.append(
                f'<line class="grid" x1="{x0}" x2="{x0 + w}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{C_GRID}"/>'
            )
            out.append(
                f'<text x="{x0 - 5}" y="{sy(t) + 3:.1f}" font-size="9" text-anchor="end">{t:g}</text>'
            )
        n = len(sp)
        for i, v in enumerate(sp):
            x = x0 + (i + 0.5) / n * w
            out.append(
                f'<circle cx="{x:.1f}" cy="{sy(v):.1f}" r="4" fill="{COLOR[k]}"><title>{esc(label(k))}, layer {i}: speed {v:.3g}</title></circle>'
            )
    y = 18 + len(keys) * row_h
    out.append(
        f'<text x="{x0 + w / 2}" y="{y + 6}" font-size="11" text-anchor="middle">layer (left = first, right = last; 6 layers small, 30 large)</text>'
    )
    svg = f'<svg viewBox="0 0 700 {y + 12}" role="img" aria-label="Per-layer speed">{"".join(out)}</svg>'
    return figure(
        svg,
        "Each layer's speed: the geometric mean of its increments over 819k validation bytes (log scale; 1 = plain RoPE's spacing). Hover a dot for its value.",
    )


def fig_ablation(abl: dict) -> str:
    keys = [
        "perlayer_s42",
        "perlayer_s1",
        "perlayer_s2",
        "control",
        "blog_perlayer",
        "shared",
        "blog_shared",
        "big_s42",
        "big_s1",
    ]
    conds = [
        ("every layer set to 1 (integer RoPE positions)", "all layers = 1", C_CONTROL),
        (
            "every layer set to its own speed (pattern removed)",
            "all layers = speed",
            C_PERLAYER,
        ),
        ("worst single layer set to its speed", None, C_SHARED),
    ]
    x0, w, row_h = 300, 380, 26
    lo, hi = 0.001, 3

    def sx(v: float) -> float:
        return (
            x0
            + (math.log10(max(v, lo)) - math.log10(lo))
            / (math.log10(hi) - math.log10(lo))
            * w
        )

    out = []
    for t in [0.001, 0.01, 0.1, 1]:
        out.append(
            f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="10" y2="{10 + len(keys) * row_h}" stroke="{C_GRID}"/>'
        )
        out.append(
            f'<text x="{sx(t):.1f}" y="{24 + len(keys) * row_h}" font-size="11" text-anchor="middle">{t:g}</text>'
        )
    for r, k in enumerate(keys):
        y = 10 + r * row_h + row_h / 2
        out.append(
            f'<text class="ink" x="0" y="{y + 4}" font-size="12">{esc(label(k))}</text>'
        )
        rows = abl[k]["rows"]
        single = {
            name: v
            for name, v in rows.items()
            if name.startswith("L") and name.endswith("speed")
        }
        worst = max(single, key=single.get)
        for name, key, c in conds:
            v = rows[key] if key else single[worst]
            what = name if key else f"{worst.split()[0]} set to its speed"
            out.append(
                f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="#f9f9f7" stroke-width="1.5"><title>{esc(label(k))}: {esc(what)}: +{v:.3f} nats</title></circle>'
            )
    out.append(
        f'<text x="{x0 + w / 2}" y="{40 + len(keys) * row_h}" font-size="11" text-anchor="middle">validation loss increase, nats per byte (log scale)</text>'
    )
    svg = f'<svg viewBox="0 0 700 {48 + len(keys) * row_h}" role="img" aria-label="Loss increase from replacing increments">{"".join(out)}</svg>'
    return figure(
        svg,
        "Loss increase when the learned increments are replaced at inference time (819k validation bytes). For shared models, per-layer rows replace the shared increments in that layer's attention only. Hover a dot for its value.",
        [(n, c) for n, _, c in conds],
    )


def fig_auc(bd: dict) -> str:
    keys = [
        "perlayer_s42",
        "perlayer_s1",
        "perlayer_s2",
        "control",
        "blog_perlayer",
        "big_s42",
        "big_s1",
    ]
    conds = [
        ("model's own uncertainty", "uncertainty", C_ROPE),
        ("increments (all layers)", "delta", C_PERLAYER),
        ("both", "both", C_SHARED),
    ]
    x0, w, row_h = 300, 380, 26
    lo, hi = 0.5, 1.0

    def sx(v: float) -> float:
        return x0 + (v - lo) / (hi - lo) * w

    out = []
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        out.append(
            f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="10" y2="{10 + len(keys) * row_h}" stroke="{C_GRID}"/>'
        )
        out.append(
            f'<text x="{sx(t):.1f}" y="{24 + len(keys) * row_h}" font-size="11" text-anchor="middle">{t:.1f}</text>'
        )
    for r, k in enumerate(keys):
        y = 10 + r * row_h + row_h / 2
        out.append(
            f'<text class="ink" x="0" y="{y + 4}" font-size="12">{esc(label(k))}</text>'
        )
        for name, key, c in conds:
            v = bd[k][key]
            out.append(
                f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="#f9f9f7" stroke-width="1.5"><title>{esc(label(k))}: {esc(name)}: AUC {v:.3f}</title></circle>'
            )
    out.append(
        f'<text x="{x0 + w / 2}" y="{40 + len(keys) * row_h}" font-size="11" text-anchor="middle">ROC-AUC against jieba word boundaries (0.5 = chance)</text>'
    )
    svg = f'<svg viewBox="0 0 700 {48 + len(keys) * row_h}" role="img" aria-label="Word boundary AUC">{"".join(out)}</svg>'
    return figure(
        svg,
        f"Cross-validated AUC of a logistic regression (5 folds grouped by 512-byte window) separating jieba word boundaries from non-boundaries at {bd['big_s42']['gaps']:,} gaps between Chinese characters. Uncertainty = next-byte entropy and next-character surprisal before the gap. Hover a dot for its value.",
        [(n, c) for n, _, c in conds],
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    wb = json.loads((DATA / "wandb.json").read_text())
    for run in wb.values():  # scan_history repeats some rows
        for key in ("val_loss", "delta_max"):
            run[key] = sorted(
                dict((s, v) for s, v in run[key] if v is not None).items()
            )
    chars = json.loads((DATA / "char_deltas.json").read_text())
    blog = json.loads((DATA / "blog_post.json").read_text())
    abl = {k: parse_ablation(v[1]) for k, v in MODELS.items()}
    bd = {
        k: parse_boundary(k)
        for k in [
            "perlayer_s42",
            "perlayer_s1",
            "perlayer_s2",
            "control",
            "blog_perlayer",
            "big_s42",
            "big_s1",
        ]
    }
    ent = {
        k: parse_entity(k)
        for k in [
            "perlayer_s42",
            "perlayer_s1",
            "perlayer_s2",
            "control",
            "blog_perlayer",
            "big_s42",
            "big_s1",
        ]
    }
    per_byte = parse_per_byte_shared()

    peak = {
        k: max(v for _, v in wb[k]["delta_max"] if v is not None)
        for k in [
            "control",
            "perlayer_s42",
            "perlayer_s1",
            "perlayer_s2",
            "big_s42",
            "big_s1",
        ]
    }
    small_keys = ["perlayer_s42", "perlayer_s1", "perlayer_s2"]
    big_keys = ["big_s42", "big_s1"]
    val_fig, final = fig_val_loss(wb)
    gain = {k: bd[k]["both"] - bd[k]["uncertainty"] for k in bd}
    all1 = {k: abl[k]["rows"]["all layers = 1"] for k in abl}
    allspeed = {k: abl[k]["rows"]["all layers = speed"] for k in abl}

    def rng(vals: list[float], d: int = 3) -> str:
        return f"{min(vals):.{d}f}–{max(vals):.{d}f}"

    body = [STYLE]
    body.append(f"""
<p>Follow-up to <a href="{blog["source"]}">How Far Apart Does a Model Think Its Tokens Are?</a>
Three changes since the post: attention no longer crosses document boundaries, per-layer
increments are no longer capped at 10, and the same experiment is repeated on a model about 20× larger.
Code: <a href="https://github.com/brendanlong/learned-position-increments-experiment/pull/3">PR #3</a>.</p>

<h2>Summary</h2>
<ul>
<li><b>Document masking removes the early blow-up.</b> Without it, an uncapped small per-layer model's largest increment reached {peak["control"]:.0f}; with it, the three seeds peaked at {rng([peak[k] for k in small_keys], 0)}. The 30-layer models still reached {rng([peak[k] for k in big_keys], 0)} without destabilizing training.</li>
<li><b>Still loss-neutral.</b> Final validation loss is within {max(abs(a - b) for a, b in final.values()):.4f} nats/byte of RoPE in every run (small: 1 RoPE seed, 3 per-layer seeds, 1 shared; large: 1 RoPE seed, 2 per-layer seeds).</li>
<li><b>But the models depend on their increments.</b> Forcing integer RoPE positions at inference costs {rng([all1[k] for k in small_keys], 2)} nats in the small per-layer models and {rng([all1[k] for k in big_keys], 2)} in the large ones. Keeping each layer's average speed but removing the per-token pattern costs {rng([allspeed[k] for k in small_keys], 2)} (small) and {rng([allspeed[k] for k in big_keys], 3)} (large).</li>
<li><b>Every per-layer model has near-position-free layers.</b> Some layers' typical increment falls below 0.1, and they put very large gaps at sentence ends and newlines. Which layers do this changes from seed to seed.</li>
<li><b>Word boundaries: the small models' increments mostly restate the model's own uncertainty; the large models' don't.</b> Adding the increments to next-byte uncertainty raises Chinese word-boundary AUC by {rng([gain[k] for k in small_keys])} in the small models and {rng([gain[k] for k in big_keys])} in the large ones, where the increments alone ({rng([bd[k]["delta"] for k in big_keys])}) beat uncertainty alone ({rng([bd[k]["uncertainty"] for k in big_keys])}).</li>
<li><b>Caveats.</b> One RoPE seed per size, two large-model seeds. The large classifier sees 30 increment features against 6. The multi-word-entity effect mostly disappears once the model's own expectation is accounted for.</li>
</ul>

<h2>Setup</h2>
<p>Byte-level transformers trained on a mix of English and Chinese Wikipedia, as in the post. A small MLP reads each
layer's input and emits a positive increment per byte; positions are the running sum and feed RoPE. Two sizes:</p>
<ul>
<li><b>Small</b>: 6.4M parameters (256 wide, 6 layers), 512-byte windows, 60k documents, 50k steps. The post's configuration.</li>
<li><b>Large</b>: 122M parameters, shaped like SmolLM2-135M (576 wide, 30 layers, QK-norm), 1024-byte windows, 300k documents, 50k steps.</li>
</ul>
<p>Every run except the control and the blog-post replicas uses <b>document-masked attention</b> (no attention across the
end-of-document byte) and <b>uncapped</b> increments. The <b>control</b> is a small per-layer model with uncapped increments and no
document mask. The <b>blog-post replicas</b> are retrains of the post's models from the repo: no mask, per-layer increments capped at 10.</p>
<p>A layer's <b>speed</b> is the geometric mean of its increments over validation data, and its <b>pattern</b> is each increment divided by the speed.
Scaling every increment in a layer is the same as changing that layer's RoPE frequencies, so the speed carries no per-token information.</p>
""")

    body.append("<h2>Document masking removes the early increment spike</h2>")
    body.append(
        "<p>The uncapped control spikes early in training, then settles. The masked small models never spike; the large masked models reach large values but stay bounded.</p>"
    )
    body.append(fig_delta_max(wb))

    body.append("<h2>Learned increments are still loss-neutral</h2>")
    body.append(val_fig)
    rows = "".join(
        f"<tr><td>{esc(label(k))}</td><td class='num'>{final[k][0]:.4f}</td><td class='num'>{final[k][1]:.4f}</td><td class='num'>{final[k][0] - final[k][1]:+.4f}</td></tr>"
        for k in final
    )
    body.append(
        f"<details><summary>Final validation loss</summary><table><tr><th>run</th><th>val loss</th><th>RoPE</th><th>difference</th></tr>{rows}</table></details>"
    )

    body.append(
        "<h2>The shared model's per-byte increments keep the post's ordering</h2>"
    )
    body.append(
        "<p>Mean increment by byte category for the shared model (one increment per byte value). The post's model had no document mask; the new one does.</p>"
    )
    trs = "".join(
        f"<tr><td>{esc(cat)}</td><td class='num'>{blog['shared_per_byte_mean'][cat]:.2f}</td><td class='num'>{per_byte[cat][2]:.2f}</td></tr>"
        for cat in blog["shared_per_byte_mean"]
    )
    body.append(
        f"<table><tr><th>byte category</th><th>blog post</th><th>document-masked</th></tr>{trs}</table>"
    )
    body.append(
        "<p>Under document masking the end-of-document byte starts a new document, so its increment shifts every later position equally and has no effect on attention.</p>"
    )

    t = chars["texts"]
    m = chars["models"]
    body.append("<h2>Character spacing: one row per layer</h2>")
    body.append(
        "<p>Each character is placed at its learned position, so the gap before a character is its increment. Each row is scaled to the same width, so compare spacing within a row. Hover a character for its increment.</p>"
    )
    body.append("<h3>Shared model</h3>")
    body.append(
        figure(
            spacing_svg(t["en"], m["shared"]["en"], ["shared"]),
            "Document-masked shared model, English.",
            wide=True,
        )
    )
    body.append(
        figure(
            spacing_svg(t["zh"], m["shared"]["zh"], ["shared"], char_w=24, font=16),
            "Document-masked shared model, Chinese (a character's increment is the sum over its three bytes).",
            wide=True,
        )
    )
    body.append("<h3>Small per-layer model</h3>")
    labels6 = [f"L{i}" for i in range(6)]
    body.append(
        figure(
            spacing_svg(t["en"], m["perlayer_s42"]["en"], labels6),
            "Small per-layer model, seed 42.",
            wide=True,
        )
    )
    body.append(
        figure(
            spacing_svg(t["zh"], m["perlayer_s42"]["zh"], labels6, char_w=24, font=16),
            "Small per-layer model, seed 42, Chinese. jieba segments this as 人工智能 / 是 / 计算机科学 / 的 / 一个 / 分支 / 。",
            wide=True,
        )
    )
    other = "".join(
        figure(spacing_svg(t["en"], m[k]["en"], labels6), label(k), wide=True)
        for k in ["perlayer_s1", "perlayer_s2", "control", "blog_perlayer"]
    )
    body.append(
        f"<details><summary>Other small per-layer models</summary>{other}</details>"
    )

    body.append("<h3>Large per-layer model</h3>")
    big = m["big_s42"]["en2"]
    sp = abl["big_s42"]["speed"]
    slow = [i for i, v in enumerate(sp) if v < 0.12]
    chosen = sorted({0, 1, 2, 29, *slow})
    body.append(
        f"<p>Two sentences, so the slow layers' sentence-boundary gap is visible. Shown: the first three layers, the last, and every layer with speed below 0.12 ({', '.join(f'L{i}' for i in slow)}). In those layers characters within a sentence pile up at nearly the same position.</p>"
    )
    body.append(
        figure(
            spacing_svg(
                t["en2"], [big[i] for i in chosen], [f"L{i}" for i in chosen], char_w=9
            ),
            "Large per-layer model, seed 42.",
            wide=True,
        )
    )
    allbig = "".join(
        figure(
            spacing_svg(
                t["en2"], m[k]["en2"], [f"L{i}" for i in range(30)], char_w=10, row_h=26
            ),
            label(k),
            wide=True,
        )
        for k in big_keys
    )
    body.append(
        f"<details><summary>All 30 layers, both large seeds</summary>{allbig}</details>"
    )

    body.append(
        "<h2>Each per-layer model has near-position-free layers, in different places per seed</h2>"
    )
    body.append(fig_speed(abl))
    pats = []
    for k in small_keys + big_keys:
        text = (A / f"ablation_{MODELS[k][1]}.txt").read_text()
        nl = [
            float(x)
            for x in re.search(r"^pattern: Newline\s+(.*)$", text, re.M)
            .group(1)
            .split()
        ]
        sp_k = abl[k]["speed"]
        slow_k = [i for i, v in enumerate(sp_k) if v < 0.12]
        pats.append(
            f"<tr><td>{esc(label(k))}</td><td>{', '.join(f'L{i} ({sp_k[i]:.3f})' for i in slow_k)}</td><td>{', '.join(f'{nl[i]:.0f}×' for i in slow_k)}</td></tr>"
        )
    body.append(
        f"<details open><summary>Slow layers (speed below 0.12) and their newline pattern</summary><table><tr><th>model</th><th>slow layers (speed)</th><th>newline increment ÷ layer speed</th></tr>{''.join(pats)}</table></details>"
    )

    body.append(
        "<h2>The models rely on their increments, mostly on each layer's speed</h2>"
    )
    body.append(fig_ablation(abl))

    def worst(k: str) -> tuple[str, float]:
        single = {
            n.split()[0]: v
            for n, v in abl[k]["rows"].items()
            if n.startswith("L") and n.endswith("speed")
        }
        name = max(single, key=single.get)
        return name, single[name]

    body.append(
        "<p>Most costly single-layer pattern: "
        + "; ".join(
            f"{esc(label(k))} {worst(k)[0]} (+{worst(k)[1]:.3f})"
            for k in small_keys + big_keys
        )
        + ".</p>"
    )
    trs = "".join(
        f"<tr><td>{esc(label(k))}</td><td class='num'>{all1[k]:+.3f}</td><td class='num'>{allspeed[k]:+.3f}</td></tr>"
        for k in MODELS
    )
    body.append(
        f"<details><summary>Table</summary><table><tr><th>model</th><th>all = 1</th><th>all = speed</th></tr>{trs}</table></details>"
    )

    body.append(
        "<h2>In the large models, increments carry word-boundary information beyond the model's uncertainty</h2>"
    )
    body.append(
        "<p>Chinese has no spaces between words, so a word boundary between two Chinese characters isn't marked in the bytes. Next-byte uncertainty is the standard unsupervised signal for it.</p>"
    )
    body.append(fig_auc(bd))
    trs = "".join(
        f"<tr><td>{esc(label(k))}</td><td class='num'>{bd[k]['entropy']:.3f}</td><td class='num'>{bd[k]['uncertainty']:.3f}</td><td class='num'>{bd[k]['delta']:.3f}</td><td class='num'>{bd[k]['both']:.3f}</td><td class='num'>{gain[k]:+.3f}</td><td>{max(range(len(bd[k]['layers'])), key=lambda i: abs(bd[k]['layers'][i] - 0.5))}: {bd[k]['layers'][max(range(len(bd[k]['layers'])), key=lambda i: abs(bd[k]['layers'][i] - 0.5))]:.3f}</td></tr>"
        for k in bd
    )
    body.append(
        f"<details><summary>Table, including entropy alone and the most informative single layer</summary><table><tr><th>model</th><th>entropy</th><th>uncertainty</th><th>increments</th><th>both</th><th>gain</th><th>layer furthest from 0.5</th></tr>{trs}</table><p>The post reported per-layer AUCs of {', '.join(f'{v:.2f}' for v in blog['perlayer_jieba_auc'])} for its per-layer model (single features, no uncertainty comparison).</p></details>"
    )

    body.append(
        "<h2>Multi-word entities: little survives the model's own expectation</h2>"
    )
    body.append(
        '<p>As in the post: the increment at the space inside a two-word entity ("New York") versus the reversed order ("York New"), 79 entities × 5 short contexts, paired Wilcoxon test. The model already predicts the second word far better in the real order, so the second test repeats the comparison on the part of the log increment that a linear fit on next-byte entropy and log p(second word) doesn\'t explain.</p>'
    )
    trs = []
    for k, e in ent.items():
        n = len(e["rows"])
        bonf = 0.05 / n
        raw = [r for r in e["rows"] if r[1] < bonf]
        beyond = [r for r in e["rows"] if r[2] < bonf]
        best = min(e["rows"], key=lambda r: r[2])
        trs.append(
            f"<tr><td>{esc(label(k))}</td><td class='num'>{e['lp_real']:.1f} / {e['lp_rev']:.1f}</td><td class='num'>{len(raw)} of {n}</td><td class='num'>{len(beyond)} of {n}</td><td>{best[0]} (p = {best[2]:.1e})</td></tr>"
        )
    body.append(
        f"<table><tr><th>model</th><th>log p(second word), real / reversed</th><th>layers significant (Bonferroni)</th><th>…beyond uncertainty</th><th>smallest p beyond uncertainty</th></tr>{''.join(trs)}</table>"
    )

    body.append(f"""
<h2>Reproduce</h2>
<p>Code: branch <code>brendanlong/doc-mask-unbounded-deltas</code> (PR #3). Results write-up: <code>experiments/selective_pe/RESULTS.md</code>, phases 13–14.
wandb project <code>brendanlong-com/byte-pe</code>; final checkpoints are the artifacts <code>selective-pe-&lt;run name&gt;:v0</code>.</p>
<pre>
# small (the post's config; add --per-layer-delta, or --pos-encoding rope; --no-doc-mask for the control)
uv run python -m experiments.selective_pe.train --dataset bilingual --tokenizer byte \\
  --pos-encoding selective_rope --max-stories 60000 --val-max-docs 2000 --context-len 512 \\
  --generate-n 10000000 --total-steps 50000 --batch-size 64 --lr 1e-3 --weight-decay 0.01 \\
  --warmup-steps 500 --save-checkpoint --seed 42

# large
uv run python -m experiments.selective_pe.train --dataset bilingual --tokenizer byte \\
  --pos-encoding selective_rope --per-layer-delta \\
  --dim 576 --n-heads 9 --n-layers 30 --intermediate-dim 1536 \\
  --max-stories 300000 --val-max-docs 2000 --context-len 1024 --generate-n 10000000 \\
  --total-steps 50000 --batch-size 32 --lr 6e-4 --weight-decay 0.1 --warmup-steps 1000 \\
  --save-every-steps 2000 --save-checkpoint --seed 42

# analyses (per checkpoint)
uv run python -m experiments.selective_pe.delta_ablation CKPT [--context-len 1024]
uv run python -m experiments.selective_pe.byte_boundary_analysis --per-layer CKPT
uv run python -m experiments.selective_pe.entity_grouping_analysis --per-layer CKPT

# this page
python reports/docmask_2026_11/build.py OUT   # reads reports/docmask_2026_11/data/
</pre>
<table><tr><th>run</th><th>wandb id</th></tr>{"".join(f"<tr><td>{esc(v['name'])}</td><td><code>{v['id']}</code></td></tr>" for v in wb.values())}<tr><td>repro_byte_selective_rope_bilingual_256d_8h_6L_lr1e3 (blog-post shared replica)</td><td><code>q19navjs</code></td></tr></table>
""")
    (out_dir / "body.html").write_text("\n".join(body))


if __name__ == "__main__":
    main()
