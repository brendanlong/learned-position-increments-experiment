"""Visualize learned position increments as character spacing.

Renders a short text with each character placed at its cumulative learned
position, so characters within a word pack together (small increment) and
boundaries spread apart (large increment). Increments are normalized to mean 1
per row, so spacing is relative to standard RoPE's uniform +1 (a gap wider than
average = the model expanding at a boundary, narrower = packing within a unit).

The models are byte-level, so a character's increment is the sum of its UTF-8
bytes' increments (a Chinese character is 3 bytes; an ASCII character is 1).

Two figure types:
- single row: the shared (per-token) model -- one increment per character.
- stacked rows: the per-layer model, one row per layer (each normalized to its
  own mean), with faint lines tracking each character across layers.

Usage:
    uv run python -m experiments.selective_pe.plot_increments \
        --shared <shared_ckpt> --per-layer <perlayer_ckpt> --out-dir data/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.selective_pe.config import BYTE_EOS_ID, SelectivePEModelConfig
from experiments.selective_pe.model import SelectivePETransformer

DEFAULT_EN = "Marie Curie, a Polish-born physicist, won two Nobel Prizes."
DEFAULT_ZH = "人工智能是计算机科学的一个分支。"


CJK_FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)


def find_cjk_font() -> fm.FontProperties | None:
    """Return a FontProperties that can render CJK glyphs (and Latin), or None.

    matplotlib often fails to auto-register .ttc collections, so we point
    FontProperties at a known Noto CJK file directly. Noto CJK also covers
    Latin, so it renders the English plots fine too.
    """
    for path in CJK_FONT_PATHS:
        if Path(path).exists():
            return fm.FontProperties(fname=path)
    return None


def load_model(path: str) -> tuple[SelectivePETransformer, SelectivePEModelConfig]:
    ckpt = torch.load(path, weights_only=True)
    cfg = SelectivePEModelConfig(**ckpt["model_config"])
    model = SelectivePETransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def char_increments(
    model: SelectivePETransformer,
    cfg: SelectivePEModelConfig,
    text: str,
    *,
    shared_scale: bool = False,
) -> np.ndarray:
    """Return (n_rows, n_chars) increments. n_rows=1 shared, n_layers per-layer.

    We prepend an EOS byte (the document separator the model trained with) so
    the first character is in-distribution (a document start) rather than a
    cold position-0 with no left context, which otherwise produces an anomalous
    spike in the deeper layers.

    Normalization: ``shared_scale=False`` divides each row by its own mean (so
    every row averages 1, good for a single row). ``shared_scale=True`` divides
    all rows by one global mean, preserving cross-layer magnitude differences
    (a layer that barely advances position renders as a short, compressed row).
    """
    byte_ids = [BYTE_EOS_ID, *text.encode("utf-8")]
    deltas = model.get_deltas(torch.tensor(byte_ids).unsqueeze(0))
    assert deltas is not None, "model is not a selective variant"
    rows = (deltas[:, 0, :] if cfg.per_layer_delta else deltas).cpu().numpy()
    rows = rows[:, 1:]  # drop the prepended-EOS column

    spans: list[tuple[int, int]] = []
    off = 0
    for ch in text:
        n = len(ch.encode("utf-8"))
        spans.append((off, off + n))
        off += n

    per_char = np.stack(
        [[row[a:b].sum() for a, b in spans] for row in rows]
    )  # (n_rows, n_chars)
    scale = per_char.mean() if shared_scale else per_char.mean(axis=1, keepdims=True)
    return per_char / scale


def plot_rows(
    text: str,
    increments: np.ndarray,
    row_labels: list[str],
    title: str,
    out_path: Path,
    font: fm.FontProperties | None,
) -> None:
    n_rows, n_chars = increments.shape
    # Place each character at its cumulative position p_t = sum_{s<=t} delta_s,
    # so the gap *before* a character equals its own increment (the position
    # jump the model makes when it reaches it). Anchor character 0 at x=0.
    pos = np.cumsum(increments, axis=1) - increments[:, :1]
    fig, ax = plt.subplots(figsize=(max(9.0, n_chars * 0.32), 1.4 + n_rows * 0.6))
    for r in range(n_rows):
        y = n_rows - 1 - r
        for i, ch in enumerate(text):
            ax.text(
                pos[r, i], y, ch, ha="left", va="center",
                fontsize=15, fontproperties=font,
            )
    if n_rows > 1:
        ys = [n_rows - 1 - r for r in range(n_rows)]
        for i in range(n_chars):
            ax.plot(pos[:, i], ys, color="tab:blue", alpha=0.25, lw=0.6, zorder=0)

    ax.set_yticks([n_rows - 1 - r for r in range(n_rows)])
    ax.set_yticklabels(row_labels)
    ax.set_xticks([])
    ax.set_xlim(-1, pos.max() + 2)
    ax.set_ylim(-0.6, n_rows - 0.4)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared", help="shared-delta checkpoint")
    p.add_argument("--per-layer", help="per-layer-delta checkpoint")
    p.add_argument("--en", default=DEFAULT_EN)
    p.add_argument("--zh", default=DEFAULT_ZH)
    p.add_argument("--label-en", default="english", help="label for --en")
    p.add_argument("--label-zh", default="chinese", help="label for --zh")
    p.add_argument("--out-dir", default="data/selective_pe/plots")
    p.add_argument(
        "--scale",
        choices=["per-row", "shared"],
        default="per-row",
        help="per-layer normalization: per-row (every row averages 1, legible) "
        "or shared (one scale, so low-budget layers render compressed).",
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    font = find_cjk_font()
    print(f"CJK font: {font.get_name() if font else 'NONE (Chinese -> boxes)'}")

    texts = [(args.label_en, args.en), (args.label_zh, args.zh)]

    if args.shared:
        model, cfg = load_model(args.shared)
        for tag, text in texts:
            incr = char_increments(model, cfg, text)
            plot_rows(
                text, incr, ["shared"],
                f"Learned position increments (shared model) - {tag}",
                out / f"shared_{tag}.png", font,
            )

    if args.per_layer:
        model, cfg = load_model(args.per_layer)
        labels = [f"L{i}" for i in range(cfg.n_layers)]
        shared_scale = args.scale == "shared"
        for tag, text in texts:
            incr = char_increments(model, cfg, text, shared_scale=shared_scale)
            plot_rows(
                text, incr, labels,
                f"Learned position increments per layer - {tag}",
                out / f"perlayer_{tag}_{args.scale}.png", font,
            )


if __name__ == "__main__":
    main()
