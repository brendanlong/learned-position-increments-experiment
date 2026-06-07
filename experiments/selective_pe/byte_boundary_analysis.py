"""Reproducible byte-level boundary analysis for the bilingual selective-PE model.

Two analyses, both on the byte-level English+Chinese Wikipedia models:

**Part 1  -  per-byte learned increment by category.** Run the *embedding-level*
DeltaMLP over the 257-symbol byte vocabulary and bucket the per-byte increment
δ by category. This is the surface, byte-identity view (for the shared model the
only DeltaMLP; for the per-layer model, layer 0). Chinese characters are split
into their UTF-8 lead byte (0xE4-0xE9, the character start) and continuation
bytes (0x80-0xBF, the character interior).

**Part 2  -  between-CJK word-boundary detection on held-out Chinese Wikipedia.**
Chinese is written with no spaces between words, so a gap between two Chinese
characters has *no surface delimiter*. For each such gap we score it by the
learned increment δ at the next character's lead byte and label it by whether
jieba places a word boundary there, then report ROC-AUC. Byte-identity δ (shared
model, or the per-layer model's layer 0) is near chance; context-aware mid
layers do better  -  the evidence that δ *discovers* unmarked word structure.

Usage:
    uv run python -m experiments.selective_pe.byte_boundary_analysis \
        --shared data/.../byte_selective_rope_bilingual_.../step_50000.pt \
        --per-layer data/.../byte_selective_rope_perlayer_bilingual_.../step_50000.pt
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterator

from experiments.selective_pe.config import SelectivePEModelConfig
from experiments.selective_pe.data import _load_texts
from experiments.selective_pe.model import DeltaMLP, SelectivePETransformer

CJK_LO, CJK_HI = 0x4E00, 0x9FFF  # CJK Unified Ideographs


def is_cjk(ch: str) -> bool:
    return CJK_LO <= ord(ch) <= CJK_HI


def load_model(path: str) -> tuple[SelectivePETransformer, SelectivePEModelConfig]:
    ckpt = torch.load(path, weights_only=True)
    cfg = SelectivePEModelConfig(**ckpt["model_config"])
    model = SelectivePETransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Part 1: per-byte category table (embedding-level / byte-identity delta)
# ---------------------------------------------------------------------------

BYTE_CATEGORIES: dict[str, list[int]] = {
    "English within-word (a-z)": list(range(97, 123)),
    "Capital letters (A-Z)": list(range(65, 91)),
    "Digits (0-9)": list(range(48, 58)),
    "Chinese char start (lead 0xE4-0xE9)": list(range(0xE4, 0xEA)),
    "Chinese char interior (cont 0x80-0xBF)": list(range(0x80, 0xC0)),
    "Space": [32],
    "Sentence punct (. ! ?)": [46, 33, 63],
    "Other punct (, ; : -)": [44, 59, 58, 45],
    "Newline": [10],
    "EOS": [256],
}


def embedding_delta_mlp(
    model: SelectivePETransformer, cfg: SelectivePEModelConfig
) -> DeltaMLP:
    """The DeltaMLP that reads the token embeddings (byte identity)."""
    if cfg.per_layer_delta:
        assert model.delta_mlps is not None
        mlp = model.delta_mlps[0]
    else:
        assert model.delta_mlp is not None
        mlp = model.delta_mlp
    assert isinstance(mlp, DeltaMLP)
    return mlp


@torch.no_grad()
def per_byte_table(model: SelectivePETransformer, cfg: SelectivePEModelConfig) -> None:
    ids = torch.arange(cfg.vocab_size).unsqueeze(0)
    deltas = embedding_delta_mlp(model, cfg).get_deltas(model.tok_emb(ids)).squeeze(0)
    print(f"{'category':<40}{'min':>8}{'max':>8}{'mean':>8}")
    for name, byte_ids in BYTE_CATEGORIES.items():
        vals = deltas[torch.tensor([b for b in byte_ids if b < cfg.vocab_size])]
        print(f"{name:<40}{vals.min():>8.2f}{vals.max():>8.2f}{vals.mean():>8.2f}")


# ---------------------------------------------------------------------------
# Part 2: between-CJK word-boundary AUC vs jieba on held-out zh wiki
# ---------------------------------------------------------------------------


def jieba_word_final_chars(text: str) -> set[int]:
    """Char indices that end a jieba word (i.e. a word boundary follows them)."""
    import jieba

    boundary: set[int] = set()
    cum = 0
    for word in jieba.cut(text, cut_all=False):
        cum += len(word)
        boundary.add(cum - 1)
    return boundary


def char_windows(
    text: str, max_bytes: int
) -> Iterator[tuple[list[str], list[int], list[int]]]:
    """Yield (chars, byte_ids, char_byte_starts) windows of <= max_bytes bytes.

    Windows are character-aligned (no character split across a window) and
    contiguous, so global char indices accumulate by len(chars) per window.
    """
    chars: list[str] = []
    byte_ids: list[int] = []
    starts: list[int] = []
    for ch in text:
        b = ch.encode("utf-8")
        if byte_ids and len(byte_ids) + len(b) > max_bytes:
            yield chars, byte_ids, starts
            chars, byte_ids, starts = [], [], []
        starts.append(len(byte_ids))
        chars.append(ch)
        byte_ids.extend(b)
    if chars:
        yield chars, byte_ids, starts


@torch.no_grad()
def _layer_deltas(
    model: SelectivePETransformer, cfg: SelectivePEModelConfig, byte_ids: list[int]
) -> Tensor:
    """Return (n_layers, seq_len) deltas for one window (n_layers=1 if shared)."""
    ids = torch.tensor(byte_ids).unsqueeze(0)
    d = model.get_deltas(ids)
    assert d is not None
    return d[:, 0, :] if cfg.per_layer_delta else d  # (L,S) or (1,S)


def boundary_auc(
    model: SelectivePETransformer,
    cfg: SelectivePEModelConfig,
    texts: list[str],
    max_bytes: int,
    max_gaps: int,
) -> tuple[list[float], int, float]:
    """ROC-AUC per layer for between-CJK gaps; returns (aucs, n_gaps, pos_rate)."""
    n_layers = cfg.n_layers if cfg.per_layer_delta else 1
    scores: list[list[float]] = [[] for _ in range(n_layers)]
    labels: list[int] = []
    for text in texts:
        word_final = jieba_word_final_chars(text)
        gchar = 0
        for chars, byte_ids, starts in char_windows(text, max_bytes):
            dd = _layer_deltas(model, cfg, byte_ids)
            for i in range(len(chars) - 1):
                if is_cjk(chars[i]) and is_cjk(chars[i + 1]):
                    lead = starts[i + 1]  # lead byte of the next character
                    labels.append(int((gchar + i) in word_final))
                    for layer in range(n_layers):
                        scores[layer].append(float(dd[layer, lead]))
            gchar += len(chars)
        if len(labels) >= max_gaps:
            break
    aucs = [float(roc_auc_score(labels, scores[layer])) for layer in range(n_layers)]
    return aucs, len(labels), sum(labels) / len(labels)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared", help="path to shared-delta checkpoint")
    p.add_argument("--per-layer", help="path to per-layer-delta checkpoint")
    p.add_argument("--max-docs", type=int, default=300)
    p.add_argument("--max-gaps", type=int, default=40000)
    p.add_argument("--max-bytes", type=int, default=512)
    args = p.parse_args()

    print(">>> Loading held-out Chinese Wikipedia (region reserved from training)")
    texts = _load_texts("chinese", "test", max_docs=args.max_docs, skip_docs=0)

    for label, path in [("SHARED", args.shared), ("PER-LAYER", args.per_layer)]:
        if not path:
            continue
        model, cfg = load_model(path)
        print(f"\n{'=' * 64}\n{label}: {path}\n{'=' * 64}")
        print("\n--- Part 1: per-byte increment by category ---")
        per_byte_table(model, cfg)
        print("\n--- Part 2: between-CJK word-boundary ROC-AUC (vs jieba) ---")
        aucs, n_gaps, pos = boundary_auc(
            model, cfg, texts, args.max_bytes, args.max_gaps
        )
        print(f"gaps={n_gaps}, boundary rate={pos:.3f}")
        if cfg.per_layer_delta:
            for layer, auc in enumerate(aucs):
                tag = "  <- byte identity (≈shared)" if layer == 0 else ""
                print(f"  L{layer}: AUC={auc:.3f}{tag}")
        else:
            print(f"  shared (byte identity): AUC={aucs[0]:.3f}")


if __name__ == "__main__":
    main()
