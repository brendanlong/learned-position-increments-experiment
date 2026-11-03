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

Word boundaries are also where the next character is hardest to predict, so
Part 2 compares δ against the model's own uncertainty before each gap (entropy
of the next-byte prediction, and surprisal of the next character), and asks
whether δ improves a boundary classifier that already has those.

Usage:
    uv run python -m experiments.selective_pe.byte_boundary_analysis \
        --shared data/.../byte_selective_rope_bilingual_.../step_50000.pt \
        --per-layer data/.../byte_selective_rope_perlayer_bilingual_.../step_50000.pt
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from experiments.selective_pe.config import SelectivePEModelConfig

from experiments.selective_pe.data import _load_texts
from experiments.selective_pe.model import (
    DeltaMLP,
    SelectivePETransformer,
    load_checkpoint,
)

CJK_LO, CJK_HI = 0x4E00, 0x9FFF  # CJK Unified Ideographs


def is_cjk(ch: str) -> bool:
    return CJK_LO <= ord(ch) <= CJK_HI


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
    deltas = embedding_delta_mlp(model, cfg)(model.tok_emb(ids)).squeeze(0)
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
def window_outputs(
    model: SelectivePETransformer, byte_ids: list[int]
) -> tuple[Tensor, Tensor, Tensor]:
    """Deltas, next-byte entropy and next-byte log-prob for one window.

    Returns (n_layers_or_1, S) deltas; (S,) entropy of the prediction made at
    each position; (S,) log-prob of the actual next byte (last entry is 0).
    """
    device = next(model.parameters()).device
    ids = torch.tensor(byte_ids, device=device).unsqueeze(0)
    logits, _, deltas = model.run(ids)
    logp = torch.log_softmax(logits[0].float(), dim=-1)  # (S, V)
    entropy = -(logp.exp() * logp).sum(-1)
    next_lp = torch.zeros(len(byte_ids), device=device)
    next_lp[:-1] = logp[:-1].gather(1, ids[0, 1:, None]).squeeze(1)
    d = torch.stack(deltas)[:, 0] if model.config.per_layer_delta else deltas[0]
    return d.cpu(), entropy.cpu(), next_lp.cpu()


class GapFeatures(NamedTuple):
    deltas: np.ndarray  # (n_gaps, n_layers_or_1), delta at the next lead byte
    entropy: np.ndarray  # (n_gaps,), next-byte entropy just before the gap
    surprisal: np.ndarray  # (n_gaps,), -log p(next character)
    labels: np.ndarray  # (n_gaps,), 1 = jieba word boundary
    windows: np.ndarray  # (n_gaps,), window index (for grouped CV)


def gap_features(
    model: SelectivePETransformer, texts: list[str], max_bytes: int, max_gaps: int
) -> GapFeatures:
    """Features for every gap between two CJK characters."""
    deltas, entropy, surprisal, labels, windows = [], [], [], [], []
    window = 0
    for text in texts:
        word_final = jieba_word_final_chars(text)
        gchar = 0
        for chars, byte_ids, starts in char_windows(text, max_bytes):
            window += 1
            dd, ent, next_lp = window_outputs(model, byte_ids)
            ends = [*starts[1:], len(byte_ids)]
            for i in range(len(chars) - 1):
                if is_cjk(chars[i]) and is_cjk(chars[i + 1]):
                    lead = starts[i + 1]  # lead byte of the next character
                    deltas.append(dd[:, lead].numpy())
                    # Predictions are made one byte earlier than the byte
                    entropy.append(float(ent[lead - 1]))
                    surprisal.append(-float(next_lp[lead - 1 : ends[i + 1] - 1].sum()))
                    labels.append(int((gchar + i) in word_final))
                    windows.append(window)
            gchar += len(chars)
        if len(labels) >= max_gaps:
            break
    return GapFeatures(
        np.array(deltas),
        np.array(entropy),
        np.array(surprisal),
        np.array(labels),
        np.array(windows),
    )


def cv_auc(features: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> float:
    """Out-of-fold ROC-AUC of a logistic regression, folds split by group."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    scores = cross_val_predict(
        clf,
        features,
        labels,
        groups=groups,
        cv=GroupKFold(5),
        method="decision_function",
    )
    return float(roc_auc_score(labels, scores))


def report_boundaries(f: GapFeatures, per_layer: bool) -> None:
    print(f"gaps={len(f.labels)}, boundary rate={f.labels.mean():.3f}")
    print("Single-feature AUC (0.5 = chance, <0.5 = boundaries score lower):")
    for layer in range(f.deltas.shape[1]):
        name = f"L{layer} delta" if per_layer else "shared delta"
        tag = "  <- byte identity" if layer == 0 else ""
        print(f"  {name:<28}{roc_auc_score(f.labels, f.deltas[:, layer]):.3f}{tag}")
    print(f"  {'next-byte entropy':<28}{roc_auc_score(f.labels, f.entropy):.3f}")
    print(f"  {'next-char surprisal':<28}{roc_auc_score(f.labels, f.surprisal):.3f}")
    # Does delta add anything beyond the model's own uncertainty?
    uncertainty = np.stack([f.entropy, f.surprisal], axis=1)
    log_d = np.log(f.deltas)
    print("Combined (logistic regression, 5-fold grouped by 512-byte window):")
    for name, x in [
        ("uncertainty", uncertainty),
        ("all-layer log delta", log_d),
        ("uncertainty + log delta", np.concatenate([uncertainty, log_d], axis=1)),
    ]:
        print(f"  {name:<28}{cv_auc(x, f.labels, f.windows):.3f}")


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for label, path in [("SHARED", args.shared), ("PER-LAYER", args.per_layer)]:
        if not path:
            continue
        model = load_checkpoint(path)
        cfg = model.config
        print(f"\n{'=' * 64}\n{label}: {path}\n{'=' * 64}")
        print("\n--- Part 1: per-byte increment by category ---")
        per_byte_table(model, cfg)
        print("\n--- Part 2: between-CJK word-boundary ROC-AUC (vs jieba) ---")
        model.to(device)
        features = gap_features(model, texts, args.max_bytes, args.max_gaps)
        report_boundaries(features, cfg.per_layer_delta)
        model.cpu()


if __name__ == "__main__":
    main()
