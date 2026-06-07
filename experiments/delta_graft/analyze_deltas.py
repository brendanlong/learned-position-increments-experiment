"""Analyze the learned per-token deltas from a delta-graft checkpoint.

Runs the trained DeltaMLP over the full base-model vocabulary and reports the
per-token stride delta_t by token category, to check whether the graft recovers
the linguistic hierarchy (EOS > sentence punctuation > content > function >
subword) seen when training selective PE from scratch.

For LoRA grafts the base embeddings are frozen, so this is exact. For full
fine-tunes the embeddings also drift (not captured in the delta-only
checkpoint), so the analysis is approximate.

Usage:
    uv run python -m experiments.delta_graft.analyze_deltas \
        --checkpoint wandb:brendanlong-com/delta-graft/delta-graft-<run>:v0 \
        --base-model HuggingFaceTB/SmolLM2-1.7B
"""

from __future__ import annotations

import argparse
import os
from typing import cast

import torch
from torch import nn

from experiments.selective_pe.model import DeltaMLP


def resolve_checkpoint(source: str) -> str:
    if not source.startswith("wandb:"):
        return source
    import wandb

    name = source.removeprefix("wandb:")
    api = wandb.Api()
    art = api.artifact(name)
    d = art.download(root=f"data/delta_graft/wandb_checkpoints/{art.name}")
    pts = [f for f in os.listdir(d) if f.endswith(".pt")]
    assert len(pts) == 1, f"expected 1 .pt, found {pts}"
    return os.path.join(d, pts[0])


def categorize(tok: str) -> str:
    """Bucket a decoded token string into a coarse linguistic category."""
    s = tok.replace("Ġ", " ").replace("Ċ", "\n")  # GPT2/Llama byte-BPE spaces
    stripped = s.strip()
    if stripped in {".", "!", "?", '."', '!"', '?"', ".)", "...", ".'"}:
        return "sentence_punct"
    if stripped in {",", ";", ":", "—", "-", "(", ")", '"', "'", "’", "”", "“"}:  # noqa: RUF001
        return "other_punct"
    if stripped.lower() in {
        "the", "a", "an", "of", "to", "and", "in", "is", "it", "that",
        "for", "on", "as", "with", "was", "at", "by", "be", "this", "or",
    }:
        return "function_word"
    if not s.startswith(" ") and stripped.isalpha():
        return "subword_continuation"
    if s.startswith(" ") and stripped.isalpha():
        return "word_start"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-1.7B")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = torch.load(resolve_checkpoint(args.checkpoint), weights_only=True)
    state = ckpt["delta_mlp_state_dict"]
    in_dim = state["fc1.weight"].shape[1]
    hidden_dim = state["fc1.weight"].shape[0]
    delta = DeltaMLP(in_dim, hidden_dim, bounded=False, n_heads=1)
    delta.load_state_dict(state)
    delta.eval()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    emb = cast("nn.Embedding", model.get_input_embeddings())
    vocab = int(emb.weight.shape[0])

    with torch.no_grad():
        ids = torch.arange(vocab).unsqueeze(0)
        deltas = delta.get_deltas(emb(ids)).squeeze(0)  # (vocab,)

    print(f"checkpoint: {args.checkpoint}")
    print(f"final_metrics: {ckpt.get('final_metrics')}")

    # A handful of vocab tokens never appear in the fine-tuning data, so their
    # (unbounded softplus) delta is never regularized and can explode. They are
    # an analysis artifact, not signal — report and exclude them from stats.
    exploded = deltas > 2.0
    n_exp = int(exploded.sum())
    finite = deltas[~exploded]
    exp_ids = torch.where(exploded)[0][:5]
    exp_examples = [tok.convert_ids_to_tokens(int(i)) for i in exp_ids]
    print(f"\nunused-token artifacts (delta>2): {n_exp} tokens (e.g. {exp_examples})")
    print(f"regularized delta: median={finite.median():.4f} mean={finite.mean():.4f} "
          f"std={finite.std():.4f} min={finite.min():.4f} max={finite.max():.4f}\n")

    eos = tok.eos_token_id
    if eos is not None:
        print(f"EOS (id {eos}): delta={deltas[eos]:.4f}\n")

    cats: dict[str, list[float]] = {}
    for i in range(vocab):
        if bool(exploded[i]):
            continue
        cats.setdefault(categorize(tok.convert_ids_to_tokens(i)), []).append(
            float(deltas[i])
        )
    print(f"{'category':<22}{'median δ':>10}{'n':>8}  (excludes artifacts)")
    for cat, vals in sorted(
        cats.items(), key=lambda kv: -torch.tensor(kv[1]).median().item()
    ):
        t = torch.tensor(vals)
        print(f"{cat:<22}{t.median():>10.4f}{len(vals):>8}")

    masked = deltas.clone()
    masked[exploded] = float("nan")
    order = torch.argsort(torch.nan_to_num(masked, nan=-1.0), descending=True)
    print(f"\nTop {args.top_k} highest-delta tokens (regularized):")
    for i in order[: args.top_k]:
        print(f"  {deltas[i]:.4f}  {tok.convert_ids_to_tokens(int(i))!r}")
    print(f"\nTop {args.top_k} lowest-delta tokens:")
    for i in reversed(order[len(order) - n_exp - args.top_k : len(order) - n_exp]):
        print(f"  {deltas[i]:.4f}  {tok.convert_ids_to_tokens(int(i))!r}")


if __name__ == "__main__":
    main()
