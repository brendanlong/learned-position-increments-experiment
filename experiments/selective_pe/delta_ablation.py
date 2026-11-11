"""Does the model use its learned increments? Ablate them one layer at a time.

Each layer's deltas split into a speed and a pattern: the layer's speed is
the geometric mean of its deltas over the eval data (scaling every delta in a
layer is the same as changing that layer's RoPE base), and the pattern is
delta / speed. The script prints both, with the pattern averaged by input
byte category.

Then, for each layer, it replaces that layer's learned deltas with either
  - 1 (plain integer RoPE positions for that layer), or
  - the layer's speed (keeps the speed, removes the per-token pattern),
and measures validation loss. A large loss increase from the speed ablation
means the model relies on the per-token pattern itself. Loss is also broken
down by the category of the input byte (e.g. "after space" = predicting a
word's first byte).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from experiments.selective_pe.byte_boundary_analysis import BYTE_CATEGORIES
from experiments.selective_pe.config import BYTE_VOCAB_SIZE
from experiments.selective_pe.data import FixedChunkDataset, prepare_byte_corpus
from experiments.selective_pe.model import SelectivePETransformer, load_checkpoint

Override = Callable[[Tensor, int], Tensor] | None


@torch.no_grad()
def collect_deltas(model: SelectivePETransformer, batches: list[Tensor]) -> Tensor:
    """(n_layers, n_batches, B, S, n_heads_or_1) learned deltas."""
    per_batch = []
    for ids in batches:
        # Same autocast as per_token_loss, so later layers see the same inputs
        with torch.autocast(ids.device.type, dtype=torch.bfloat16):
            deltas = torch.stack(model.run(ids)[2])  # (L, B, S[, H])
        if deltas.dim() == 3:
            deltas = deltas.unsqueeze(-1)
        per_batch.append(deltas)
    return torch.stack(per_batch, dim=1)


def constant_override(layers: set[int], values: Tensor) -> Override:
    """Replace deltas in ``layers`` with values[layer] (per head if per-head)."""

    def override(deltas: Tensor, layer_idx: int) -> Tensor:
        if layer_idx not in layers:
            return deltas
        value = values[layer_idx]
        if deltas.dim() == 2:
            value = value.squeeze(-1)
        return value.expand_as(deltas).to(deltas.dtype)

    return override


@torch.no_grad()
def per_token_loss(
    model: SelectivePETransformer, batches: list[Tensor], override: Override
) -> Tensor:
    """(n_batches, B, S) next-byte loss in nats."""
    losses = []
    for ids in batches:
        with torch.autocast(ids.device.type, dtype=torch.bfloat16):
            logits = model.run(ids[:, :-1], delta_override=override)[0]
        losses.append(
            F.cross_entropy(
                logits.float().transpose(1, 2), ids[:, 1:], reduction="none"
            )
        )
    return torch.stack(losses)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="local path or hf:<experiment>/<run_name>/<file>")
    p.add_argument("--dataset", default="bilingual")
    p.add_argument("--val-max-docs", type=int, default=2000)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-batches", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint).to(device)
    assert model.config.vocab_size == BYTE_VOCAB_SIZE, "byte-level checkpoints only"
    n_layers = model.config.n_layers

    # Same val windows the training loop evaluates on
    val = FixedChunkDataset(
        prepare_byte_corpus("test", args.dataset, args.val_max_docs),
        args.context_len,
    )
    # Windows spread evenly over the val set (it interleaves English and
    # Chinese documents, and individual documents are long)
    n = min(args.n_batches * args.batch_size, len(val))
    windows = [val[int(i)] for i in torch.linspace(0, len(val) - 1, n)]
    # input_ids plus the final target, so targets are ids[:, 1:]
    seqs = torch.stack(
        [torch.cat([w["input_ids"], w["targets"][-1:]]) for w in windows]
    )
    batches = list(seqs.to(device).split(args.batch_size))
    inputs = torch.stack([ids[:, :-1] for ids in batches])  # (N, B, S)

    categories = {
        name: torch.isin(inputs, torch.tensor(ids, device=device))
        for name, ids in BYTE_CATEGORIES.items()
    }
    categories = {name: m for name, m in categories.items() if m.any()}

    deltas = collect_deltas(model, [ids[:, :-1] for ids in batches])
    log_deltas = deltas.log()
    speeds = log_deltas.mean(dim=(1, 2, 3)).exp()  # (L, H)
    ones = torch.ones_like(speeds)
    pattern = (log_deltas - speeds.log()[:, None, None, None]).exp().mean(-1)
    print(f"checkpoint: {args.checkpoint}")
    print("\nPer-layer speed (geometric mean delta) and pattern (delta / speed)")
    print(f"{'':<48}" + "".join(f"{f'L{i}':>8}" for i in range(n_layers)))
    print(f"{'speed':<48}" + "".join(f"{v:>8.3f}" for v in speeds.mean(-1)))
    print(
        f"{'max delta':<48}"
        + "".join(f"{deltas[i].max().item():>8.2f}" for i in range(n_layers))
    )
    for name, m in categories.items():
        print(
            f"{'pattern: ' + name:<48}"
            + "".join(f"{pattern[i][m].mean().item():>8.2f}" for i in range(n_layers))
        )

    conditions: list[tuple[str, Override]] = [("learned", None)]
    all_layers = set(range(n_layers))
    conditions.append(("all layers = 1", constant_override(all_layers, ones)))
    conditions.append(("all layers = speed", constant_override(all_layers, speeds)))
    for layer in range(n_layers):
        conditions.append((f"L{layer} = 1", constant_override({layer}, ones)))
        conditions.append((f"L{layer} = speed", constant_override({layer}, speeds)))

    rows: list[tuple[str, float, dict[str, float]]] = []
    for name, override in conditions:
        loss = per_token_loss(model, batches, override)
        by_cat = {f"after {c}": loss[m].mean().item() for c, m in categories.items()}
        rows.append((name, loss.mean().item(), by_cat))

    base_loss, base_cats = rows[0][1], rows[0][2]
    cat_names = list(base_cats)
    n_tokens = inputs.numel()
    print(f"\n{n_tokens:,} tokens; loss in nats; change vs learned in parentheses")
    print(
        f"\n{'condition':<20} {'loss':>16}  " + "  ".join(f"{c:>24}" for c in cat_names)
    )
    print(
        f"{'(tokens)':<20} {n_tokens:>16,}  "
        + "  ".join(f"{int(m.sum()):>24,}" for m in categories.values())
    )
    for name, loss, by_cat in rows:
        cells = "  ".join(
            f"{by_cat[c]:>14.4f} ({by_cat[c] - base_cats[c]:+.4f})" for c in cat_names
        )
        print(f"{name:<20} {loss:>7.4f} ({loss - base_loss:+.4f})  {cells}")


if __name__ == "__main__":
    main()
