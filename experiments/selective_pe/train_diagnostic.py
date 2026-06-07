"""Train on the indirect indexing diagnostic task.

Tests whether PoPE's content-position decoupling helps on a task that
specifically requires content-conditional positional queries.

The PoPE paper showed 95% accuracy for PoPE vs 11% for RoPE on this task.

Usage:
    # Quick test
    uv run python -m experiments.selective_pe.train_diagnostic \
        --pos-encoding rope --dim 128 --n-heads 4 --n-layers 4 \
        --total-steps 5000 --no-wandb

    # Full comparison
    for pe in rope pope selective_rope selective_pope; do
        uv run python -m experiments.selective_pe.train_diagnostic \
            --pos-encoding $pe --total-steps 20000 \
            --wandb-run-name "diag_${pe}"
    done
"""

from __future__ import annotations

import argparse
import time
from typing import get_args

import torch
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from experiments.selective_pe.config import (
    PosEncodingType,
    SelectivePEModelConfig,
)
from experiments.selective_pe.indirect_indexing import (
    IndirectIndexingConfig,
    IndirectIndexingDataset,
    collate_indirect_indexing,
    evaluate_indirect_indexing,
)
from experiments.selective_pe.model import SelectivePETransformer, print_model_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on indirect indexing diagnostic task",
    )

    # Model
    parser.add_argument(
        "--pos-encoding",
        type=str,
        default="rope",
        choices=list(get_args(PosEncodingType)),
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--per-layer-delta", action="store_true")

    # Task
    parser.add_argument("--num-pairs", type=int, default=16)
    parser.add_argument("--num-values", type=int, default=32)

    # Training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--log-every-steps", type=int, default=100)

    # Compatibility with SkyPilot YAML (which injects --checkpoint-dir)
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    # Wandb
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="byte-pe")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Task config
    task_config = IndirectIndexingConfig(
        num_pairs=args.num_pairs,
        num_values=args.num_values,
    )

    # Model config
    model_config = SelectivePEModelConfig(
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        intermediate_dim=args.dim * 4,
        vocab_size=task_config.vocab_size,
        max_seq_len=task_config.seq_len,
        pos_encoding=args.pos_encoding,
        per_layer_delta=args.per_layer_delta,
        norm_type="layernorm",
        activation="gelu",
        init_std=None,  # PyTorch default init
        dropout=0.0,
    )

    model = SelectivePETransformer(model_config)
    print_model_summary(model)
    model = model.to(device)

    # Keep uncompiled ref for .parameters() and eval
    raw_model = model
    if not args.no_compile and device.type == "cuda":
        model = torch.compile(model)  # type: ignore[assignment]

    # Datasets
    train_dataset = IndirectIndexingDataset(
        task_config,
        n_examples=args.total_steps * args.batch_size,
        seed=args.seed,
    )
    eval_dataset = IndirectIndexingDataset(
        task_config,
        n_examples=2000,
        seed=args.seed + 999,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_indirect_indexing,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr)
    warmup_sched = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / max(1, args.warmup_steps)),
    )
    decay_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.total_steps - args.warmup_steps),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, decay_sched],
        milestones=[args.warmup_steps],
    )

    # Auto-generate run name
    run_name = args.wandb_run_name
    if run_name is None:
        pl = "_perlayer" if args.per_layer_delta else ""
        run_name = f"diag_{args.pos_encoding}{pl}_{args.dim}d"

    # Wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": model_config.model_dump(),
                "task": {
                    "num_pairs": args.num_pairs,
                    "num_values": args.num_values,
                },
                "training": {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "total_steps": args.total_steps,
                },
            },
            reinit=True,
        )

    # Training loop
    print(
        f"\nTraining {args.pos_encoding} on indirect indexing "
        f"({args.num_pairs} pairs, {args.num_values} values)"
    )
    print(f"  Steps: {args.total_steps}, batch: {args.batch_size}, lr: {args.lr}")
    print()

    raw_model.train()
    global_step = 0
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    t0 = time.time()

    for batch in train_loader:
        if global_step >= args.total_steps:
            break

        input_ids = batch["input_ids"].to(device)
        answer = batch["answer"].to(device)
        answer_pos = batch["answer_position"].to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(input_ids)

        # Loss only at the answer position
        batch_idx = torch.arange(logits.shape[0], device=device)
        answer_logits = logits[batch_idx, answer_pos]
        loss = F.cross_entropy(answer_logits.float(), answer)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Track metrics
        preds = answer_logits.detach().argmax(dim=-1)
        running_correct += (preds == answer).sum().item()
        running_total += answer.shape[0]
        running_loss += loss.detach().item()
        global_step += 1

        if global_step % args.log_every_steps == 0:
            avg_loss = running_loss / args.log_every_steps
            accuracy = running_correct / max(1, running_total)
            sps = args.log_every_steps / (time.time() - t0)

            log_dict: dict[str, float] = {
                "train/loss": avg_loss,
                "train/accuracy": accuracy,
                "perf/steps_per_sec": sps,
            }

            print(
                f"  step {global_step:>6d} | loss {avg_loss:.4f} | "
                f"acc {accuracy:.3f} | {sps:.1f} sps"
            )

            if global_step % args.eval_every_steps == 0:
                eval_results = evaluate_indirect_indexing(
                    raw_model,
                    eval_dataset,
                    device,
                )
                log_dict.update(eval_results)
                print(
                    f"    eval: acc={eval_results['indirect_indexing/accuracy']:.3f} "
                    f"loss={eval_results['indirect_indexing/loss']:.4f}"
                )

            if use_wandb:
                wandb.log(log_dict, step=global_step)

            running_loss = 0.0
            running_correct = 0
            running_total = 0
            t0 = time.time()

    # Final eval
    final = evaluate_indirect_indexing(raw_model, eval_dataset, device)
    print(
        f"\nFinal: accuracy={final['indirect_indexing/accuracy']:.3f} "
        f"loss={final['indirect_indexing/loss']:.4f}"
    )

    if use_wandb:
        wandb.log(final, step=global_step)
        wandb.finish()


if __name__ == "__main__":
    main()
