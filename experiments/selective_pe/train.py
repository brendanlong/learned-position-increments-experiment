"""Training script for selective PE comparison experiment.

Usage:
    # Smoke test (small model, no wandb)
    uv run python -m experiments.selective_pe.train \
        --pos-encoding rope --dim 64 --n-heads 2 --n-layers 2 \
        --context-len 64 --generate-n 1000 --batch-size 16 \
        --no-wandb --no-compile

    # Full experiment (streaming, one PE variant)
    uv run python -m experiments.selective_pe.train \
        --pos-encoding selective_rope --generate-n 5000000

    # HP sweep run (short, with wandb)
    uv run python -m experiments.selective_pe.train \
        --pos-encoding rope --generate-n 500000 --total-steps 5000 \
        --lr 3e-4 --weight-decay 0.01 \
        --wandb-run-name "sweep_lr3e-4_wd0.01"

    # Via SkyPilot
    ./train.sh local selective_pe -- --pos-encoding selective_pope \
        --generate-n 5000000
"""

from __future__ import annotations

import argparse
from typing import get_args

import torch
from torch.utils.data import DataLoader

from experiments.selective_pe.config import (
    BYTE_VOCAB_SIZE,
    GPT2_VOCAB_SIZE,
    PosEncodingType,
    SelectivePETrainingConfig,
    selective_pe_model_config,
)
from experiments.selective_pe.data import (
    DATASET_CONFIGS,
    FixedChunkDataset,
    StreamingChunkDataset,
    collate_chunks,
    prepare_byte_corpus,
    prepare_corpus,
    train_test_share_source,
)
from experiments.selective_pe.model import SelectivePETransformer
from experiments.selective_pe.training import train_selective_pe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train selective PE transformer on SimpleStories",
    )

    # Model
    parser.add_argument(
        "--pos-encoding",
        type=str,
        default="rope",
        choices=list(get_args(PosEncodingType)),
        help="Positional encoding type",
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--delta-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--per-layer-delta",
        action="store_true",
        help="Use per-layer DeltaMLPs instead of shared (for selective variants)",
    )
    parser.add_argument(
        "--per-head-delta",
        action="store_true",
        help="Per-head learned position rates (per-head RoPE frequency scaling)",
    )
    parser.add_argument(
        "--decay",
        type=str,
        default="none",
        choices=["none", "alibi", "learned"],
        help="Attention decay: none, alibi (fixed per-head slopes), or learned "
        "(per-head per-token forget gate)",
    )
    parser.add_argument(
        "--frozen-delta-from",
        type=str,
        default=None,
        help="Path to a selective PE checkpoint to extract frozen deltas from",
    )
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--activation",
        type=str,
        default="swiglu",
        choices=["swiglu", "gelu"],
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="simplestories",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to train on",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        choices=["gpt2", "byte"],
        help="Tokenizer: gpt2 (BPE, 50257 vocab) or byte (raw UTF-8, 257 vocab)",
    )
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument(
        "--generate-n",
        type=int,
        default=None,
        help="Streaming mode: generate N examples, 1 epoch",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Cap training stories (for debugging)",
    )
    parser.add_argument(
        "--val-max-docs",
        type=int,
        default=5000,
        help="Cap val/test docs (enough for eval_batches; avoids OOM on "
        "uncapped streaming datasets like Wikipedia)",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * this)",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
    )
    parser.add_argument("--no-compile", action="store_true")

    # Eval / logging / checkpoint
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/selective_pe/checkpoints",
    )

    # Wandb
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="byte-pe")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Upload final checkpoint to wandb as artifact",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Vocab size from tokenizer choice
    vocab_size = BYTE_VOCAB_SIZE if args.tokenizer == "byte" else GPT2_VOCAB_SIZE

    # Model config
    model_config = selective_pe_model_config(
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        pos_encoding=args.pos_encoding,
        init_std=args.init_std,
        dropout=args.dropout,
        max_seq_len=args.context_len,
        delta_hidden_dim=args.delta_hidden_dim,
        activation=args.activation,
        vocab_size=vocab_size,
        per_layer_delta=args.per_layer_delta,
        per_head_delta=args.per_head_delta,
        decay=args.decay,
    )

    # Training config
    training_config = SelectivePETrainingConfig(
        context_len=args.context_len,
        dataset_name=f"{args.dataset} ({args.tokenizer})",
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        max_grad_norm=args.max_grad_norm,
        eval_every_steps=args.eval_every_steps,
        log_every_steps=args.log_every_steps,
        save_every_steps=args.save_every_steps,
        eval_batches=args.eval_batches,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_wandb=not args.no_wandb,
        seed=args.seed,
        generate_n=args.generate_n,
    )

    # Auto-generate run name if not specified
    run_name = args.wandb_run_name
    if run_name is None:
        tok_tag = "byte" if args.tokenizer == "byte" else ""
        ds_tag = f"_{args.dataset}" if args.dataset != "simplestories" else ""
        pl_tag = "_perlayer" if args.per_layer_delta else ""
        ph_tag = "_perhead" if args.per_head_delta else ""
        fg_tag = f"_{args.decay}" if args.decay != "none" else ""
        run_name = (
            f"{args.pos_encoding}_{args.dim}d_{args.n_heads}h_{args.n_layers}L"
            + pl_tag
            + ph_tag
            + fg_tag
            + (f"_{tok_tag}" if tok_tag else "")
            + ds_tag
        )

    # Load data. Cap the val corpus: we only evaluate eval_batches batches, so
    # a few thousand docs is plenty. Without a cap, streaming datasets with no
    # explicit test split (e.g. Wikipedia) load millions of docs and OOM.
    #
    # When train and test stream the same source (Wikipedia has no test split),
    # reserve the first val_max_docs for val and skip them in train to avoid
    # train/test leakage.
    train_skip = args.val_max_docs if train_test_share_source(args.dataset) else 0
    print(f"Loading {args.dataset} corpus ({args.tokenizer} tokenizer)...")
    if args.tokenizer == "byte":
        train_corpus = prepare_byte_corpus(
            "train",
            dataset=args.dataset,
            max_docs=args.max_stories,
            skip_docs=train_skip,
        )
        val_corpus = prepare_byte_corpus(
            "test",
            dataset=args.dataset,
            max_docs=args.val_max_docs,
        )
    else:
        train_corpus = prepare_corpus(
            "train",
            dataset=args.dataset,
            max_docs=args.max_stories,
            skip_docs=train_skip,
        )
        val_corpus = prepare_corpus(
            "test",
            dataset=args.dataset,
            max_docs=args.val_max_docs,
        )

    # Create datasets
    if args.generate_n is not None:
        # Each optimizer step consumes grad_accum_steps micro-batches
        micro_batches = args.generate_n // args.batch_size
        total_steps = min(
            args.total_steps,
            micro_batches // args.grad_accum_steps,
        )
        effective_batch = args.batch_size * args.grad_accum_steps
        print(
            f"Streaming mode: {args.generate_n:,} examples, "
            f"{total_steps:,} steps "
            f"(effective batch {effective_batch})"
        )
        train_dataset = StreamingChunkDataset(
            train_corpus,
            args.context_len,
            args.generate_n,
            seed=args.seed + 100,
        )
    else:
        total_steps = args.total_steps
        print(f"Fixed dataset mode: {len(train_corpus):,} tokens")
        train_dataset = FixedChunkDataset(train_corpus, args.context_len)

    val_dataset = FixedChunkDataset(val_corpus, args.context_len)

    # num_workers=0: the dataset is in-memory tensor slicing (CPU-bound and
    # fast), so workers add overhead with no benefit. It also avoids the
    # IterableDataset duplication trap where multiple workers each yield the
    # same random chunks (they'd share seed + epoch with no worker offset).
    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not isinstance(train_dataset, StreamingChunkDataset),
        collate_fn=collate_chunks,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_chunks,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Create model
    model = SelectivePETransformer(model_config)

    # Install frozen deltas if requested
    if args.frozen_delta_from is not None:
        from experiments.selective_pe.model import extract_frozen_deltas

        frozen_deltas = extract_frozen_deltas(args.frozen_delta_from)
        model.install_frozen_deltas(frozen_deltas)
        n_params = model.count_parameters()
        print(f"  Frozen deltas installed ({n_params:,} trainable params)")

    # Train
    result = train_selective_pe(
        model,
        train_loader,
        val_loader,
        model_config,
        device,
        total_steps=total_steps,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        max_grad_norm=args.max_grad_norm,
        eval_every_steps=args.eval_every_steps,
        log_every_steps=args.log_every_steps,
        save_every_steps=args.save_every_steps,
        eval_batches=args.eval_batches,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
        wandb_config={
            "model": model_config.model_dump(),
            "training": training_config.model_dump(),
        },
        save_checkpoint_to_wandb=args.save_checkpoint,
        use_compile=not args.no_compile,
    )

    print(f"\n{'=' * 50}")
    print("Training complete")
    print(f"  Total steps: {result.total_steps:,}")
    print(f"  Best val loss: {result.best_val_loss:.4f}")
    print(f"  Final val loss: {result.final_val_loss:.4f}")
    if result.checkpoint_path is not None:
        print(f"  Checkpoint: {result.checkpoint_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
