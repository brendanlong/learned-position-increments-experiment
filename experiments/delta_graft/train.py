"""Train a delta-graft model on FineWeb-Edu (SmolLM2 / OLMo 2).

Examples:
    # LoRA graft on SmolLM2-1.7B
    uv run python -m experiments.delta_graft.train \
        --base-model HuggingFaceTB/SmolLM2-1.7B --adapt lora --total-steps 2000

    # Full fine-tune graft (lower LR auto-selected)
    uv run python -m experiments.delta_graft.train --adapt full
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import wandb
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiments.delta_graft.config import DeltaGraftConfig, DeltaGraftTrainingConfig
from experiments.delta_graft.data import make_streams
from experiments.delta_graft.model import (
    DeltaGraftModel,
    build_delta_graft_model,
    count_trainable_params,
)
from shared.checkpoint import upload_checkpoint_to_wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Delta-graft training")
    # Model / grafting
    p.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-1.7B")
    p.add_argument("--adapt", choices=["lora", "full"], default="lora")
    p.add_argument("--delta-hidden-dim", type=int, default=None)
    p.add_argument(
        "--delta-disabled",
        action="store_true",
        help="Pass integer positions (un-grafted baseline) through same code.",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    # Data
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--context-len", type=int, default=1024)
    p.add_argument("--val-docs", type=int, default=2000)
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for the train stream (sharded, parallel "
        "tokenization). 0 tokenizes synchronously in the main process.",
    )
    # Optimization
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Default: 2e-4 for LoRA, 2e-5 for full fine-tune.",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--total-steps", type=int, default=2000)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # Logging / eval
    p.add_argument("--eval-every-steps", type=int, default=200)
    p.add_argument("--log-every-steps", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--save-checkpoint", action="store_true")
    p.add_argument("--checkpoint-dir", default="data/delta_graft/checkpoints")
    # Wandb
    p.add_argument("--wandb-project", default="delta-graft")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_configs(
    args: argparse.Namespace,
) -> tuple[DeltaGraftConfig, DeltaGraftTrainingConfig]:
    model_cfg = DeltaGraftConfig(
        base_model=args.base_model,
        delta_hidden_dim=args.delta_hidden_dim,
        delta_enabled=not args.delta_disabled,
        adapt=args.adapt,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    lr = args.lr if args.lr is not None else (2e-4 if args.adapt == "lora" else 2e-5)
    run_name = args.wandb_run_name or _default_run_name(args)
    train_cfg = DeltaGraftTrainingConfig(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        context_len=args.context_len,
        val_docs=args.val_docs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        max_grad_norm=args.max_grad_norm,
        eval_every_steps=args.eval_every_steps,
        log_every_steps=args.log_every_steps,
        eval_batches=args.eval_batches,
        save_checkpoint=args.save_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )
    return model_cfg, train_cfg


def _default_run_name(args: argparse.Namespace) -> str:
    short = args.base_model.split("/")[-1]
    tag = "baseline" if args.delta_disabled else "delta"
    return f"{short}_{args.adapt}_{tag}"


def _make_scheduler(
    optimizer: torch.optim.Optimizer, warmup: int, total: int
) -> SequentialLR:
    warmup_sched = LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup))
    )
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup))
    return SequentialLR(optimizer, [warmup_sched, cosine], milestones=[warmup])


@torch.no_grad()
def evaluate(
    model: DeltaGraftModel,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    autocast_ctx: torch.autocast,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with autocast_ctx:
            out = model(input_ids, labels=labels)
        total_loss += float(out.loss)
        n += 1
    model.train()
    avg = total_loss / max(1, n)
    return {"val/loss": avg, "val/ppl": math.exp(min(avg, 20.0))}


def _log_delta_stats(
    model: DeltaGraftModel, sample: Tensor, use_wandb: bool, step: int
) -> None:
    if not model.delta_enabled:
        return
    d = model.sample_deltas(sample).float().cpu()
    stats = {
        "delta/mean": d.mean().item(),
        "delta/std": d.std().item(),
        "delta/min": d.min().item(),
        "delta/max": d.max().item(),
    }
    print(
        f"    delta: mean={stats['delta/mean']:.3f} std={stats['delta/std']:.3f} "
        f"min={stats['delta/min']:.3f} max={stats['delta/max']:.3f}"
    )
    if use_wandb:
        wandb.log(stats, step=step)


def save_delta_checkpoint(
    model: DeltaGraftModel,
    model_cfg: DeltaGraftConfig,
    cfg: DeltaGraftTrainingConfig,
    step: int,
    final_metrics: dict[str, float],
) -> Path:
    """Save the trained DeltaMLP (+ config) for downstream delta analysis.

    Only the DeltaMLP is saved; the base model is reloaded from HF by name. In
    LoRA mode the embeddings are frozen so this fully reconstructs the learned
    positions; in full mode the embeddings also drift (not captured here).
    """
    out_dir = Path(cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "delta_mlp.pt"
    torch.save(
        {
            "step": step,
            "delta_mlp_state_dict": model.delta_mlp.state_dict(),
            "model_config": model_cfg.model_dump(),
            "final_metrics": final_metrics,
        },
        path,
    )
    print(f"  Saved delta checkpoint: {path}")
    return path


def main() -> None:
    args = parse_args()
    model_cfg, cfg = build_configs(args)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # LoRA: bf16 (frozen base, tiny fp32 adapters + DeltaMLP). Full fine-tune:
    # fp32 — bf16 weight updates underflow at small LR (1.0 + 2e-5 rounds back
    # to 1.0), so full FT needs fp32 master weights. fp32 1.7B fits a 48GB GPU
    # only at small batch (no gradient checkpointing — it breaks our positions).
    dtype = torch.bfloat16 if model_cfg.adapt == "lora" else torch.float32

    print(f">>> Loading {model_cfg.base_model} (adapt={model_cfg.adapt}, {dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model)
    model = build_delta_graft_model(model_cfg, dtype=dtype)
    model.to(device)
    # Keep the DeltaMLP and its cumsum in fp32 for precise positions.
    model.delta_mlp.float()

    trainable, total = count_trainable_params(model)
    pct = 100 * trainable / total
    print(f"    trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    train_stream, val_stream = make_streams(
        dataset=cfg.dataset,
        dataset_config=cfg.dataset_config,
        text_column=cfg.text_column,
        tokenizer=tokenizer,
        context_len=cfg.context_len,
        val_docs=cfg.val_docs,
        seed=cfg.seed,
    )
    train_loader = DataLoader(
        train_stream,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_stream, batch_size=cfg.batch_size, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = _make_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)
    autocast_ctx = torch.autocast(device.type, dtype=torch.bfloat16)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={**model_cfg.model_dump(), **cfg.model_dump()},
        )

    # Step-0 baseline: with identity-init deltas this must match the base model.
    base_metrics = evaluate(model, val_loader, device, autocast_ctx, cfg.eval_batches)
    print(
        f">>> step 0 (identity init): val/loss={base_metrics['val/loss']:.4f} "
        f"val/ppl={base_metrics['val/ppl']:.3f}"
    )
    if cfg.use_wandb:
        wandb.log(base_metrics, step=0)

    model.train()
    optimizer.zero_grad()
    step = 0
    micro = 0
    t0 = time.time()
    running = 0.0
    train_iter = iter(train_loader)
    while step < cfg.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with autocast_ctx:
            out = model(input_ids, labels=labels)
            loss = out.loss / cfg.grad_accum
        loss.backward()
        running += float(out.loss)
        micro += 1
        if micro % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % cfg.log_every_steps == 0:
                avg = running / (cfg.log_every_steps * cfg.grad_accum)
                running = 0.0
                sps = cfg.log_every_steps / (time.time() - t0)
                t0 = time.time()
                lr = scheduler.get_last_lr()[0]
                print(
                    f"step {step}/{cfg.total_steps} loss={avg:.4f} "
                    f"ppl={math.exp(min(avg, 20.0)):.3f} lr={lr:.2e} {sps:.2f} it/s"
                )
                if cfg.use_wandb:
                    wandb.log(
                        {"train/loss": avg, "train/ppl": math.exp(min(avg, 20.0)),
                         "lr": lr}, step=step
                    )
                _log_delta_stats(model, input_ids, cfg.use_wandb, step)

            if step % cfg.eval_every_steps == 0:
                m = evaluate(
                    model, val_loader, device, autocast_ctx, cfg.eval_batches
                )
                delta_vs_base = m["val/loss"] - base_metrics["val/loss"]
                print(
                    f">>> eval step {step}: val/loss={m['val/loss']:.4f} "
                    f"val/ppl={m['val/ppl']:.3f} (vs base {delta_vs_base:+.4f})"
                )
                if cfg.use_wandb:
                    wandb.log({**m, "val/loss_vs_base": delta_vs_base}, step=step)

    final = evaluate(model, val_loader, device, autocast_ctx, cfg.eval_batches)
    final["val/loss_vs_base"] = final["val/loss"] - base_metrics["val/loss"]
    print(
        f">>> FINAL step {step}: val/loss={final['val/loss']:.4f} "
        f"val/ppl={final['val/ppl']:.3f} (vs base {final['val/loss_vs_base']:+.4f})"
    )
    if model.delta_enabled:
        sample = next(iter(val_loader))["input_ids"].to(device)
        _log_delta_stats(model, sample, cfg.use_wandb, step)

    if cfg.save_checkpoint:
        ckpt = save_delta_checkpoint(model, model_cfg, cfg, step, final)
        if cfg.use_wandb:
            upload_checkpoint_to_wandb(
                ckpt,
                artifact_name=f"delta-graft-{cfg.wandb_run_name}",
                metadata={**model_cfg.model_dump(), **final},
            )

    if cfg.use_wandb:
        wandb.log(final, step=step)
        wandb.finish()


if __name__ == "__main__":
    main()
    # HF datasets streaming leaves background threads that crash py3.12's
    # interpreter shutdown (PyGILState_Release -> SIGABRT, exit 134), which
    # would make SkyPilot mark a finished job as failed. All work is done and
    # wandb is flushed by now, so exit hard after flushing stdio.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
