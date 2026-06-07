"""Training utilities for selective PE experiment."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from experiments.selective_pe.model import SelectivePETransformer, print_model_summary
from shared.checkpoint import upload_checkpoint_to_wandb

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from experiments.selective_pe.config import SelectivePEModelConfig


@dataclass
class TrainingResult:
    """Result of a training run."""

    best_val_loss: float
    final_val_loss: float
    checkpoint_path: Path | None
    total_steps: int


def save_model_checkpoint(
    model: torch.nn.Module,
    step: int,
    model_config: SelectivePEModelConfig,
    checkpoint_dir: Path,
) -> Path:
    """Save checkpoint, stripping torch.compile prefix from keys."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()
    }
    ckpt_path = checkpoint_dir / f"step_{step}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "model_config": model_config.model_dump(),
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")
    return ckpt_path


@torch.no_grad()
def evaluate(
    model: SelectivePETransformer,
    val_loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    max_batches: int = 50,
) -> dict[str, float]:
    """Evaluate loss and perplexity on validation set.

    Returns dict with val/loss (nats), val/ppl, val/bpt (bits per token).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    autocast_ctx = torch.autocast(device.type, dtype=torch.bfloat16)
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        with autocast_ctx:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum",
            )
        total_loss += loss.float().item()
        total_tokens += targets.numel()
    model.train()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))  # cap to avoid overflow
    bpt = avg_loss / math.log(2)
    return {"val/loss": avg_loss, "val/ppl": ppl, "val/bpt": bpt}


def _log_delta_stats(
    model: SelectivePETransformer,
    sample_batch: Tensor,
    global_step: int,
    use_wandb: bool,
) -> None:
    """Log delta position statistics for selective variants.

    For shared delta: logs overall stats.
    For per-layer delta: logs per-layer stats (delta_L0/mean, etc.)
    and overall aggregate.
    """
    deltas = model.get_deltas(sample_batch)
    if deltas is None:
        return

    delta_vals = deltas.cpu().float()
    # Shapes: (B,S) | (B,S,H) per-head | (n_layers,B,S[,H]) per-layer.
    # Use the config flag, not the tensor rank, to decide the layout.
    stats: dict[str, float] = {
        "delta/mean": delta_vals.mean().item(),
        "delta/std": delta_vals.std().item(),
        "delta/min": delta_vals.min().item(),
        "delta/max": delta_vals.max().item(),
    }

    if model.config.per_layer_delta:
        n_layers = delta_vals.shape[0]
        for layer_idx in range(n_layers):
            ld = delta_vals[layer_idx]
            stats[f"delta_L{layer_idx}/mean"] = ld.mean().item()
            stats[f"delta_L{layer_idx}/std"] = ld.std().item()
            stats[f"delta_L{layer_idx}/max"] = ld.max().item()
        summary = "  ".join(
            f"L{i}={delta_vals[i].mean().item():.2f}±{delta_vals[i].std().item():.2f}"
            for i in range(n_layers)
        )
        print(f"  delta per-layer: {summary}")
    else:
        print(
            f"  delta: mean={stats['delta/mean']:.3f} "
            f"std={stats['delta/std']:.3f} "
            f"min={stats['delta/min']:.3f} "
            f"max={stats['delta/max']:.3f}"
        )

    if use_wandb:
        stats["delta/histogram"] = wandb.Histogram(  # type: ignore[assignment]
            delta_vals.numpy().flatten().tolist(),
        )
        wandb.log(stats, step=global_step)


def train_selective_pe(
    model: SelectivePETransformer,
    train_loader: DataLoader[dict[str, Tensor]],
    val_loader: DataLoader[dict[str, Tensor]],
    model_config: SelectivePEModelConfig,
    device: torch.device,
    *,
    total_steps: int,
    grad_accum_steps: int = 1,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    lr_schedule: str = "cosine",
    max_grad_norm: float = 1.0,
    eval_every_steps: int = 500,
    log_every_steps: int = 50,
    save_every_steps: int = 5000,
    eval_batches: int = 50,
    checkpoint_dir: str = "data/selective_pe/checkpoints",
    use_wandb: bool = True,
    wandb_project: str = "byte-pe",
    wandb_run_name: str | None = None,
    wandb_config: dict[str, object] | None = None,
    save_checkpoint_to_wandb: bool = False,
    use_compile: bool = True,
) -> TrainingResult:
    """Train a selective PE transformer on next-token prediction.

    Args:
        model: The transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        model_config: Model configuration (for checkpoints)
        device: Device to train on
        total_steps: Maximum training steps
        lr: Peak learning rate
        weight_decay: AdamW weight decay
        warmup_steps: Linear warmup steps
        lr_schedule: "cosine" or "constant"
        max_grad_norm: Gradient clipping norm
        eval_every_steps: Evaluation cadence
        log_every_steps: Logging cadence
        save_every_steps: Checkpoint cadence
        eval_batches: Number of val batches per evaluation
        checkpoint_dir: Where to save checkpoints
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name
        wandb_config: Config dict to log to wandb
        save_checkpoint_to_wandb: Upload final checkpoint as wandb artifact
        use_compile: Whether to use torch.compile

    Returns:
        TrainingResult with final metrics and checkpoint path
    """
    print_model_summary(model)
    model = model.to(device)

    # torch.compile
    if use_compile and device.type == "cuda":
        print("Compiling model with torch.compile...")
        model = torch.compile(model)  # type: ignore[assignment]

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # LR schedule: linear warmup -> cosine or constant
    warmup_sched = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / max(1, warmup_steps)),
    )
    if lr_schedule == "cosine":
        decay_steps = max(1, total_steps - warmup_steps)
        decay_sched = CosineAnnealingLR(optimizer, T_max=decay_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, decay_sched],
            milestones=[warmup_steps],
        )
    else:
        scheduler = warmup_sched

    # Wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config or {},
            reinit=True,
        )

    # Training state
    global_step = 0
    running_loss = torch.tensor(0.0, device=device)
    running_count = 0
    best_val_loss = float("inf")
    last_checkpoint_path: Path | None = None
    t0 = time.time()

    # Keep a sample batch for delta logging
    sample_input_ids: Tensor | None = None

    ckpt_dir = Path(checkpoint_dir)

    effective_batch = (train_loader.batch_size or 1) * grad_accum_steps

    print(f"\nTraining for {total_steps:,} steps...")
    print(f"  LR: {lr}, WD: {weight_decay}, schedule: {lr_schedule}")
    print(
        f"  Batch size: {train_loader.batch_size} x {grad_accum_steps} accum"
        f" = {effective_batch} effective, dtype: bf16 autocast"
    )
    print(f"  Grad clip: {max_grad_norm}")
    print()

    model.train()
    done = False
    micro_step = 0

    for batch in train_loader:
        if done:
            break

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        # Save first batch for delta logging
        if sample_input_ids is None:
            sample_input_ids = input_ids.detach()

        # Forward + backward (bf16 autocast for memory efficiency)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        # Scale loss for accumulation, track unscaled for metrics
        running_loss += loss.detach()
        scaled_loss = loss / grad_accum_steps if grad_accum_steps > 1 else loss
        scaled_loss.backward()
        micro_step += 1

        # Only step optimizer after accumulating enough gradients
        if micro_step % grad_accum_steps != 0:
            continue

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # running_loss accumulates per micro-step; average over all micro-steps
        running_count += grad_accum_steps
        global_step += 1

        # Periodic logging
        if global_step % log_every_steps == 0:
            avg_loss = (running_loss / running_count).item()
            ppl = math.exp(min(avg_loss, 20.0))
            bpt = avg_loss / math.log(2)
            elapsed = time.time() - t0
            # Optimizer steps per second (logging fires every log_every_steps)
            steps_per_sec = log_every_steps / elapsed

            current_lr = optimizer.param_groups[0]["lr"]

            log_dict: dict[str, float] = {
                "train/loss": avg_loss,
                "train/ppl": ppl,
                "train/bpt": bpt,
                "perf/steps_per_sec": steps_per_sec,
                "lr": current_lr,
            }

            print(
                f"  step {global_step:>6d} | "
                f"loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                f"bpt {bpt:.3f} | "
                f"lr {current_lr:.2e} | "
                f"{steps_per_sec:.1f} steps/s"
            )

            # Evaluation
            do_eval = global_step % eval_every_steps == 0
            if do_eval:
                eval_results = evaluate(
                    model,
                    val_loader,
                    device,
                    eval_batches,
                )
                log_dict.update(eval_results)

                print(
                    f"    val: loss {eval_results['val/loss']:.4f} | "
                    f"ppl {eval_results['val/ppl']:.2f} | "
                    f"bpt {eval_results['val/bpt']:.3f}"
                )

                if eval_results["val/loss"] < best_val_loss:
                    best_val_loss = eval_results["val/loss"]
                    print(f"    *** new best val loss: {best_val_loss:.4f}")

                # Delta stats for selective variants
                if sample_input_ids is not None:
                    _log_delta_stats(
                        model,
                        sample_input_ids,
                        global_step,
                        use_wandb,
                    )

            if use_wandb:
                wandb.log(log_dict, step=global_step)

            # Reset running metrics
            running_loss = torch.tensor(0.0, device=device)
            running_count = 0
            t0 = time.time()

        # Periodic checkpoint
        if global_step % save_every_steps == 0:
            last_checkpoint_path = save_model_checkpoint(
                model,
                global_step,
                model_config,
                ckpt_dir,
            )

        if global_step >= total_steps:
            done = True

    # Final checkpoint
    last_checkpoint_path = save_model_checkpoint(
        model,
        global_step,
        model_config,
        ckpt_dir,
    )

    # Final eval
    final_eval = evaluate(model, val_loader, device, eval_batches)
    print(
        f"\nFinal eval: loss {final_eval['val/loss']:.4f} | "
        f"ppl {final_eval['val/ppl']:.2f} | "
        f"bpt {final_eval['val/bpt']:.3f}"
    )
    print(f"Best val loss: {best_val_loss:.4f}")

    if use_wandb:
        wandb.log(final_eval, step=global_step)

        if save_checkpoint_to_wandb and last_checkpoint_path is not None:
            upload_checkpoint_to_wandb(
                last_checkpoint_path,
                artifact_name=(
                    f"selective-pe-{wandb_run_name}"
                    if wandb_run_name
                    else "selective-pe-model"
                ),
                metadata={
                    "total_steps": global_step,
                    "best_val_loss": best_val_loss,
                    "final_val_loss": final_eval["val/loss"],
                    "final_val_ppl": final_eval["val/ppl"],
                    "pos_encoding": model_config.pos_encoding,
                },
            )
        wandb.finish()

    return TrainingResult(
        best_val_loss=best_val_loss,
        final_val_loss=final_eval["val/loss"],
        checkpoint_path=last_checkpoint_path,
        total_steps=global_step,
    )
