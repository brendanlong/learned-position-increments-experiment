"""Configuration for the delta-graft experiment."""

from typing import Literal

from pydantic import BaseModel

AdaptMode = Literal["lora", "full"]


class DeltaGraftConfig(BaseModel):
    """Model/grafting configuration."""

    base_model: str = "HuggingFaceTB/SmolLM2-1.7B"

    # DeltaMLP producing per-token position strides delta_t > 0 from the
    # (frozen or trainable) token embeddings. None -> hidden_size // 4.
    delta_hidden_dim: int | None = None
    # When False, the wrapper passes integer positions (identical to the base
    # model) — used to measure the un-grafted baseline with the same code path.
    delta_enabled: bool = True

    # Adaptation strategy for the base model:
    #   "lora" - freeze base weights, train low-rank adapters on attention
    #            projections (+ the DeltaMLP). Cheap; "off-the-shelf" story.
    #   "full" - unfreeze all base weights (+ the DeltaMLP). Continued
    #            pretraining with deltas; more expensive, possibly richer.
    adapt: AdaptMode = "lora"

    # LoRA hyperparameters (ignored when adapt == "full").
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )


class DeltaGraftTrainingConfig(BaseModel):
    """Training hyperparameters."""

    # Data
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str | None = "sample-10BT"
    text_column: str = "text"
    context_len: int = 1024
    # Docs reserved from the head of the stream for validation (train skips
    # them) to avoid train/val leakage on a single-source stream.
    val_docs: int = 2000

    # Optimization
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 2e-4  # good default for LoRA; lower for full FT (see train.py)
    weight_decay: float = 0.0
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    warmup_steps: int = 100
    total_steps: int = 2000
    max_grad_norm: float = 1.0

    # Logging / eval / checkpointing
    eval_every_steps: int = 200
    log_every_steps: int = 20
    eval_batches: int = 50
    save_checkpoint: bool = False
    checkpoint_dir: str = "data/delta_graft/checkpoints"

    # Wandb
    wandb_project: str = "delta-graft"
    wandb_run_name: str | None = None
    use_wandb: bool = True

    seed: int = 42
