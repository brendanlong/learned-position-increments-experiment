"""Configuration for selective PE model architecture and training."""

from typing import Literal

from pydantic import BaseModel, computed_field, model_validator

PosEncodingType = Literal["rope", "pope", "selective_rope", "selective_pope"]

# GPT-2 BPE vocabulary constants
GPT2_VOCAB_SIZE = 50257
GPT2_EOS_ID = 50256

# Byte-level vocabulary: 256 raw bytes + EOS separator
BYTE_VOCAB_SIZE = 257
BYTE_EOS_ID = 256  # byte values 0-255, EOS=256


class SelectivePEModelConfig(BaseModel):
    """Model architecture configuration for selective PE experiment."""

    dim: int
    n_heads: int
    n_layers: int
    intermediate_dim: int
    vocab_size: int = GPT2_VOCAB_SIZE
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    pos_encoding: PosEncodingType = "rope"
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    activation: Literal["swiglu", "gelu"] = "swiglu"
    init_std: float | None = 0.02

    # Selective PE config
    delta_hidden_dim: int | None = None  # MLP hidden dim; None = dim // 4
    per_layer_delta: bool = False  # Per-layer DeltaMLPs (vs shared)
    # Per-head delta: each head gets its own learned position rate (per-head
    # RoPE frequency scaling). Composes with per_layer_delta.
    per_head_delta: bool = False
    # Attention decay (recency bias), composes with any pos_encoding:
    #   "none"    - no decay (standard causal attention)
    #   "alibi"   - fixed per-head ALiBi slopes (linear distance penalty)
    #   "learned" - learned per-head, per-token forget gate (content-dependent
    #               generalization of ALiBi). alpha in (0,1) per token per head;
    #               cumulative log-alpha becomes the additive decay bias.
    # Both decay modes are per-head and shared across layers.
    decay: Literal["none", "alibi", "learned"] = "none"

    @model_validator(mode="after")
    def _validate_architecture(self) -> "SelectivePEModelConfig":
        if self.dim % self.n_heads != 0:
            msg = f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            raise ValueError(msg)
        head_dim = self.dim // self.n_heads
        if head_dim % 2 != 0:
            msg = f"head_dim ({head_dim}) must be even for {self.pos_encoding}"
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_delta_hidden_dim(self) -> int:
        if self.delta_hidden_dim is not None:
            return self.delta_hidden_dim
        return self.dim // 4


def selective_pe_model_config(
    dim: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    pos_encoding: PosEncodingType = "rope",
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm",
    activation: Literal["swiglu", "gelu"] = "swiglu",
    init_std: float | None = 0.02,
    dropout: float = 0.0,
    max_seq_len: int = 256,
    delta_hidden_dim: int | None = None,
    vocab_size: int = GPT2_VOCAB_SIZE,
    per_layer_delta: bool = False,
    per_head_delta: bool = False,
    decay: Literal["none", "alibi", "learned"] = "none",
) -> SelectivePEModelConfig:
    """Create model config with sensible defaults for selective PE experiments."""
    return SelectivePEModelConfig(
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        intermediate_dim=dim * 4,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        pos_encoding=pos_encoding,
        norm_type=norm_type,
        activation=activation,
        init_std=init_std,
        dropout=dropout,
        delta_hidden_dim=delta_hidden_dim,
        per_layer_delta=per_layer_delta,
        per_head_delta=per_head_delta,
        decay=decay,
    )


class SelectivePETrainingConfig(BaseModel):
    """Training hyperparameters for selective PE experiment."""

    # Data
    context_len: int = 256
    dataset_name: str = "SimpleStories/SimpleStories"

    # Training
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    lr_schedule: Literal["cosine", "constant"] = "cosine"
    warmup_steps: int = 500
    total_steps: int = 50000
    max_grad_norm: float = 1.0

    # Streaming mode
    generate_n: int | None = None

    # Step-based logging and evaluation
    eval_every_steps: int = 500
    log_every_steps: int = 50
    save_every_steps: int = 5000
    eval_batches: int = 50

    # Checkpointing
    checkpoint_dir: str = "data/selective_pe/checkpoints"

    # Wandb
    wandb_project: str = "byte-pe"
    wandb_run_name: str | None = None
    use_wandb: bool = True

    # Reproducibility
    seed: int = 42
