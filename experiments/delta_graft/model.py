"""Wrap a pretrained Llama-architecture model with a learned position delta."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from experiments.selective_pe.model import DeltaMLP

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from experiments.delta_graft.config import DeltaGraftConfig


class GradRotaryEmbedding(nn.Module):
    """Gradient-enabled clone of a Llama-family rotary embedding.

    HF's ``LlamaRotaryEmbedding.forward`` is decorated ``@torch.no_grad()`` for
    inference efficiency, which severs the gradient path from the loss back to
    ``position_ids``. Since our learned positions ARE ``position_ids``, the
    DeltaMLP would never receive a gradient. This module reproduces the exact
    same math (inv_freq, attention_scaling, float32) without the no_grad, so
    gradients flow into the positions and thus into the DeltaMLP.
    """

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        inv_freq = cast("Tensor", original.inv_freq)  # type: ignore[attr-defined]
        self.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
        self.attention_scaling: float = float(
            getattr(original, "attention_scaling", 1.0)
        )

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = cast("Tensor", self.inv_freq)
        inv_freq_expanded = (
            inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _patch_rotary_for_grad(model: nn.Module) -> int:
    """Replace model-level rotary embeddings with gradient-enabled clones.

    Returns the number of modules replaced (expected: 1 for Llama-family).
    """
    targets = [
        name
        for name, mod in model.named_modules()
        if mod.__class__.__name__.endswith("RotaryEmbedding")
    ]
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, GradRotaryEmbedding(model.get_submodule(name)))
    return len(targets)


class DeltaGraftModel(nn.Module):
    """A pretrained causal LM with grafted-on learned per-token position deltas.

    A small DeltaMLP reads the token embeddings and emits a per-token stride
    delta_t > 0; the exclusive cumulative sum gives fractional positions that
    are passed to the base model as ``position_ids``. The DeltaMLP's output
    layer is zero-initialized so that delta_t == 1 everywhere at init, making
    the fractional positions exactly ``[0, 1, 2, ...]`` and the wrapped model
    bit-identical to the base model on the first step. Any subsequent change in
    behavior is therefore attributable purely to the learned deltas.
    """

    def __init__(self, base: nn.Module, config: DeltaGraftConfig) -> None:
        super().__init__()
        self.base = base
        self.config = config
        self.delta_enabled = config.delta_enabled

        hidden = int(self._base.config.hidden_size)
        hidden_dim = config.delta_hidden_dim or hidden // 4
        self.delta_mlp = DeltaMLP(hidden, hidden_dim, bounded=False, n_heads=1)
        # Zero the output weight so delta == softplus(bias) == 1.0 exactly at
        # init (the bias is set to 0.5413 by DeltaMLP). Mirrors LoRA's zero-init
        # B matrix: the graft is a true identity until training moves it.
        nn.init.zeros_(self.delta_mlp.fc2.weight)

    @property
    def _base(self) -> PreTrainedModel:
        # Cast for the type checker; at runtime self.base is a HF model or a
        # PeftModel, both of which expose config/get_input_embeddings/__call__.
        return cast("PreTrainedModel", self.base)

    def _embed(self, input_ids: Tensor) -> Tensor:
        return self._base.get_input_embeddings()(input_ids)

    def _raw_deltas(self, embed: Tensor) -> Tensor:
        """Per-token deltas in fp32, with autocast disabled.

        Positions are an exclusive cumsum of these deltas and must be fp32:
        bf16 cannot represent distinct integer positions past ~256 (its spacing
        at 1024 is ~8), which would corrupt long-context RoPE phases.
        """
        with torch.autocast(device_type=embed.device.type, enabled=False):
            return self.delta_mlp.get_deltas(embed.float())  # (B, S), > 0

    def _positions(self, embed: Tensor) -> Tensor:
        """Exclusive cumsum of per-token deltas: token 0 -> position 0."""
        delta = self._raw_deltas(embed)  # == 1 everywhere at init
        return torch.cumsum(delta, dim=1) - delta

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        embed = self._embed(input_ids)
        position_ids = self._positions(embed) if self.delta_enabled else None
        return self._base(
            inputs_embeds=embed,
            position_ids=position_ids,
            labels=labels,
        )

    @torch.no_grad()
    def token_deltas(self, vocab_size: int, device: torch.device) -> Tensor:
        """Run the DeltaMLP over every token id -> (vocab_size,) raw deltas."""
        ids = torch.arange(vocab_size, device=device).unsqueeze(0)
        return self._raw_deltas(self._embed(ids)).squeeze(0)

    @torch.no_grad()
    def sample_deltas(self, input_ids: Tensor) -> Tensor:
        """Raw per-token deltas for a batch of ids -> (B, S)."""
        return self._raw_deltas(self._embed(input_ids))


def build_delta_graft_model(
    config: DeltaGraftConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> DeltaGraftModel:
    """Load the pretrained base model, apply the adaptation strategy, and wrap."""
    from transformers import AutoModelForCausalLM

    base: nn.Module = AutoModelForCausalLM.from_pretrained(
        config.base_model, dtype=dtype
    )

    # Make RoPE differentiable w.r.t. position_ids so the DeltaMLP can learn.
    n_patched = _patch_rotary_for_grad(base)
    assert n_patched >= 1, (
        f"No RotaryEmbedding found in {config.base_model}; "
        "the delta graft requires a RoPE-based model."
    )

    # NB: we deliberately do NOT enable gradient checkpointing. HF forces
    # use_cache=False whenever checkpointing is active during training, and in
    # this transformers version use_cache=False makes the model regenerate
    # integer position_ids — silently discarding our learned fractional
    # positions. Full fine-tuning is fit into memory via bf16 instead.

    if config.adapt == "lora":
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, lora)
    # "full" leaves every base parameter trainable (continued pretraining).

    return DeltaGraftModel(base, config)


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
