"""Transformer with configurable positional encoding variants.

Supports four PE modes:
- rope: Standard RoPE with integer positions
- pope: PoPE (softplus magnitude + position-only phase)
- selective_rope: Learned per-token delta positions + RoPE rotation
- selective_pope: Learned deltas + PoPE decoupling

The selective variants use a small MLP (DeltaMLP) to produce per-token
position strides delta_t > 0 from token embeddings. Cumulative sum gives
real-valued positions. This is computed once and shared across all layers.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.selective_pe.config import SelectivePEModelConfig

# ---------------------------------------------------------------------------
# Positional encoding primitives
# ---------------------------------------------------------------------------


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tensor:
    """Precompute RoPE complex frequencies for integer positions."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def compute_rope_freqs_for_positions(
    positions: Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> Tensor:
    """Compute RoPE complex frequencies for arbitrary real-valued positions.

    Args:
        positions: (batch, seq_len) for a shared per-token rate, or
            (batch, seq_len, n_heads) for per-head rates.
        head_dim: head dimension (must be even)
        theta: base frequency

    Returns:
        (batch, seq_len, n_heads_or_1, head_dim // 2) complex tensor. The
        head dim is 1 for shared positions (broadcasts over heads) or n_heads
        for per-head positions.
    """
    if positions.dim() == 2:
        positions = positions.unsqueeze(-1)  # (B, S, 1) shared across heads
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    )
    # positions: (B, S, Hh, 1) x inv_freq: (HD//2,) -> (B, S, Hh, HD//2)
    angles = positions.unsqueeze(-1) * inv_freq
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply RoPE with precomputed integer-position frequencies.

    Args:
        x: (batch, seq_len, n_heads, head_dim)
        freqs: (seq_len, head_dim // 2) complex
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    f = freqs[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    x_rotated = torch.view_as_real(x_complex * f).flatten(-2)
    return x_rotated.type_as(x)


def apply_rope_batched(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply RoPE with per-batch real-valued position frequencies.

    Args:
        x: (batch, seq_len, n_heads, head_dim)
        freqs: (batch, seq_len, n_heads_or_1, head_dim // 2) complex
               (head dim 1 broadcasts over heads; n_heads for per-head rates)
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)


def apply_pope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply PoPE: decouple content magnitude from positional phase.

    Uses softplus magnitude for content matching and position-only phase,
    eliminating the what-where confound (Milsom et al., arXiv:2509.10534).

    Args:
        x: (batch, seq_len, n_heads, head_dim)
        freqs: (seq_len, head_dim // 2) complex
    """
    x_paired = x.float().reshape(*x.shape[:-1], -1, 2)
    magnitude = F.softplus(x_paired.norm(dim=-1))

    f = freqs[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    cos_pos = f.real
    sin_pos = f.imag

    real_part = magnitude * cos_pos
    imag_part = magnitude * sin_pos
    result = torch.stack([real_part, imag_part], dim=-1).flatten(-2)
    return result.type_as(x)


def apply_pope_batched(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply PoPE with per-batch real-valued position frequencies.

    Args:
        x: (batch, seq_len, n_heads, head_dim)
        freqs: (batch, seq_len, n_heads_or_1, head_dim // 2) complex
    """
    x_paired = x.float().reshape(*x.shape[:-1], -1, 2)
    magnitude = F.softplus(x_paired.norm(dim=-1))

    cos_pos = freqs.real
    sin_pos = freqs.imag

    real_part = magnitude * cos_pos
    imag_part = magnitude * sin_pos
    result = torch.stack([real_part, imag_part], dim=-1).flatten(-2)
    return result.type_as(x)


# ---------------------------------------------------------------------------
# Delta MLP for selective variants
# ---------------------------------------------------------------------------


class DeltaMLP(nn.Module):
    """Small MLP producing per-token position deltas.

    Takes hidden states and outputs a scalar delta_t > 0 for each token.
    Cumulative sum gives positions.

    Two modes controlled by ``bounded``:
    - Unbounded (default, for shared delta on embeddings):
      Linear -> GELU -> Linear -> softplus.
      Output bias initialized so softplus(bias) ~ 1.0.
    - Bounded (for per-layer delta on hidden states):
      RMSNorm -> Linear -> GELU -> Linear -> sigmoid * max_delta.
      Input normalization prevents hidden-state magnitude from driving
      explosion; sigmoid bounding prevents output divergence.
      Bias initialized so sigmoid(bias) * max_delta ~ 1.0.

    Set ``n_heads > 1`` for per-head rates: the output is (B, S, n_heads)
    instead of (B, S), letting each head learn its own "rate of time"
    (per-head RoPE frequency scaling).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        bounded: bool = False,
        max_delta: float = 10.0,
        n_heads: int = 1,
    ) -> None:
        super().__init__()
        self.bounded = bounded
        self.max_delta = max_delta
        self.n_heads = n_heads
        self.norm: nn.Module | None = nn.RMSNorm(dim) if bounded else None
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_heads)
        self._init_bias()

    def _init_bias(self) -> None:
        """Set output bias so initial delta ~ 1.0."""
        if self.bounded:
            # sigmoid(x) * max_delta = 1.0  =>  x = log(1/(max_delta-1))
            import math

            bias_val = math.log(1.0 / (self.max_delta - 1.0))
            nn.init.constant_(self.fc2.bias, bias_val)
        else:
            # softplus(0.5413) ~ 1.0
            nn.init.constant_(self.fc2.bias, 0.5413)

    def _compute_delta(self, x: Tensor) -> Tensor:
        """Compute raw delta values (before cumsum). (B,S) or (B,S,H)."""
        if self.norm is not None:
            x = self.norm(x)
        h = F.gelu(self.fc1(x))
        raw = self.fc2(h)  # (B, S, n_heads)
        if self.n_heads == 1:
            raw = raw.squeeze(-1)  # (B, S)
        if self.bounded:
            return self.max_delta * torch.sigmoid(raw)
        return F.softplus(raw)

    def forward(self, x: Tensor) -> Tensor:
        """Compute cumulative positions from hidden states.

        Returns (B, S) for a shared rate or (B, S, n_heads) for per-head.
        Cumsum runs over the seq dim; for per-head we transpose so the scan
        is over the last (contiguous) dim (avoids a torch.compile cumsum bug).
        """
        delta = self._compute_delta(x)
        if delta.dim() == 3:  # (B, S, H) -> cumsum over S via last-dim scan
            return torch.cumsum(delta.transpose(1, 2), dim=-1).transpose(1, 2)
        return torch.cumsum(delta, dim=1)

    def get_deltas(self, x: Tensor) -> Tensor:
        """Return raw deltas (before cumsum) for interpretability."""
        return self._compute_delta(x)


class FrozenDeltaLookup(nn.Module):
    """Frozen per-token position deltas from a pretrained DeltaMLP.

    Instead of computing deltas from embeddings via an MLP, this module
    stores a precomputed delta for each token ID as a non-trainable buffer.
    This isolates the effect of learned position rates from the optimization
    dynamics of joint training.
    """

    def __init__(self, deltas_per_token: Tensor) -> None:
        super().__init__()
        # deltas_per_token: (vocab_size,) - precomputed delta for each token ID
        self.register_buffer("deltas", deltas_per_token)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Look up deltas by token ID and cumsum.

        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            positions: (batch, seq_len) cumulative positions
        """
        assert isinstance(self.deltas, Tensor)
        delta = self.deltas[input_ids]  # (B, S)
        return torch.cumsum(delta, dim=1)

    def get_deltas(self, input_ids: Tensor) -> Tensor:
        """Return raw deltas (before cumsum)."""
        assert isinstance(self.deltas, Tensor)
        return self.deltas[input_ids]


def _resolve_checkpoint_path(source: str) -> str:
    """Resolve a checkpoint source to a local file path.

    Supports:
    - Local file paths (returned as-is)
    - wandb artifact paths (prefixed with "wandb:"): downloads the
      artifact and returns the path to the .pt file inside it.

    Example: "wandb:brendanlong-com/byte-pe/artifact-name:v0"
    """
    if not source.startswith("wandb:"):
        return source

    import wandb as wb

    artifact_name = source.removeprefix("wandb:")
    print(f"  Downloading wandb artifact: {artifact_name}")
    api = wb.Api()
    artifact = api.artifact(artifact_name)
    download_dir = artifact.download(
        root=f"data/selective_pe/wandb_checkpoints/{artifact.name}",
    )

    # Find the .pt file in the download directory
    import os

    pt_files = [f for f in os.listdir(download_dir) if f.endswith(".pt")]
    assert len(pt_files) == 1, (
        f"Expected 1 .pt file in artifact, found {len(pt_files)}: {pt_files}"
    )
    return os.path.join(download_dir, pt_files[0])


def extract_frozen_deltas(source: str) -> Tensor:
    """Extract per-token deltas from a trained selective PE checkpoint.

    Args:
        source: Local path or "wandb:org/project/artifact:version"

    Loads the checkpoint, runs the DeltaMLP on all token embeddings,
    and returns a (vocab_size,) tensor of frozen deltas.
    """
    checkpoint_path = _resolve_checkpoint_path(source)
    ckpt = torch.load(checkpoint_path, weights_only=True)
    src_config = SelectivePEModelConfig(**ckpt["model_config"])
    src_model = SelectivePETransformer(src_config)
    src_model.load_state_dict(ckpt["model_state_dict"])
    src_model.eval()

    assert src_model.delta_mlp is not None, (
        f"Checkpoint {checkpoint_path} is not a selective PE model"
    )

    with torch.no_grad():
        all_ids = torch.arange(src_config.vocab_size).unsqueeze(0)
        embeddings = src_model.tok_emb(all_ids)
        deltas = src_model.delta_mlp.get_deltas(embeddings).squeeze(0)

    print(f"  Extracted frozen deltas from {source}")
    print(
        f"  delta stats: mean={deltas.mean():.3f}, "
        f"std={deltas.std():.3f}, "
        f"min={deltas.min():.3f}, max={deltas.max():.3f}"
    )
    return deltas


def alibi_slopes(n_heads: int) -> Tensor:
    """Standard ALiBi per-head slopes: geometric sequence 2^(-8h/n).

    For n_heads=8 this is 2^-1, 2^-2, ..., 2^-8.
    """
    ratio = 2.0 ** (-8.0 / n_heads)
    return torch.tensor([ratio ** (h + 1) for h in range(n_heads)])


def causal_decay_mask(bias: Tensor, seq_len: int) -> Tensor:
    """Apply causal masking (-inf above the diagonal) to a decay bias.

    bias: (..., S, S) additive attention bias (<= 0 below diagonal).
    """
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=bias.device)
    )
    return bias.masked_fill(~causal, float("-inf"))


class ForgetMLP(nn.Module):
    """Per-head, per-token forget gate: emits log(alpha) <= 0, alpha in (0,1).

    Computes a per-head decay rate from token embeddings. The cumulative sum of
    log-alpha gives an additive attention bias: query i attending to key j (in
    head h) is penalized by (A_i^h - A_j^h) where A_t^h = sum_{k<=t} log alpha_k^h,
    down-weighting distant past tokens. A learned, content-dependent, per-head
    generalization of ALiBi's fixed per-head slopes. Initialized so alpha ~ 1
    (near-zero decay), i.e. starts close to plain RoPE.
    """

    def __init__(self, dim: int, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_heads)
        # sigmoid(6.9) ~ 0.999 -> log(alpha) ~ -1e-3: minimal decay at init
        nn.init.constant_(self.fc2.bias, 6.9)

    def log_alpha(self, x: Tensor) -> Tensor:
        """Return log(alpha) <= 0 per token per head. (B,S,dim) -> (B,S,H)."""
        h = F.gelu(self.fc1(x))
        return F.logsigmoid(self.fc2(h))

    def decay_mask(self, x: Tensor) -> Tensor:
        """Build the (B, H, S, S) additive causal+decay attention bias."""
        # (B,S,H) -> (B,H,S) so cumsum runs over the last (contiguous) dim.
        # A middle-dim scan trips a torch.compile/inductor codegen bug.
        la = self.log_alpha(x).float().permute(0, 2, 1).contiguous()  # (B, H, S)
        cum = torch.cumsum(la, dim=-1)  # A_t^h, decreasing
        # bias[b,h,i,j] = A_i^h - A_j^h  (<= 0 for i >= j)
        bias = cum.unsqueeze(3) - cum.unsqueeze(2)  # (B, H, S, S)
        return causal_decay_mask(bias, x.shape[1])


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


def _create_norm(config: SelectivePEModelConfig) -> nn.Module:
    """Create normalization layer based on config."""
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.dim, eps=config.norm_eps)
    return nn.RMSNorm(config.dim, eps=config.norm_eps)


class Attention(nn.Module):
    """Multi-head self-attention supporting all 4 PE variants."""

    def __init__(self, config: SelectivePEModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.pos_encoding = config.pos_encoding

        self.wq = nn.Linear(
            config.dim,
            config.n_heads * config.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            config.dim,
            config.n_heads * config.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            config.dim,
            config.n_heads * config.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            config.n_heads * config.head_dim,
            config.dim,
            bias=False,
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        rope_freqs: Tensor,
        batched_freqs: Tensor | None = None,
        decay_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with positional encoding.

        Args:
            x: (batch, seq_len, dim)
            rope_freqs: (max_seq_len, head_dim//2) complex (for rope/pope)
            batched_freqs: (batch, seq_len, head_dim//2) complex
                           (for selective variants)
            decay_mask: (batch_or_1, n_heads, seq_len, seq_len) additive
                        attention bias combining causal masking + per-head
                        decay (ALiBi or learned forget gate). When provided,
                        replaces the built-in causal mask.
        """
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_heads, self.head_dim)

        if self.pos_encoding == "rope":
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)
        elif self.pos_encoding == "pope":
            q = apply_pope(q, rope_freqs)
            k = apply_pope(k, rope_freqs)
        elif self.pos_encoding == "selective_rope":
            assert batched_freqs is not None
            q = apply_rope_batched(q, batched_freqs)
            k = apply_rope_batched(k, batched_freqs)
        elif self.pos_encoding == "selective_pope":
            assert batched_freqs is not None
            q = apply_pope_batched(q, batched_freqs)
            k = apply_pope_batched(k, batched_freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if decay_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=decay_mask.to(q.dtype), is_causal=False
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.resid_dropout(self.wo(out))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GELUFeedForward(nn.Module):
    """GELU feed-forward network."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.gelu(self.w_up(x))))


def _create_ffn(config: SelectivePEModelConfig) -> nn.Module:
    """Create feed-forward network based on config."""
    if config.activation == "gelu":
        return GELUFeedForward(
            config.dim,
            config.intermediate_dim,
            config.dropout,
        )
    return SwiGLUFeedForward(
        config.dim,
        config.intermediate_dim,
        config.dropout,
    )


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention + FFN."""

    def __init__(self, config: SelectivePEModelConfig) -> None:
        super().__init__()
        self.attn_norm = _create_norm(config)
        self.attn = Attention(config)
        self.ffn_norm = _create_norm(config)
        self.ffn = _create_ffn(config)

    def forward(
        self,
        x: Tensor,
        rope_freqs: Tensor,
        batched_freqs: Tensor | None = None,
        decay_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs, batched_freqs, decay_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class SelectivePETransformer(nn.Module):
    """Transformer LM with configurable positional encoding.

    For selective variants, a DeltaMLP computes per-token position strides.
    The cumulative sum gives real-valued positions used for RoPE/PoPE rotation.

    Two delta modes:
    - Shared (default): single DeltaMLP on token embeddings, same positions
      for all layers.
    - Per-layer (per_layer_delta=True): separate DeltaMLP per layer, each
      computing positions from the current hidden state. This allows
      different layers to operate at different positional granularities
      (e.g., early layers at character-level, late layers at sentence-level).
    """

    def __init__(self, config: SelectivePEModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = _create_norm(config)

        # Precomputed RoPE/PoPE frequencies for standard variants
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(
                config.head_dim,
                config.max_seq_len,
                config.rope_theta,
            ),
            persistent=False,
        )

        # Delta MLP(s) for selective variants
        self._is_selective = config.pos_encoding in (
            "selective_rope",
            "selective_pope",
        )
        self.delta_mlp: DeltaMLP | None = None
        self.delta_mlps: nn.ModuleList | None = None
        self.frozen_delta: FrozenDeltaLookup | None = None

        delta_heads = config.n_heads if config.per_head_delta else 1
        if self._is_selective:
            if config.per_layer_delta:
                self.delta_mlps = nn.ModuleList(
                    [
                        DeltaMLP(
                            config.dim,
                            config.effective_delta_hidden_dim,
                            bounded=True,
                            n_heads=delta_heads,
                        )
                        for _ in range(config.n_layers)
                    ]
                )
            else:
                self.delta_mlp = DeltaMLP(
                    config.dim,
                    config.effective_delta_hidden_dim,
                    n_heads=delta_heads,
                )

        # Attention decay (recency bias), composes with any pos_encoding
        self.forget_mlp: ForgetMLP | None = None
        if config.decay == "learned":
            self.forget_mlp = ForgetMLP(
                config.dim, config.effective_delta_hidden_dim, config.n_heads
            )
        elif config.decay == "alibi":
            # Fixed per-head ALiBi bias: (H, max_seq_len, max_seq_len).
            # dist[i,j] = i - j (>=0 for past keys j<=i; 0 above the diagonal,
            # which is masked to -inf anyway).
            pos = torch.arange(config.max_seq_len)
            dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0).float()
            slopes = alibi_slopes(config.n_heads)  # (H,)
            bias = -slopes[:, None, None] * dist[None, :, :]  # (H, S, S)
            self.register_buffer(
                "alibi_bias",
                causal_decay_mask(bias, config.max_seq_len),
                persistent=False,
            )

        if config.init_std is not None:
            self._init_weights(config.init_std)

    def _decay_mask(self, x: Tensor, input_ids: Tensor) -> Tensor | None:
        """Build the attention decay mask for the configured decay mode."""
        if self.config.decay == "learned":
            assert self.forget_mlp is not None
            return self.forget_mlp.decay_mask(x)
        if self.config.decay == "alibi":
            assert isinstance(self.alibi_bias, Tensor)
            s = input_ids.shape[1]
            return self.alibi_bias[:, :s, :s].unsqueeze(0)  # (1, H, S, S)
        return None

    def _init_weights(self, std: float) -> None:
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        # Re-initialize DeltaMLP bias(es) after general init zeroed them
        for mlp in self._all_delta_mlps():
            mlp._init_bias()
        # Re-init ForgetMLP bias so alpha ~ 1 (near-zero decay) at start
        if self.forget_mlp is not None:
            nn.init.constant_(self.forget_mlp.fc2.bias, 6.9)

    def _all_delta_mlps(self) -> list[DeltaMLP]:
        """Return all DeltaMLP instances (shared or per-layer)."""
        if self.delta_mlps is not None:
            return list(self.delta_mlps)  # type: ignore[arg-type]
        if self.delta_mlp is not None:
            return [self.delta_mlp]
        return []

    def _compute_batched_freqs(
        self,
        x: Tensor,
        layer_idx: int | None = None,
        input_ids: Tensor | None = None,
    ) -> Tensor | None:
        """Compute position frequencies for selective variants.

        Args:
            x: hidden states to compute positions from
            layer_idx: if per-layer, which layer's DeltaMLP to use
            input_ids: token IDs (used by FrozenDeltaLookup)

        Returns:
            (batch, seq_len, head_dim//2) complex, or None for standard PE
        """
        if not self._is_selective:
            return None

        if self.frozen_delta is not None:
            assert input_ids is not None
            positions = self.frozen_delta(input_ids)
        elif self.delta_mlps is not None:
            assert layer_idx is not None
            positions = self.delta_mlps[layer_idx](x)
        elif self.delta_mlp is not None:
            positions = self.delta_mlp(x)
        else:
            return None

        return compute_rope_freqs_for_positions(
            positions,
            self.config.head_dim,
            self.config.rope_theta,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.tok_emb(input_ids)

        # Shared delta: compute once from embeddings (or frozen lookup)
        batched_freqs = (
            self._compute_batched_freqs(x, input_ids=input_ids)
            if not self.config.per_layer_delta
            else None
        )
        # Decay (forget gate / ALiBi): one mask, shared across layers
        decay_mask = self._decay_mask(x, input_ids)

        assert isinstance(self.rope_freqs, Tensor)
        for layer_idx, layer in enumerate(self.layers):
            # Per-layer delta: compute from current hidden state
            if self.config.per_layer_delta:
                batched_freqs = self._compute_batched_freqs(x, layer_idx)
            x = layer(x, self.rope_freqs, batched_freqs, decay_mask)
        x = self.final_norm(x)
        # Tied embeddings
        return F.linear(x, self.tok_emb.weight)

    def forward_with_layer_hooks(
        self,
        input_ids: Tensor,
        hook: Callable[[Tensor, int], Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass that calls hook(residual, layer_idx) after each layer.

        Returns:
            logits: (batch, seq_len, vocab_size)
            residuals: list of post-hook (batch, seq_len, dim) tensors
        """
        residuals: list[Tensor] = []
        x = self.tok_emb(input_ids)

        batched_freqs = (
            self._compute_batched_freqs(x, input_ids=input_ids)
            if not self.config.per_layer_delta
            else None
        )
        decay_mask = self._decay_mask(x, input_ids)

        assert isinstance(self.rope_freqs, Tensor)
        for layer_idx, layer in enumerate(self.layers):
            if self.config.per_layer_delta:
                batched_freqs = self._compute_batched_freqs(x, layer_idx)
            x = layer(x, self.rope_freqs, batched_freqs, decay_mask)
            x = hook(x, layer_idx)
            residuals.append(x)
        x = self.final_norm(x)
        return F.linear(x, self.tok_emb.weight), residuals

    @torch.no_grad()
    def get_deltas(self, input_ids: Tensor) -> Tensor | None:
        """Return per-token position deltas for interpretability.

        For shared delta: returns (batch, seq_len) from embedding layer.
        For per-layer delta: returns (n_layers, batch, seq_len) by running
        a forward pass and collecting each layer's deltas.

        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            deltas tensor, or None if not a selective variant
        """
        if not self._is_selective:
            return None

        if self.frozen_delta is not None:
            return self.frozen_delta.get_deltas(input_ids)

        if self.delta_mlp is not None:
            # Shared: deltas from embeddings
            x = self.tok_emb(input_ids)
            return self.delta_mlp.get_deltas(x)

        if self.delta_mlps is not None:
            # Per-layer: run forward, collect deltas at each layer
            x = self.tok_emb(input_ids)
            layer_deltas: list[Tensor] = []
            assert isinstance(self.rope_freqs, Tensor)
            for layer_idx, layer in enumerate(self.layers):
                delta_mlp = self.delta_mlps[layer_idx]
                assert isinstance(delta_mlp, DeltaMLP)
                deltas = delta_mlp.get_deltas(x)
                layer_deltas.append(deltas)
                batched_freqs = self._compute_batched_freqs(x, layer_idx)
                x = layer(x, self.rope_freqs, batched_freqs)
            return torch.stack(layer_deltas)  # (n_layers, B, S)

        return None

    def install_frozen_deltas(self, frozen_deltas: Tensor) -> None:
        """Replace the learned DeltaMLP with a frozen lookup table.

        The model keeps its selective PE pos_encoding type but uses
        fixed per-token deltas instead of the learned MLP. The DeltaMLP
        is removed so its parameters aren't trained.
        """
        self.frozen_delta = FrozenDeltaLookup(frozen_deltas)
        # Remove the learned MLP so it's not trained
        self.delta_mlp = None
        self.delta_mlps = None

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def print_model_summary(model: SelectivePETransformer) -> None:
    """Print a summary of model architecture and parameter counts."""
    config = model.config

    emb_params = model.tok_emb.weight.numel()
    layer_params = sum(p.numel() for p in model.layers.parameters())
    norm_params = sum(p.numel() for p in model.final_norm.parameters())
    delta_params = sum(
        p.numel() for mlp in model._all_delta_mlps() for p in mlp.parameters()
    )
    total = model.count_parameters()

    pe_label = config.pos_encoding
    if config.per_layer_delta:
        pe_label += " (per-layer)"

    print(f"\n{'=' * 50}")
    print(f"SelectivePE Transformer ({pe_label})")
    print(f"{'=' * 50}")
    print(f"  Hidden dim:     {config.dim}")
    print(f"  Heads:          {config.n_heads}")
    print(f"  Layers:         {config.n_layers}")
    print(f"  MLP dim:        {config.intermediate_dim}")
    print(f"  Vocab size:     {config.vocab_size}")
    print(f"  Context len:    {config.max_seq_len}")
    print(f"  Pos encoding:   {config.pos_encoding}")
    print("\nParameters:")
    print(f"  Embeddings:     {emb_params:>12,}")
    print(f"  Layers:         {layer_params:>12,}")
    if delta_params > 0:
        print(f"  Delta MLP:      {delta_params:>12,}")
    print(f"  Final norm:     {norm_params:>12,}")
    print(f"  Total:          {total:>12,}")
    print(f"{'=' * 50}\n")
