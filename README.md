# Learned Position Increments

Code for the experiments behind the post **"Learning the Distance Between
Context Positions"** (LessWrong — link TBD).

Standard RoPE gives every token an integer position, so the rotation between two
tokens depends only on how many tokens apart they are. This repo asks a
different question: **what if the model could learn how far to advance the
position at each token?** A small MLP reads each token's embedding and emits a
positive stride δ_t; the position is the running sum p_t = Σ_{s≤t} δ_s, and RoPE
rotates by those (now fractional) positions. The relative property is preserved
(attention still depends only on position *differences*), but the model can now
pack tokens close together (small δ) or spread them apart (large δ).

The headline finding is that these **learned increments are loss-neutral but
encode interpretable linguistic structure**: boundaries (EOS, punctuation,
word/clause edges) advance position most, and mid-layers discover structure that
the byte stream does not mark — Chinese word boundaries, multi-word-entity
continuation — even though a from-scratch integer-position model does just as
well on perplexity.

## Two experiments

### `experiments/selective_pe/` — from-scratch byte/BPE models

A small Transformer (default 256d / 8h / 6L) trained from scratch with one of
four positional-encoding variants (`rope`, `pope`, `selective_rope`,
`selective_pope`). The "selective" variants add a **DeltaMLP**
(`Linear → GELU → Linear → softplus`, bias-initialized so δ ≈ 1 at start) whose
output stride is shared across layers, per-layer, or per-head. Trained on
SimpleStories (BPE) or a bilingual English/Chinese byte corpus (vocab 257 =
256 byte values + EOS).

Analysis / figure scripts:

- `plot_increments.py` — render text with each character placed at its
  cumulative learned position, so within-word characters pack and boundaries
  spread (single-row shared model, or stacked per-layer rows). Bilingual.
- `byte_boundary_analysis.py` — per-byte category δ table, plus a ROC-AUC of
  the learned increment against `jieba` Chinese word segmentation on held-out
  Wikipedia (the byte stream has no spaces between Chinese words).
- `entity_grouping_analysis.py` — reversed-order control testing whether the
  increment at the internal space of a two-word entity ("New York") is lower for
  the real order than the reverse. (Measures causal *forward-continuation*
  expectation that tracks entity bigrams — see the script's docstring for the
  honest framing of what this does and does not show.)
- `indirect_indexing.py` / `train_diagnostic.py` — the PoPE indirect-indexing
  diagnostic task (a side check on content-conditional positional queries).

### `experiments/delta_graft/` — grafting deltas onto a pretrained model

Tests whether the same learned-increment mechanism can be **added on top of an
off-the-shelf, open-data pretrained model** (SmolLM2, then optionally OLMo 2)
with minimal continued training and without hurting perplexity. The DeltaMLP's
output weight is zero-initialized, so at init the wrapped model is bit-identical
to the base; any change is attributable to the learned deltas. See
`experiments/delta_graft/RESULTS.md` for the loss-neutral result and the
interpretable per-token δ structure (and the HF gotchas: gradient-severing
`@torch.no_grad` rotary, gradient-checkpointing dropping float positions, bf16
underflow).

`analyze_deltas.py` runs the trained DeltaMLP over the full vocab and reports
the biggest movers by token category.

## Setup

```bash
uv sync
```

Uses [uv](https://github.com/astral-sh/uv). Python ≥ 3.12, PyTorch ≥ 2.0.
`ruff` (lint/format) and `pyright` (types) are the dev tooling.

## Running

Training goes through SkyPilot via `train.sh` (`local` uses a local GPU, `remote`
uses RunPod managed jobs, `vast` uses Vast.ai):

```bash
# Byte-level bilingual (English/Chinese) model with per-layer selective RoPE
./train.sh local selective_pe -- \
  --pos-encoding selective_rope --per-layer-delta \
  --dataset bilingual --tokenizer byte --total-steps 50000 --save-checkpoint

# Graft deltas onto SmolLM2 with LoRA
./train.sh remote delta_graft -- \
  --base-model HuggingFaceTB/SmolLM2-1.7B --adapt lora \
  --total-steps 2000 --save-checkpoint
```

Or invoke a module directly without SkyPilot:

```bash
uv run python -m experiments.selective_pe.train --help
uv run python -m experiments.delta_graft.train --help
```

Analysis scripts take checkpoint paths:

```bash
uv run python -m experiments.selective_pe.plot_increments \
  --shared <shared_ckpt> --per-layer <perlayer_ckpt> --out-dir data/plots
uv run python -m experiments.selective_pe.byte_boundary_analysis \
  --shared <shared_ckpt> --per-layer <perlayer_ckpt>
uv run python -m experiments.selective_pe.entity_grouping_analysis \
  --shared <shared_ckpt> --per-layer <perlayer_ckpt>
```

> **Note:** the SkyPilot fragments in `skypilot/` sync checkpoints to the
> author's private S3 bucket (`s3://brendanlong-experiments/...`). Point them at
> your own bucket, or drop the `aws s3 sync` line, to reproduce.

## Results write-ups

- `experiments/selective_pe/RESULTS.md`
- `experiments/delta_graft/RESULTS.md`

## License

MIT — see [LICENSE](LICENSE).
