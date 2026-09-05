# Learned Position Increments

Code for the experiments behind the blog post [**"How Far Apart Does a Model Think Its Tokens Are?"**](https://www.brendanlong.com/how-far-apart-does-a-model-think-its-tokens-are.html) (also on [LessWrong](https://www.lesswrong.com/posts/Bxju8Fmpo2eW4oj9t/how-far-apart-does-a-model-think-its-tokens-are)).

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
# (the exact configuration behind the blog post; see "Reproducing the blog
# post" below for what each flag does)
RUN_NAME=perlayer ./train.sh local selective_pe -- \
  --pos-encoding selective_rope --per-layer-delta \
  --dataset bilingual --tokenizer byte --max-stories 60000 --val-max-docs 2000 \
  --context-len 512 --generate-n 10000000 --total-steps 50000 \
  --batch-size 64 --grad-accum-steps 1 --lr 1e-3 --weight-decay 0.01 \
  --warmup-steps 500 --save-checkpoint

# Graft deltas onto SmolLM2 with LoRA
./train.sh remote delta_graft -- \
  --base-model HuggingFaceTB/SmolLM2-1.7B --adapt lora \
  --total-steps 2000 --save-checkpoint
```

`RUN_NAME` namespaces the S3 checkpoint path; `--save-checkpoint` also uploads
the final checkpoint as a wandb artifact named `selective-pe-<wandb-run-name>`,
which is the easiest way to get it back from a remote job.

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

## Reproducing the blog post

The two byte-level bilingual models in the post are wandb runs
[`rt7cv5ww`](https://wandb.ai/brendanlong-com/byte-pe/runs/rt7cv5ww) (shared
DeltaMLP) and [`ho7v6eou`](https://wandb.ai/brendanlong-com/byte-pe/runs/ho7v6eou)
(per-layer). Both were trained with exactly these arguments (the per-layer run
adds `--per-layer-delta`); note that several differ from the CLI defaults:

```bash
uv run python -m experiments.selective_pe.train \
  --dataset bilingual --tokenizer byte --pos-encoding selective_rope \
  --max-stories 60000 --val-max-docs 2000 --context-len 512 \
  --generate-n 10000000 --total-steps 50000 --batch-size 64 --grad-accum-steps 1 \
  --lr 1e-3 --weight-decay 0.01 --warmup-steps 500 \
  --eval-every-steps 1000 --log-every-steps 200 --save-every-steps 10000 \
  --wandb-project byte-pe --save-checkpoint
```

What this trains on: the first 60,000 documents of a round-robin interleave of
English and Chinese Wikipedia (`20231101.en` / `20231101.zh`), about 380 MB of
UTF-8 bytes, with the first 2,000 interleaved documents (about 41 MB) held out
for validation. Training draws 50,000 × 64 random 512-byte windows, i.e. about
1.6 GB, so each byte is seen roughly four times at different offsets. Without
`--max-stories` the loader tries to stream all of Wikipedia into RAM.

Two caveats about the original runs:

- They were trained before the train/val leakage fix now in `data.py` (the
  train stream skips the reserved validation documents). Retrained models see
  slightly different data and score a val loss of about 1.07 instead of the
  1.045 logged on the original runs, and the "held-out" Chinese Wikipedia
  text used by `byte_boundary_analysis.py` was part of the original models'
  training data. All of the post's qualitative results hold for retrained
  models (see the replica run below).
- The shared model's checkpoint is the wandb artifact
  `brendanlong-com/byte-pe/selective-pe-byte_selective_rope_bilingual_256d_8h_6L_lr1e3:v0`.
  The per-layer model's artifact was lost. Replicas of both models trained
  from this repo with the command above are wandb runs `q19navjs` (shared)
  and `0eqetsep` (per-layer), artifacts `selective-pe-repro_*:v0`; see
  `experiments/selective_pe/REPRODUCTION.md` for a side-by-side comparison
  of every table in the post.

The analysis commands that produced the post's tables and figures:

```bash
# Per-byte δ table (Part 1) and Chinese word-boundary AUC vs jieba (Part 2)
uv run python -m experiments.selective_pe.byte_boundary_analysis \
  --shared <shared_ckpt> --per-layer <perlayer_ckpt>
# Multi-word-entity reversed-order control
uv run python -m experiments.selective_pe.entity_grouping_analysis \
  --shared <shared_ckpt> --per-layer <perlayer_ckpt>
# Character-spacing figures (per-row normalization, then the shared-scale one)
uv run python -m experiments.selective_pe.plot_increments \
  --shared <shared_ckpt> --per-layer <perlayer_ckpt> --out-dir data/plots
uv run python -m experiments.selective_pe.plot_increments \
  --per-layer <perlayer_ckpt> --scale shared --out-dir data/plots
```

The post's "Punctuation" row is the script's sentence-punctuation category
(`. ! ?`); `, ; : -` are reported separately as "Other punct".

`entity_grouping_analysis.py` here is the version the post's table came from
(79 entity pairs). A later revision of the analysis dedupes pairs that share a
word (59 pairs kept, since the causal measurement makes such pairs identical
datapoints); the effect survives with L2 p=5e-4, L4 p=2e-5, L5 p=9e-5.

The SmolLM2 fine-tune shown at the end of the post is wandb project
`delta-graft`; its DeltaMLP checkpoints are the artifacts
`delta-graft-SmolLM2-1.7B_lora_delta:v0` and `delta-graft-SmolLM2-1.7B_full_delta:v0`,
and `experiments/delta_graft/RESULTS.md` is regenerated by
`analyze_deltas.py --checkpoint wandb:brendanlong-com/delta-graft/<artifact>`.

## Results write-ups

- `experiments/selective_pe/RESULTS.md`
- `experiments/delta_graft/RESULTS.md`

## License

MIT — see [LICENSE](LICENSE).
