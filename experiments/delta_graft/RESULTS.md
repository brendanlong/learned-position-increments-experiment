# Delta-Graft: adding learned position deltas to a pretrained model

**Question.** Can interpretable per-token learned position rates (deltas, from
the `selective_pe` experiment) be grafted onto an *off-the-shelf, open-data*
pretrained model with minimal continued training and **without hurting
perplexity**? If so, "take a real model and add interpretable learned positions
on top" becomes a practical post-hoc interpretability tool.

We use open-data models so we can fine-tune on (approximately) the original
training distribution: **SmolLM2** (FineWeb-Edu) for cheap signal, then
optionally **OLMo 2 7B** (Dolma). Frontier models (Llama/Qwen/GPT) have closed
data, so the clean same-distribution test is only possible on open-data models.

## TL;DR

Grafting learned deltas onto SmolLM2-1.7B is **loss-neutral** — in both LoRA
and full fine-tuning, the delta run's validation curve is indistinguishable
from a no-delta baseline given the *same* continued training (≤ 0.001 nats at
every step). The deltas nonetheless learn **clean, interpretable per-token
structure**, but as a tiny (~1–3%) nudge: a pretrained model already has a
near-optimal integer-position solution, so the deltas only fine-tune around it.
This reproduces the from-scratch `selective_pe` finding on a real model:
learned positions are a loss-neutral *re-parametrization* that the model
organizes linguistically but does not need.

## Mechanism

SmolLM2 and OLMo 2 are `LlamaForCausalLM`: RoPE is computed once at the model
level from `position_ids` and broadcast to every attention layer. We:

1. Read token embeddings, run a small **DeltaMLP** → per-token stride δ_t > 0.
2. Take an **exclusive** cumsum (token 0 → position 0) → fractional positions.
3. Pass those as `position_ids` (they may be float).

The DeltaMLP's output weight is **zero-initialized** (bias gives δ ≡ 1), so at
init the fractional positions are exactly `[0, 1, 2, …]` and the wrapped model
is **bit-identical to the base** (verified: max logit diff = 0.0). Any later
change is attributable purely to the learned deltas — the identity-at-init
property LoRA gets from its zero B matrix.

### Implementation gotchas (each cost a debugging cycle)

- **HF rotary is `@torch.no_grad()`.** `LlamaRotaryEmbedding.forward` is wrapped
  in no_grad for inference, which severs the gradient from the loss back to
  `position_ids`. Our learned positions *are* `position_ids`, so the DeltaMLP
  never learned (δ frozen at exactly 1.000). Fix: `GradRotaryEmbedding`, a
  gradient-enabled clone with identical math. This is the only base-model
  surgery; it changes no weights and reproduces base outputs at identity.
- **Gradient checkpointing silently discards the positions.** HF forces
  `use_cache=False` during checkpointed training, and in this transformers
  version `use_cache=False` makes the model regenerate integer `position_ids`,
  ignoring ours (verified: 28-nat divergence with fractional positions). So GC
  is unusable here. Full FT is fit into a 48 GB A40 via fp32 + small batch.
- **bf16 full FT doesn't learn.** At LR 2e-5, weight updates underflow bf16
  (1.0 + 2e-5 rounds back to 1.0). Full FT runs in fp32; LoRA in bf16.
- **Positions must be fp32.** bf16 spacing at position 1024 is ~8, so a bf16
  cumsum cannot represent distinct long-context positions. DeltaMLP + cumsum
  run fp32 with autocast disabled.

## Result 1 — perplexity is unchanged (loss-neutral)

SmolLM2-1.7B, FineWeb-Edu, 1024 ctx, held-out val. Each delta run is paired with
an identical `--delta-disabled` baseline (integer positions, same data/steps/LR)
so the delta effect is isolated from plain continued-training gains.

| mode | steps | base ppl | delta ppl | baseline ppl | **delta effect** |
|------|-------|----------|-----------|--------------|------------------|
| LoRA (q/k/v/o + DeltaMLP) | 2000 | 9.660 | 9.666 | 9.653 | **+0.001 (noise)** |
| full FT (fp32) | 800 | 9.69 | 9.469 | 9.471 | **−0.0001 (none)** |

Full FT improves ppl by 0.023 nats — but the no-delta baseline improves by the
*same* 0.023. The delta-vs-baseline curves overlay step-for-step (within ±0.0004
nats), including the sharp drop around step 300 (9.66 → 9.58) that appears
identically in both — that S-curve is a continued-training / LR-schedule
artifact, not a delta effect:

| step | full **delta** vs base | full **baseline** vs base |
|------|------------------------|---------------------------|
| 100  | −0.0013 | −0.0016 |
| 200  | −0.0032 | −0.0032 |
| 300  | −0.0117 | −0.0120 |
| 400  | −0.0177 | −0.0176 |
| 800  | −0.0233 | −0.0234 |

LoRA (base frozen, only adapters + DeltaMLP added) is the cleanest "does it
hurt?" test: ppl stays flat at +0.0007 nats. Adding the delta mechanism is free.

## Result 2 — the deltas learn interpretable structure (the biggest movers)

Running the trained DeltaMLP over the full 49 k vocab (exact for LoRA — frozen
embeddings; approximate for full FT — embeddings also drift). δ = 1 means "no
change from integer position"; δ > 1 advances position faster (more boundary-
like), δ < 1 advances slower (glued to neighbors). **What matters is the
relative drift across tokens, not the absolute level** — see the note below.

### Full fine-tune (mean δ 1.002, std 0.003, EOS 1.014)

Keys on **complete, frequent tokens**. The biggest *upward* movers are function
words and punctuation; the biggest *downward* movers are word-initial fragments
that will be continued.

| | tokens (largest change) |
|---|---|
| **highest δ** (1.016–1.021) | `Ġand Ġto Ġin Ġfor . Ġwith , Ġas Ġof Ġby Ġis Ġor Ġfrom Ġon Ġat Ġwhen Ġif Ġthat Ġbut Ġthe` |
| **lowest δ** (0.985–0.990) | `Ġpries(t) Ġpsychiat(ric) Ġstrugg(le) Ġneighb(or) Ġencou(rage) Ġacknow(ledge) Ġappre(ciate) Ġseiz(e) Ġtradem(ark) Ġnewsp(aper)` |

category medians: other_punct 1.011 > function_word 1.008 > sentence_punct
1.007 > word_start 1.0024 > subword_continuation 1.0017.

### LoRA (mean δ 1.033, std 0.011, EOS 1.053)

Larger spread, and keys on **word position** rather than word identity. The
biggest *upward* movers are word-*final* fragments, newline, colon, and EOS;
the biggest *downward* movers are single letters and word-*initial* fragments.

| | tokens (largest change) |
|---|---|
| **highest δ** (1.08–1.13) | `forgettable paralleled (lawsu)it emeteries (disen)franch (ad)vantages (T)olkien (sym)ptomatic : astrous(disastrous) (contra)ceptives (sur)mountable Ċ(newline)` |
| **lowest δ** (0.984–0.995) | `c aph ad w U ph ich em end d Ġprel ĠSt b ex ch im ra con` |

category medians: EOS 1.053 > other_punct 1.044 > sentence_punct 1.042 >
subword_continuation 1.034 > word_start 1.032 > **function_word 1.028** (lowest).

### The contrast is the interesting part

Both modes independently put **EOS and punctuation high** (document/sentence
boundaries advance position most) — the landmark from the from-scratch
experiment. But they disagree sharply on everything else:

- **Full FT: function words high, content-word prefixes low.** It advances after
  complete frequent units and holds position through the start of a long word.
- **LoRA: word-endings high, function words and word-starts low.** It advances
  after a word *completes* and holds through word-internal pieces.

Both are coherent **"advance at word/clause boundaries"** heuristics — they just
latch onto different, correlated surface cues for "this token ends a unit." The
loss is indifferent to which cue is used (it's loss-neutral), so the two
optimizations settled on different encodings of the same idea. This *is* the
finding: learned positions are underdetermined; many equivalent solutions exist,
and which one you get depends on the optimization, not the objective.

## Why "compare relative drift," not absolute δ

RoPE attention depends on query–key position *differences*. A constant δ ≡ c is
just uniform spacing — a single global "time-dilation" scalar (it scales every
rotary frequency equally) with **zero per-token structure**. All the
interpretable signal is in the *deviations* from the per-run mean — which tokens
drift up vs down, and their rank order. So to compare runs, mean-center (or
z-score) and compare the *pattern*, not the level.

The absolute spread answers a *different* question — *how hard the model leans
on the mechanism.* From scratch it can be huge (EOS δ = 17 in `selective_pe`);
grafted onto a pretrained model it is tiny (std 0.003–0.011) because the model
already has a good positional solution and only nudges. Tiny spread ≠ no signal.

## Throughput / cost (for a larger run)

A40, SmolLM2-1.7B, FineWeb-Edu streaming (worker-sharded tokenization):

- LoRA bf16, eff. batch 32, 1024 ctx: ~0.2 it/s → 2000 steps ≈ 2.6 h.
- full FT fp32, eff. batch 32: ~0.13 it/s → 800 steps ≈ 1.8 h.

A 7B run needs an 80 GB GPU for full FT (no GC available) and would be ~4× the
compute; LoRA is the realistic "cheap, off-the-shelf" path at that scale. Given
the 1.7B result is cleanly loss-neutral with interpretable structure, a 7B run
would mainly test whether the *structure* sharpens with scale — not whether ppl
improves (it won't).

## Runs (wandb project `delta-graft`)

| run | mode | steps | final val ppl | vs base | artifact |
|-----|------|-------|---------------|---------|----------|
| SmolLM2-1.7B_lora_delta    | LoRA | 2000 | 9.666 | +0.0007 | `delta-graft-SmolLM2-1.7B_lora_delta` |
| SmolLM2-1.7B_lora_baseline | LoRA | 2000 | 9.653 | −0.0007 | — |
| SmolLM2-1.7B_full_delta    | full | 800  | 9.469 | −0.0233 | `delta-graft-SmolLM2-1.7B_full_delta` |
| SmolLM2-1.7B_full_baseline | full | 800  | 9.471 | −0.0234 | — |

## Files

- `model.py` — `DeltaGraftModel`, `GradRotaryEmbedding`, `build_delta_graft_model`.
- `data.py` — FineWeb-Edu streaming, packed windows, worker-sharded, held-out val.
- `train.py` — training loop, step-0 identity baseline, `val/loss_vs_base`,
  delta logging, final DeltaMLP checkpoint upload.
- `analyze_deltas.py` — per-token δ over the vocab, by category + biggest movers
  (robust to unused-token softplus blow-ups).
- `skypilot/train-delta-graft.yaml` — SkyPilot task fragment.
