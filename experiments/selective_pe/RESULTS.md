# Selective PE Experiment Results

Comparing positional encoding variants (RoPE, PoPE, Selective RoPE,
Selective PoPE) across datasets, model sizes, and delta-sharing schemes.

## TL;DR

The **selective mechanism is loss-neutral** for language modeling across
every configuration tested (4 PE variants × SimpleStories/CodeParrot ×
19M/152M params × shared/per-layer/frozen deltas). But it learns **clean,
interpretable hierarchical position structure** — function words get
compressed, boundaries get expanded, and per-layer variants assign
different layers to different structural roles (e.g. a layer that isolates
scope delimiters in code). The interpretability artifacts are the finding,
not a perplexity win.

## ⚠️ Data-pipeline caveat (affects runs before 2026-05-28)

All runs in Phases 1-9 below used `num_workers=2` on the streaming train
loader. Because `StreamingChunkDataset` is an `IterableDataset` that seeded
only on `seed + epoch` (no worker-id offset), the two workers yielded the
**same** random chunks — so streaming runs saw each chunk ~2× rather than
fully unique data. This does **not** affect the validity of cross-variant
comparisons (every variant used the identical pipeline), but it undercuts the
"every example is unique" intent.

Fixed 2026-05-28: train loader now uses `num_workers=0`, and the dataset
mixes worker id into its seed. **Runs trained after this fix are not
comparable to the numbers below.**

Verification: re-running the rope baseline (lr=1e-3, 50K steps) with the fix
(`rope_256d_8h_6L_lr1e3_nw0`) gave **val loss 1.755 (ppl 5.78)** vs the old
buggy **1.803 (ppl 6.07)** — a **0.048 nat** improvement from unique data.
This is *larger than any inter-variant difference we measured* (all PE
variants were within ~0.02 nats of each other). So the data-pipeline bug
affected loss more than the positional encoding choice did. The cross-variant
comparisons remain valid (all variants shared the buggy pipeline), but a
clean re-run of all variants with the fix would be needed to report absolute
numbers — and the "selective PE is loss-neutral" conclusion should still hold
since the bug hit every variant equally.

## Model Architecture

- 256d / 8h / 6L, SwiGLU, RMSNorm, tied embeddings
- GPT-2 BPE tokenizer (50257 vocab)
- ~19.2M params (rope/pope), ~19.2M params (selective variants, +16.5K delta MLP)
- Context length: 256 tokens
- bf16 autocast, batch_size=16 x 4 grad_accum = 64 effective

## Phase 1: Hyperparameter Sweep

LR × weight decay sweep on `rope` baseline, 5K steps each (local RTX 3060).

Command template:
```bash
uv run python -m experiments.selective_pe.train \
  --pos-encoding rope --generate-n 500000 --total-steps 5000 \
  --batch-size 16 --grad-accum-steps 4 \
  --lr $lr --weight-decay $wd --warmup-steps 200 \
  --eval-every-steps 1000 --log-every-steps 500 \
  --wandb-run-name "sweep_lr${lr}_wd${wd}" --wandb-project byte-pe
```

| LR   | WD=0.0 | WD=0.01 | WD=0.1 |
|------|--------|---------|--------|
| 1e-4 | 2.895  | 2.897   | 2.912  |
| 3e-4 | 2.363  | 2.363   | 2.373  |
| 1e-3 | 2.146  | **2.144** | 2.146 |
| 3e-3 | 2.174  | 2.148   | 2.118  |

Best at 5K steps: **lr=3e-3, wd=0.1** (2.118), though lr=1e-3 with any WD is within noise.
For full 50K step runs we used **lr=1e-3, wd=0.01** as a safer choice for longer cosine decay.

## Phase 2: Full PE Comparison (lr=3e-4, local GPU)

50K steps, lr=3e-4 (default, before sweep results), local RTX 3060, ~24 steps/sec.

Command template:
```bash
uv run python -m experiments.selective_pe.train \
  --pos-encoding $pe --generate-n 5000000 --total-steps 50000 \
  --batch-size 16 --grad-accum-steps 4 \
  --lr 3e-4 --weight-decay 0.01 --warmup-steps 500 \
  --wandb-run-name "${pe}_256d_8h_6L" --wandb-project byte-pe --save-checkpoint
```

| PE Variant      | Val Loss | Val PPL | wandb run                    |
|-----------------|----------|---------|------------------------------|
| rope            | 1.859    | 6.42    | rope_256d_8h_6L              |
| pope            | 1.870    | 6.49    | pope_256d_8h_6L              |
| selective_rope  | 1.856    | 6.39    | selective_rope_256d_8h_6L    |
| selective_pope  | 1.866    | 6.46    | selective_pope_256d_8h_6L    |

(lr=3e-4 is suboptimal; the tuned lr=1e-3 runs in Phase 3 supersede these.)

## Phase 3: Full PE Comparison (lr=1e-3, RunPod, tuned)

50K steps, lr=1e-3 (from sweep), RunPod GPUs, 30-96 steps/sec depending on GPU.

Command template:
```bash
./train.sh remote selective_pe -- \
  --pos-encoding $pe --generate-n 5000000 --total-steps 50000 \
  --batch-size 16 --grad-accum-steps 4 \
  --lr 1e-3 --weight-decay 0.01 --warmup-steps 500 \
  --wandb-run-name "${pe}_256d_8h_6L_lr1e3" --wandb-project byte-pe --save-checkpoint
```

| PE Variant      | Val Loss | Val PPL | BPT  | Final δ stats                              | wandb run                         |
|-----------------|----------|---------|------|--------------------------------------------|------------------------------------|
| **rope**        | **1.803**| **6.07**| 2.60 | n/a                                        | rope_256d_8h_6L_lr1e3             |
| pope            | 1.819    | 6.17    | 2.63 | n/a                                        | pope_256d_8h_6L_lr1e3             |
| **selective_rope** | **1.803** | **6.04** | 2.60 | mean=1.09, std=1.04, min=0.18, max=17.7 | selective_rope_256d_8h_6L_lr1e3 |
| selective_pope  | 1.820    | 6.17    | 2.63 | mean=0.98, std=2.36, min=0.15, max=43.5   | selective_pope_256d_8h_6L_lr1e3   |

## Phase 4: Large Model (768d/12h/12L, 152M params, lr=6e-4, 100K steps)

Tests the capacity hypothesis — the 19M models have only ~6.3M params in
transformer layers, far below the 125M+ where PoPE-paper effects appear.

| PE Variant     | Val Loss | Val PPL |
|----------------|----------|---------|
| rope           | 1.471    | 4.35    |
| selective_rope | 1.472    | 4.36    |

Scaling 19M → 152M dropped loss 0.33 nats, but the rope/selective gap stayed
zero. Capacity isn't the blocker.

## Phase 5: Byte-level (256d/8h/6L, 6.4M params, lr=1e-3, 512-byte ctx)

Raw UTF-8 bytes (257 vocab). Tests whether forcing the model to discover
character/word/sentence boundaries from bytes reveals a selective advantage.

| PE Variant          | Val Loss | Val PPL | BPB  |
|---------------------|----------|---------|------|
| byte_rope           | 0.522    | 1.69    | 0.75 |
| byte_selective_rope | 0.524    | 1.69    | 0.76 |
| byte_pope           | 0.526    | 1.69    | 0.76 |
| byte_selective_pope | 0.528    | 1.69    | 0.76 |

Same loss-neutral pattern. Spread is 0.006 nats (within noise).

## Phase 6: CodeParrot (256d/8h/6L, lr=1e-3, 50K steps)

Python code: structurally demanding (indentation, brackets, scope).

| PE Variant     | Val Loss | Val PPL |
|----------------|----------|---------|
| code_rope      | 1.544    | 4.69    |
| code_selective_rope | 1.548 | 4.70  |
| code_pope      | 1.598    | 4.94    |
| code_selective_pope | 1.595 | 4.93  |

Code is harder than stories (higher ppl) but the PE ranking is unchanged.

## Phase 7: Frozen Delta (lr=1e-3, SimpleStories)

Extract per-token deltas from the trained selective_rope checkpoint, freeze
them, and train a fresh model with those fixed position rates. Isolates
whether the learned positions help independent of joint-training dynamics.

| Run                   | Positions               | Val Loss |
|-----------------------|-------------------------|----------|
| rope                  | integer                 | 1.803    |
| selective_rope (joint)| learned (jointly)       | 1.803    |
| frozen_selective_rope | learned, then frozen    | 1.802    |

**Frozen deltas match the baseline.** This rules out "joint optimization is
the bottleneck" — the position rates genuinely don't improve prediction even
when handed to the model for free. Standard RoPE already reconstructs the
needed positional information from context.

## Phase 8: Per-Layer Delta (lr=1e-3, 50K steps)

Each layer gets its own DeltaMLP computing positions from the current hidden
state. v1 diverged (deltas → 500K+); v2 uses RMSNorm input + sigmoid-bounded
output (max_delta=10) and is stable.

| Run                          | Val Loss | vs rope |
|------------------------------|----------|---------|
| per-layer v2 (SimpleStories) | 1.799    | −0.004  |
| per-layer v2 (CodeParrot)    | 1.551    | +0.007  |

Per-layer is the only variant to (barely) beat the SimpleStories baseline,
but it's within single-seed noise and didn't help on code.

### Emergent per-layer specialization (CodeParrot)

Mean δ by code token category, per layer:

| Category       | L0   | L1   | L2   | L3   | L4   | L5   |
|----------------|------|------|------|------|------|------|
| newline/indent | 4.82 | 9.96 | 8.35 | 1.54 | 0.92 | 1.25 |
| brackets       | 3.24 | 9.99 | 8.39 | 0.47 | 0.70 | 1.65 |
| operators      | 2.57 | 9.98 | 1.92 | 0.44 | 1.07 | 1.24 |
| punctuation    | 2.24 | 9.98 | 1.36 | 0.24 | 0.67 | 0.95 |
| space          | 0.87 | 0.74 | 0.67 | 0.51 | 1.46 | 0.29 |
| identifiers    | 1.86 | 9.96 | 0.88 | 0.51 | 0.35 | 0.95 |

Layers specialized into distinct roles:
- **L0**: general boundary detector (newline > bracket > operator), spaces compressed
- **L1**: content/whitespace binary, saturated at the max_delta=10 ceiling
  (the bound is binding — worth re-running with a higher cap)
- **L2**: **scope-delimiter detector** — newlines and brackets get ~8, everything
  else ~1. These are exactly the tokens that delimit scope in Python.
- **L3-L5**: mild / near-uniform

This is the strongest emergent-hierarchy evidence: per-layer deltas allocate
different layers to different structural roles, with L2 isolating syntactic
scope boundaries.

## Phase 9: Indirect Indexing Diagnostic (128d/4h/4L, 20K steps)

Synthetic key-value lookup task from the PoPE paper (which reported 95% PoPE
vs 11% RoPE). Validates our PoPE implementation. See `indirect_indexing.py`.

| PE Variant     | Accuracy | Loss  |
|----------------|----------|-------|
| selective_rope | 92.6%    | 0.180 |
| rope           | 90.3%    | 0.204 |
| pope           | 85.4%    | 0.280 |
| selective_pope | 84.9%    | 0.289 |

We did **not** reproduce the PoPE >> RoPE gap. In our setup RoPE variants
beat PoPE variants, and all variants solve the task at 85-93%. Likely our
single-hop lookup differs from the paper's multi-hop indirect indexing, or
the task is too easy at this scale to expose the structural difference.
Flagged as a discrepancy to investigate, not a validated reproduction.

## Per-Token Delta Analysis (shared delta, SimpleStories BPE)

Running the trained DeltaMLP over the full GPT-2 vocab reveals a clean
linguistic hierarchy (selective_rope, lr=1e-3):

| Category                | mean δ |
|-------------------------|--------|
| EOS (document boundary) | 17.7   |
| quote-ending (`."`,`?"`)| 2.1    |
| period / `!` / `?`      | 1.9    |
| content words           | 1.4    |
| function words (the,of,a)| 1.1   |
| subword suffixes (ing,sh)| 0.08-0.17 |

Selective PoPE shows the same ordering but more extreme (EOS=43.5, std=2.36
vs 1.04) — its content/phase decoupling frees the position channel to take
extreme values without distorting content matching. Correlation between
rope and pope per-token deltas: r=0.87.

## Phase 10: Can deltas be a boundary/chunking signal? (segmentation)

Motivated by H-Net / BLT-style decoupled chunking: could a frozen δ signal mark
boundaries for a downsampling model, sidestepping the hard joint chunking
optimization? Tests whether δ *discovers* structure or just detects surface
delimiters. (All runs post-date the `num_workers=0` data fix.)

**English byte δ recovers delimiters cleanly but it's surface detection.**
Thresholded shared-delta bytes detect space/newline/punctuation at
ROC-AUC 0.97, F1 0.93 (P=0.91, R=0.95). Sharp enough to pool on — but the
top-δ bytes are dominated by literal spaces and periods.

**Chinese (no whitespace between words) is the decisive test.** Shared-delta
byte model, held-out zh wiki, jieba reference. Between two CJK characters
(no surface cue): δ at a true word boundary = 0.758 vs mid-word = 0.762;
**ROC-AUC = 0.46 (chance).** No discovery of unmarked word structure.

**Bilingual model (en+zh wiki in one model) confirms within-model.** Shared
delta:

| | English | Chinese |
|---|---------|---------|
| δ at word boundary | 1.05 (space) | 0.91 (char) |
| δ mid-word | 0.77 (letter) | 0.91 (char) |
| boundary/mid ratio | **1.36×** | **1.00×** |
| between-CJK AUC | — | 0.50 (chance) |

English raises δ 1.36× at spaces; Chinese is completely flat at word
boundaries. The model does NOT compensate for missing whitespace. (Chinese
chars do get a higher *uniform* δ: 0.94 vs 0.77 for English letters — a denser
per-character unit, but no word structure.) **Architectural reason:** shared
δ = f(byte identity) only, so it can *only* key on bytes that are themselves
delimiters (spaces/punct). It is structurally incapable of context-dependent
boundary detection.

**Per-layer δ DOES discover unmarked Chinese boundaries (the positive result).**
Per-layer DeltaMLPs compute δ from contextualized hidden states. Bilingual
per-layer model, between-CJK word-boundary AUC by layer:

| Layer | EN space/letter | ZH word-boundary AUC |
|-------|-----------------|----------------------|
| L0 | 4.93× | 0.501 (chance — byte-identity, like shared) |
| L1 | 1.14× | 0.530 |
| **L2** | 1.27× | **0.656** ✓ |
| L3 | 1.00× | 0.384 (δ≈0, noise) |
| **L4** | 2.05× | **0.629** ✓ |
| L5 | 3.44× | 0.501 (marks EN words via space, not ZH) |

L2 and L4 recover unmarked Chinese word boundaries *via context* (no space, no
punct), the same mechanism as the code L2 scope-detector. L0 — which sees only
byte identity — is at chance, exactly like shared delta.

**Takeaway:** shared/surface δ is a whitespace detector (useless where BPE
struggles, e.g. Chinese). Context-aware *mid-layer* δ genuinely discovers
unmarked word structure (AUC ~0.66 — modest, jieba is an imperfect reference,
6.4M-param model). This is the one positive result for the chunking idea, and
it maps directly onto H-Net's design (chunk from hidden states between stages,
not from the input).

## Phase 11: Forget gate vs ALiBi (attention decay, bilingual byte)

Tests whether a *learned* per-token decay (forget gate) beats RoPE, and
whether learning it beats a *fixed* ALiBi schedule. Decay composes with RoPE
as an additive per-head attention bias: query i -> key j penalized by
A_i^h - A_j^h, with A^h = cumsum(log alpha^h). A constant alpha == ALiBi, so
the three modes isolate "does decay help" and "learned vs fixed". Per-head
(ALiBi's heterogeneous receptive fields are the point). All leakage-clean,
matched config (256d/8h/6L byte, bilingual en+zh wiki, lr=1e-3, 50K steps).

| Run | decay | Val Loss | vs RoPE |
|-----|-------|----------|---------|
| RoPE | none | 1.0753 | — |
| **RoPE + ALiBi** | fixed per-head | **1.0687** | **-0.0066** |
| RoPE + forget gate | learned per-head | 1.0726 | -0.0027 |

**Recency bias genuinely helps** (ALiBi -0.0066 nats) — the first non-trivial
positive effect in the project. **But learning the decay does not beat fixing
it**: the forget gate recovers only ~40% of ALiBi's gain.

The why: the learned per-head slopes (mean effective slope -E[log alpha] per
head) **independently rediscovered ALiBi's geometric schedule**, starting from
alpha~1 (no decay) at init:

| head | learned slope | ALiBi 2^(-8h/n) |
|------|---------------|-----------------|
| 0 | 0.430 | 0.500 |
| 1 | 0.210 | 0.250 |
| 2 | 0.080 | 0.125 |
| 3 | 0.043 | 0.063 |
| 4 | 0.019 | 0.031 |
| 5 | 0.014 | 0.016 |
| 6 | 0.012 | 0.008 |
| 7 | 0.004 | 0.004 |

It learned to forget (overall mean alpha=0.91) and converged to ~ALiBi's
heterogeneous per-head profile — but lands slightly worse, a noisier
content-dependent approximation of a hand-picked schedule that's already
near-optimal in-distribution. (Caveat: in-distribution at fixed 512 ctx;
ALiBi's marquee win is length extrapolation, which a content-dependent gate
might still take — deliberately deferred.)

## Phase 12: Per-head learned delta (per-head RoPE frequency scaling)

Each head gets its own learned position rate δ^h (= per-head RoPE frequency
scaling). The one principled gap left in the design matrix, motivated by the
forget-gate result (per-head mattered there). Bilingual byte, leakage-clean,
matched config.

| Variant | Val Loss | vs RoPE |
|---------|----------|---------|
| RoPE (baseline) | 1.0753 | — |
| selective_rope shared | 1.0747 | −0.0006 |
| selective_rope shared **per-head** | 1.0740 | −0.0013 |
| selective_rope per-layer | 1.0722 | −0.0031 |
| selective_rope per-layer **per-head** | 1.0701 | −0.0052 |

**Capability: loss-neutral**, as predicted. Per-head δ rescales RoPE's
*already-present* per-head frequency spectrum, so it adds no representational
power RoPE lacked (unlike decay, which RoPE genuinely lacks — Phase 11). The
faint monotonic ordering (per-head and per-layer each nudge down, stacking
both best) is the same magnitude as noise and below the ALiBi effect.

**Heads do specialize, though.** Per-layer per-head mean δ (layer × head):

```
L0:  1.5  6.3  3.8  2.5  3.2  1.2  4.0  3.5   (std 1.50)
L1:  7.1  6.3  1.2  2.2  1.7  1.5  5.5  1.0   (std 2.37)
L2:  1.4  1.5  1.2  1.0  1.5  1.7  1.1  1.2   (std 0.22, near-uniform)
L3:  1.3  0.8  0.4  0.03 1.2  6.3  2.1  1.4   (std 1.86)
L4:  1.0  6.1  0.01 0.7  1.3  0.8  7.9  1.2   (std 2.71)
L5:  0.05 0.4  0.07 0.8  2.1  0.1  0.8  0.6   (std 0.64)
```

Heads differentiate into a wide range of rates (some layers span 0.01–7.9),
layer-dependent (L2 stays uniform; L1/L4 spread hard).

**Functional, not drift — but it changes the solution, doesn't improve it.**
Correlating each head's δ against its actual attention lookback distance
(48 layer×head pairs): Pearson r=−0.349, Spearman=−0.478. A moderate, sensible
negative correlation — **larger δ → more local attention** (bigger phase
rotation decorrelates q·k faster with distance), the position analog of
ALiBi's local↔global head split. Within-layer r is negative in 5/6 layers
(L2 −0.48, L5 −0.55, L4 −0.40, L3 −0.38, L0 −0.30), with L1 the exception
(r≈+0.05 — that layer's δ spread *is* drift). So the model genuinely wired δ
into per-head receptive fields, but only moderately (|r|≈0.4, content does
most of the work): it re-parametrizes a head specialization it already
achieves via q/k content. The loss landscape pushes the model to *use* δ
(it's not cosmetic), but at the same capability — a different solution, not a
better one.

## Phase 13: Document masking, unbounded per-layer deltas, and ablations

**Question.** Per-layer deltas were capped (sigmoid, max_delta = 10) because
they exploded. Training samples random 512-byte windows over EOS-joined
documents, and attention used to cross EOS. The hypothesis: a huge position
jump scrambles RoPE phase across a boundary, which was the model's only way to
stop attending to the previous, unrelated document. So: block attention across
EOS, remove the cap, and check (a) whether the explosion goes away and (b)
whether the model actually uses its deltas.

**Setup.** The blog-post config (byte-level bilingual Wikipedia, 256d/8h/6L,
50k steps, batch 64 × 512, lr 1e-3). Per-layer DeltaMLPs keep the RMSNorm
input but use the same unbounded softplus as the shared one. Runs (wandb
`byte-pe`, lab Slurm A40s):

| run | wandb | doc mask | final val loss |
|---|---|---|---|
| `docmask_rope_s42` (baseline) | `ycqqcknn` | yes | 1.0752 |
| `docmask_shared_s42` | `4zee13hf` | yes | 1.0716 |
| `docmask_perlayer_s42` | `zp4muno6` | yes | 1.0711 |
| `docmask_perlayer_s1` | `qi0epy7e` | yes | 1.0698 |
| `docmask_perlayer_s2` | `3866xiyd` | yes | 1.0697 |
| `nodoc_mask_perlayer_unbounded_s42` (control) | `rrzpmhro` | no | 1.0787 |

The control was evaluated without masking, so its val loss isn't directly
comparable. The masked runs are again loss-neutral to slightly better than the
RoPE baseline (−0.004 to −0.006, across 3 per-layer seeds vs 1 baseline seed).
The control's checkpoint config was written before `doc_mask` existed; it was
patched to `doc_mask: false` on the cluster before analysis; the released
checkpoint has the same patch.

Every table below comes from `delta_ablation.py`, `byte_boundary_analysis.py`
and `entity_grouping_analysis.py` run on the final checkpoints
(`hf:selective_pe/<run name>/step_50000.pt`).

### Explosion

Largest delta on the logged sample batch (one batch per log step, so noisy):

| step | control (no mask) | masked s42 | masked s1 | masked s2 |
|---|---|---|---|---|
| 1k | 2.9 | 1.9 | 3.9 | 2.0 |
| 3k | 48.7 (L1 **mean** 45) | 3.1 | 8.8 | 3.0 |
| 5k | **107.4** (L1 mean 23) | 3.5 | 11.0 | 3.3 |
| 10k | 14.8 | 7.8 | 10.4 | 7.1 |
| 50k | 25.0 | 8.4 | 15.3 | 7.2 |
| peak | 107.4 | 13.7 | 17.0 | 12.9 |

- With RMSNorm on the DeltaMLP input, even the unmasked control doesn't diverge
  to the 500K+ of the original v1; it spikes early, then settles.
- Masking removes the early spike entirely and keeps every layer's typical
  delta near 1. Two masked seeds still exceed 10, so the old cap would have
  bound some tokens even with masking.
- At the end of training, the control's L3 sets EOS to **700×** that layer's
  typical delta (≈ 25, the largest delta in the model), and the blog-post
  replica (bounded, unmasked) sets EOS to 200× in L5. That is the
  document-cutting use the hypothesis predicted. In the masked runs EOS starts
  a document, so its delta has no effect (it shifts every later position
  equally) and the EOS rows below are meaningless.

### Speed and pattern

Scaling every delta in a layer is the same as changing that layer's RoPE base,
so each layer's deltas split into a **speed** (geometric mean delta over val
data) and a **pattern** (delta / speed). This is done in analysis, not in the
model (`delta_ablation.py`), as agreed with Brendan.

Layer speeds:

| run | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| masked s42 | 1.44 | 1.49 | 0.97 | 0.59 | 0.65 | **0.089** |
| masked s1 | 1.70 | 1.63 | 0.98 | 1.07 | **0.093** | 0.26 |
| masked s2 | 1.46 | 0.85 | 0.43 | 1.07 | 0.94 | **0.053** |
| control | 2.17 | 2.30 | 1.17 | **0.035** | 0.84 | 0.70 |
| blog replica | 1.95 | 1.58 | 1.29 | 0.43 | 1.12 | **0.049** |

Every per-layer model has exactly one deep layer whose speed collapses to
≈ 0.04–0.09: positions there barely change within a 512-byte window, so that
layer runs almost without position information *except* at large-pattern
tokens. In that layer, newline is 11–23× the typical delta (masked seeds), and
the character plots show it packing whole words together with gaps between
them. Which layer does this varies by seed.

### Do the deltas matter? (layer-by-layer ablation)

Loss increase (nats/byte, 819k val tokens spread over the val set) when a
layer's deltas are replaced by 1 (integer RoPE) or by the layer's speed (keeps
speed, removes pattern):

| run | all = 1 | all = speed | L0 = speed | largest other layer = speed | largest other layer = 1 |
|---|---|---|---|---|---|
| masked per-layer s42 | +0.71 | +0.13 | +0.097 | L1 +0.008 | L5 +0.118 |
| masked per-layer s1 | +1.36 | +0.30 | +0.261 | L4 +0.011 | L1 +0.280 |
| masked per-layer s2 | +0.65 | +0.20 | +0.090 | L2 +0.054 | L5 +0.098 |
| masked shared | +0.27 | +0.061 | +0.046 | L1 +0.003 | L1 +0.014 |
| control (no mask) | +2.11 | +0.32 | +0.256 | L1 +0.017 | L1 +0.949 |
| blog replica, per-layer | +1.11 | +0.30 | +0.265 | L5 +0.005 | L1 +0.193 |
| blog replica, shared | +0.16 | +0.024 | +0.009 | L1 +0.007 | L1 +0.029 |

For the shared models the same deltas feed every layer; a per-layer row
replaces them only in that layer's attention.

- **"Loss-neutral" doesn't mean unused.** Each model matches plain RoPE's loss,
  but takes it from the deltas: forcing integer positions everywhere costs
  0.65–1.36 nats in the masked per-layer models.
- **The pattern that matters is L0's**, i.e. the byte-identity pattern
  (+0.09 to +0.26 nats when removed), consistently across seeds.
- **The deeper layers' contextual patterns barely matter**: removing one costs
  ≤ 0.011 nats, apart from s2 L2 (+0.054). These are the layers behind the
  blog post's Chinese-word-boundary and multi-word-entity findings, so that
  structure is real but cheap for the model to lose.
- **Deep-layer speed matters**: setting the near-position-free layer back to
  speed 1 costs 0.10–0.21 nats in every per-layer model.

### Interpretability results across seeds (masked)

Chinese word-boundary ROC-AUC vs jieba (0.5 = chance; values below 0.5 mean
boundaries get *smaller* deltas):

| layer | s42 | s1 | s2 | control | blog (retrained, unmasked) |
|---|---|---|---|---|---|
| L0 | 0.52 | 0.52 | 0.51 | 0.49 | ~0.5 |
| L1 | 0.54 | 0.53 | 0.64 | 0.47 | ~0.5 |
| L2 | 0.57 | 0.54 | 0.57 | 0.55 | 0.63 |
| L3 | 0.53 | 0.61 | 0.64 | 0.63 | 0.41 |
| L4 | 0.47 | **0.24** | 0.63 | 0.52 | 0.72 |
| L5 | 0.46 | 0.39 | 0.56 | 0.46 | 0.41 |

Some contextual layer separates word boundaries in every seed, but which layer,
how strongly, and in which direction varies. With masking, s42 is the weakest
(max |AUC − 0.5| = 0.07).

Multi-word entity test (delta at the internal space, real order vs reversed;
two-sided Wilcoxon p):

| layer | s42 | s1 | s2 | control |
|---|---|---|---|---|
| L1 | 2e-4 | 1e-3 | 0.9 | 0.28 |
| L2 | 8e-4 | 2e-8 | 1e-4 | 0.11 |
| L3 | 8e-7 | 9e-5 | 4e-9 | 2e-6 |
| L4 | 0.08 | 3e-12 | 6e-10 | 2e-8 |
| L5 | 1e-7 | 2e-8 | 6e-9 | 1e-6 |

This is the most robust contextual result: real-order entities get a smaller
internal-space delta in L2/L3/L5 in all three masked seeds.

### Is delta more than the model's own uncertainty?

Word boundaries are where the next character is hardest to predict, and "New"
predicts "York" much better than "York" predicts "New". So both results could
be the model's ordinary next-byte uncertainty showing up in the deltas. The
analysis scripts now compute that uncertainty from the same forward pass.

**Chinese word boundaries.** Single-feature AUC of the model's next-byte
entropy just before each gap, and cross-validated AUC (logistic regression,
5 folds grouped by 512-byte window) on uncertainty (entropy + next-character
surprisal), on all six layers' log delta, and on both:

| model | entropy alone | uncertainty | log delta | both | gain from delta |
|---|---|---|---|---|---|
| masked s42 | 0.862 | 0.876 | 0.621 | 0.882 | +0.006 |
| masked s1 | 0.862 | 0.876 | 0.810 | 0.907 | +0.031 |
| masked s2 | 0.860 | 0.875 | 0.720 | 0.886 | +0.011 |
| control (no mask) | 0.862 | 0.877 | 0.647 | 0.884 | +0.007 |
| blog replica | 0.860 | 0.876 | 0.750 | 0.891 | +0.015 |

The model's own uncertainty separates word boundaries far better (0.86) than
any layer's delta ever did (best single layer 0.72). Deltas add 0.006–0.031
AUC on top, so they carry a little boundary information beyond uncertainty,
but mostly they re-express it.

**Multi-word entities.** At the internal space the model is already much more
confident in the real order: log p(second word) is about −4 for real entities
vs −9 reversed, and next-byte entropy is lower (2.5 vs 2.9–3.0 nats). Repeating
the paired test on the part of log delta a linear fit on those two doesn't
explain (two-sided Wilcoxon p; 6 tests per model, so Bonferroni 0.0083):

| layer | s42 | s1 | s2 | control | blog replica |
|---|---|---|---|---|---|
| L1 | 0.18 | 0.28 | 0.95 | 0.77 | 0.49 |
| L2 | 0.11 | 0.049 | 0.25 | 0.38 | 0.36 |
| L3 | 0.23 | 0.55 | 0.49 | 0.077 | 0.78 |
| L4 | 0.82 | **0.0038** | 0.093 | 0.35 | 0.29 |
| L5 | 0.022 | 0.068 | **0.0075** | 0.019 | **0.0014** |

Most of the entity effect is the model's forward expectation. What survives is
in the slow, word-grouping layer (L5, or L4 in s1, the layer whose speed
collapses to ≈ 0.05–0.09): it passes Bonferroni in 3 of 5 models and is
p ≤ 0.022 in the other two. The fit uses final-output uncertainty and is
linear, so a residual effect could still be uncertainty the fit misses.

### Not done yet

- More than one seed for the RoPE baseline.
- A frequency-matched non-entity bigram control for the entity test.

## Phase 14: SmolLM2-135M-sized model (576d/9h/30L, 122M params)

**Question.** Are the Phase 13 results artifacts of a 6.4M-parameter model?
At a size where the model is reasonably good, do the per-layer patterns
converge across seeds, and do deltas carry more than the model's own
uncertainty?

**Setup.** Before scaling up, the model and optimizer were brought in line
with SmolLM2/OLMo 2 practice (at Brendan's request). The architecture was
already SmolLM2's (Llama: pre-norm RMSNorm, SwiGLU, RoPE, no biases, tied
embeddings). Changed:
- AdamW β2 0.95; weight decay 0.1 on weight matrices only. It used to apply
  to every parameter, including the DeltaMLP bias, pulling every delta toward
  softplus(0).
- QK-norm (OLMo 2).
- SmolLM2-135M shape: 576 wide, 30 layers, 9 heads of 64, FFN 1536.

RoPE θ stays 10,000 (1024-byte context). Training: 1024-byte windows over
300k bilingual Wikipedia documents (1.32 GB), batch 32, 50k steps (≈1.2
passes), lr 6e-4, warmup 1000, cosine, document-masked attention via
`flex_attention`. Runs were on lab-cluster A40s at low priority, about 9.5 h
each. The per-layer runs were stopped by the 8-hour limit and resumed from
saved state at step ~41k.

| run | wandb | final val loss |
|---|---|---|
| `big_docmask_rope_s42` | `ac95615f` | 0.8718 |
| `big_docmask_perlayer_s42` | `0669d9cc` | 0.8734 |
| `big_docmask_perlayer_s1` | `3adb9f11` | 0.8719 |

Loss-neutral again: within 0.0016 of RoPE at every 10k-step eval.

**Large deltas without runaway.** Even with masking, the largest delta on the
logged batch peaked at 115 (s42, step 16k) and 42.5 (s1), then settled at 34
and 16. Training was unaffected.

### Speed and pattern

Both seeds have several **near-position-free layers** (speed ≤ 0.11), all in
the middle to late network, and these are where the extreme patterns live:

| seed | slow layers (speed) | pattern in those layers |
|---|---|---|
| s42 | L10 (0.001), L15 (0.009), L22 (0.001), L26 (0.017) | newline 600–1500×, sentence punct 270–540× (L22, L26) |
| s1 | L13 (0.07), L15 (0.11), L16 (0.003), L17 (0.005) | newline 340–380×, sentence punct 150–180×, space ≈ 23× (L16, L17) |

These layers barely move positions within a sentence and put a huge gap at
sentence and paragraph boundaries. The *role* reproduces across seeds, but the
layer indices don't, as in the small model.

### Ablation

| | s42 | s1 | small model (Phase 13) |
|---|---|---|---|
| all layers = 1 | +1.18 | +0.39 | +0.65 to +1.36 |
| all layers = speed | **+0.041** | **+0.042** | +0.13 to +0.30 |
| largest single layer = speed | L26 +0.004 | L0 +0.008 | L0 +0.09 to +0.26 |
| largest single layer = 1 | L1 +0.55 | L3 +0.021 | +0.10 to +0.28 |

The model still depends heavily on the deltas' speeds (s42's L1 runs at speed
5.8; resetting it costs 0.55 nats). The per-token pattern is now spread out:
removing all of it costs the same 0.04 nats in both seeds, but no single
layer's pattern costs more than 0.008. The small model's dominant L0
byte-identity pattern doesn't reappear.

### Word boundaries vs the model's own uncertainty

| | s42 | s1 | small model |
|---|---|---|---|
| next-byte entropy | 0.869 | 0.868 | 0.860–0.862 |
| uncertainty (CV) | 0.880 | 0.879 | 0.875–0.877 |
| all-layer log delta (CV) | **0.921** | **0.915** | 0.62–0.81 |
| uncertainty + log delta (CV) | **0.930** | **0.930** | 0.88–0.91 |
| gain from delta | **+0.050** | **+0.051** | +0.006 to +0.031 |
| best single layer | L26 0.890 | L29 0.763 (L13 0.355) | ≤ 0.72 (|AUC − 0.5| ≤ 0.26) |

At this size the deltas separate Chinese word boundaries **better than the
model's own uncertainty does**, and add 0.05 AUC on top of it, identically in
both seeds. In s42 a single slow layer (L26) beats next-byte entropy on its
own. Caveat: the classifier has 30 delta features here vs 6 in the small
model (cross-validated by window, 40k gaps, so overfitting is unlikely to
explain 0.05, but more layers also means more chances to find signal).

### Multi-word entities vs uncertainty

The model is more confident than the small one that real entities continue
(log p(second word) ≈ −2.7 real vs −9.5 reversed, against −4 vs −9). With 30
tests per model, Bonferroni is p < 0.0017:
- s42: L26 survives control for uncertainty (p = 2.5e-4), the same slow layer
  that best separates word boundaries.
- s1: nothing passes Bonferroni. Eight layers have p < 0.05, smallest L11
  (4.3e-3).

So the entity signal beyond uncertainty is suggestive at best.

### Takeaways

- **Convergence:** the two seeds agree on everything aggregate: loss, pattern
  cost (0.041 vs 0.042), boundary AUC (0.930 vs 0.930), and having a cluster
  of slow, boundary-heavy mid-to-late layers. They disagree on which layers.
  Per-layer interpretation therefore has to be by role (find the slow layers),
  not by index.
- **Scale helps the interpretability case:** the deltas' word-boundary
  information beyond the model's uncertainty grows from +0.006–0.031 to +0.05
  AUC, and the slow layers are the natural place to look.
- **The small model's "L0 pattern matters" result was a small-model effect.**

## Overall Conclusions

1. **Loss-neutral everywhere (positions).** Across all delta phases, the
   selective *position* mechanism neither helped nor hurt perplexity beyond
   single-seed noise.
2. **Not a capacity or optimization artifact.** Scaling to 152M didn't change
   it; freezing the deltas didn't change it.
3. **Real, interpretable structure.** Shared deltas learn a function/content/
   boundary/document hierarchy; per-layer deltas specialize layers (scope
   delimiters in code, word boundaries in Chinese via mid-layers). Standard
   RoPE apparently reconstructs equivalent positional information from context
   without needing it in the encoding.
4. **PoPE < RoPE on natural-language LM**, consistent with the PoPE paper's
   modest LM gains (its wins are on music / diagnostic tasks).
5. **Shared δ = surface detector; mid-layer δ = context-driven discovery.**
   The chunking idea fails for shared/whitespace-based deltas but is partially
   viable via context-aware mid-layer deltas (Chinese word boundaries at
   AUC 0.66) — i.e. H-Net-style hidden-state chunking, not input-level.
6. **Decay (recency bias) is the one thing that helped — but fixed ALiBi beats
   the learned forget gate.** The learned per-head gate rediscovered ALiBi's
   geometric slope schedule from scratch, yet ended up slightly worse than
   just hard-coding it. Net pattern of the project: these learned mechanisms
   reconstruct sensible structure but don't outperform good fixed/standard
   choices on in-distribution perplexity.
7. **Per-head δ: used but not useful.** Loss-neutral, yet moderately functional
   — larger-δ heads attend more locally (r≈−0.4 vs attention range), the
   position analog of ALiBi's local↔global head split. The model wires δ into
   per-head receptive fields (not drift) but only re-parametrizes a
   specialization it already gets from content: a different solution at the
   same capability, not a better one.

## Open Threads / TODO

- Length-extrapolation eval (train 512, test longer): the regime where ALiBi /
  a content-dependent forget gate would actually be expected to win.
- Reconcile the indirect-indexing discrepancy (single- vs multi-hop task).
- Music (MAESTRO) — where PoPE showed its largest gains; needs a tokenizer.
