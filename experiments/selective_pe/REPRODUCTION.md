# Reproduction check (2026-09-05)

Retrained both bilingual byte models from this repo with the command in the
README's "Reproducing the blog post" section, on the post-leakage-fix data
pipeline, and reran every analysis script. Original = the checkpoints behind
the blog post (wandb runs `rt7cv5ww` shared, `ho7v6eou` per-layer); replica =
wandb runs `q19navjs` (shared, RunPod via `./train.sh remote`) and `0eqetsep`
(per-layer, local RTX 3060 Ti via `./train.sh local`). Replica checkpoints:
wandb artifacts `selective-pe-repro_byte_selective_rope_bilingual_256d_8h_6L_lr1e3:v0`
and `selective-pe-repro_byte_selective_rope_perlayer_bilingual_256d_8h_6L_lr1e3:v0`
(also `s3://brendanlong-experiments/selective_pe/checkpoints/repro_{shared,perlayer}/`).

## Validation loss

All three columns use the same 2,000-document validation set; "leaky" means
the original runs' training set also contained those documents.

| model | original (val docs in train set) | replica (clean train set) | RESULTS.md Phase 12 (clean, 2026-06-01) |
|-------|----------------------------------|---------------------------|------------------------------------------|
| shared | 1.0468 | 1.0748 | 1.0747 |
| per-layer | 1.0436 | 1.0712 | 1.0722 |

The original shared checkpoint evaluates to 1.04675 with this repo's code
(wandb logged 1.04677). The original per-layer checkpoint no longer exists.

## Shared model: per-byte δ (`byte_boundary_analysis.py`, Part 1)

| category | original checkpoint | replica |
|----------|---------------------|---------|
| a-z | 0.68–0.96 (0.79) | 0.71–0.96 (0.82) |
| Chinese continuation | 0.73–0.86 (0.80) | 0.78–0.92 (0.84) |
| Chinese lead | 0.84–0.98 (0.92) | 0.83–0.95 (0.90) |
| space | 1.05 | 1.07 |
| A-Z | 1.01–1.29 (1.10) | 1.00–1.30 (1.12) |
| `. ! ?` | 1.10–1.29 (1.18) | 1.14–1.36 (1.26) |
| `, ; : -` | 1.01–1.25 (1.14) | 1.01–1.25 (1.13) |
| newline | 2.12 | 2.09 |
| EOS | 2.90 | 2.48 |
| between-CJK AUC | 0.498 | 0.497 |

Every row that appears in the post's table matches it exactly (the post's
"Punctuation" row is the `. ! ?` category; the `, ; : -` and AUC rows come
from the script only). The shared-model spacing figures regenerate
pixel-identical to the post's images.

## Per-layer model, layer-0 δ (original numbers from the post)

| category | blog (original) | replica |
|----------|-----------------|---------|
| a-z | 1.21–2.53 (1.64) | 1.08–1.89 (1.42) |
| Chinese continuation | 1.57–2.08 (1.79) | 1.39–2.03 (1.62) |
| Chinese lead | 2.04–2.72 (2.43) | 1.89–2.61 (2.40) |
| A-Z | 2.87–9.98 (9.52) | 1.81–2.58 (2.17) |
| `. ! ?` | 9.80–9.98 (9.90) | 4.33–5.94 (4.96) |
| EOS | 9.82 | 8.67 |
| space | 9.99 | 9.87 |
| newline | 9.99 | 9.89 |

Same broad hierarchy (lowercase < continuation < lead-byte/uppercase <
punctuation < EOS/space/newline), but the replica pushes fewer categories to
the `max_delta=10` ceiling, and one detail does not replicate: the post has
uppercase (9.52) far above Chinese lead bytes (2.43), whereas the replica has
uppercase (2.17) slightly *below* lead bytes (2.40).

## Chinese word-boundary AUC vs jieba (per-layer)

| layer | blog | RESULTS.md Phase 10 | replica |
|-------|------|---------------------|---------|
| L0 | 0.50 | 0.501 | 0.516 |
| L1 | 0.54 | 0.530 | 0.528 |
| L2 | 0.68 | 0.656 | 0.634 |
| L3 | 0.37 | 0.384 | 0.414 |
| L4 | 0.63 | 0.629 | 0.722 |
| L5 | 0.47 | 0.501 | 0.411 |

Same pattern: L0 at chance, L2/L4 detect boundaries, L3 anti-correlated.
The blog and Phase 10 columns disagree slightly (L2, L5); both came from the
now-lost checkpoint, so the analysis settings behind each cannot be rerun. For the original models the "held-out" Chinese text was inside the
training set; for the replica it is genuinely held out.

## Entity reversed-order control (`entity_grouping_analysis.py`)

| layer | blog: real / reversed / % smaller / p | replica |
|-------|----------------------------------------|---------|
| L0 | 9.99 / 9.99 / 0% / 1.0 | 9.87 / 9.87 / 0% / 1.0 |
| L1 | 1.42 / 1.43 / 51% / 0.28 | 1.68 / 1.69 / 48% / 0.55 |
| L2 | 1.43 / 1.54 / 71% / 3e-5 | 1.31 / 1.34 / 61% / 0.04 |
| L3 | 0.06 / 0.10 / 66% / 6e-5 | 0.58 / 0.71 / 72% / 1e-5 |
| L4 | 0.86 / 1.21 / 77% / 3e-8 | 1.04 / 1.16 / 63% / 2e-4 |
| L5 | 0.47 / 0.64 / 78% / 3e-7 | 0.12 / 0.20 / 82% / 6e-10 |

Same conclusion (no effect at L0/L1, real order gets the smaller increment in
L2–L5), with the strength per layer varying between seeds.

## Figures

`plot_increments.py` on the replica reproduces the qualitative features the
post describes: punctuation/word boundaries in L0, tighter concept-like
grouping in a middle layer, and much smaller increments in the deepest layers
on the shared scale. Which deep layer collapses differs (L3 in the original,
L5 in the replica).

## Delta-graft

`analyze_deltas.py` on the artifacts `delta-graft-SmolLM2-1.7B_lora_delta:v0`
and `delta-graft-SmolLM2-1.7B_full_delta:v0` reproduces every number in
`../delta_graft/RESULTS.md` (LoRA: mean 1.033, std 0.011, EOS 1.053; full:
mean 1.002, std 0.003, EOS 1.014; same category medians).
