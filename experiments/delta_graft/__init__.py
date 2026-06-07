"""Graft learned position deltas onto a pretrained model.

Tests whether interpretable per-token learned position rates (deltas) can be
added to an off-the-shelf, open-data pretrained model (SmolLM2, then OLMo 2)
with minimal continued training and without hurting perplexity.

The base models are Llama-architecture: RoPE frequencies are computed once at
the model level from ``position_ids`` and broadcast to every attention layer.
``position_ids`` may be float, so the shared-delta variant needs no surgery to
the base model — we compute delta_t > 0 from the token embeddings, take an
*exclusive* cumulative sum (so token 0 -> position 0, exactly reproducing the
base model at initialization), and pass the result as fractional position_ids.
"""
