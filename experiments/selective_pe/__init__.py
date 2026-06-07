"""Byte-level transformer experiment comparing positional encoding variants.

Trains small BPE-tokenized language models on SimpleStories, comparing:
- Standard RoPE (integer positions)
- Standard PoPE (decoupled content/position)
- Selective RoPE (learned per-token delta positions)
- Selective PoPE (learned deltas + decoupled content/position)
"""
