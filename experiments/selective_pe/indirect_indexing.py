"""Indirect indexing diagnostic task for testing positional encoding quality.

Inspired by the PoPE paper (Gopalakrishnan et al., arXiv 2509.10534),
which showed 95% accuracy for PoPE vs 11% for RoPE on this task.

The task: given a sequence of key-value pairs followed by a query key,
predict the value associated with that key. This requires:
1. Matching the query key to a position (content-based search)
2. Reading the value at that position (positional retrieval)

This creates a content-conditional positional query that standard RoPE
handles poorly due to content-position entanglement.

Sequence format (using a small integer vocabulary):
    [k1, v1, k2, v2, ..., kN, vN, SEP, query_key, PREDICT]
    Target: the value associated with query_key

Vocabulary: integers 0..num_values-1 for keys/values, plus SEP and PREDICT
special tokens.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from collections.abc import Iterator


class IndirectIndexingConfig:
    """Configuration for indirect indexing task."""

    def __init__(
        self,
        num_pairs: int = 16,
        num_values: int = 32,
    ) -> None:
        self.num_pairs = num_pairs
        self.num_values = num_values
        # Token IDs: 0..num_values-1 = values, num_values = SEP, num_values+1 = PREDICT
        self.sep_id = num_values
        self.predict_id = num_values + 1
        self.vocab_size = num_values + 2
        # Seq: num_pairs*2 (kv) + SEP + query + PREDICT
        self.seq_len = num_pairs * 2 + 3

    def generate_example(
        self,
        rng: random.Random,
    ) -> tuple[list[int], int]:
        """Generate one indirect indexing example.

        Returns:
            tokens: list of token IDs (length = seq_len)
            answer: the correct value token ID
        """
        # Generate unique keys (no duplicates)
        keys = rng.sample(range(self.num_values), self.num_pairs)
        # Generate random values
        values = [rng.randint(0, self.num_values - 1) for _ in range(self.num_pairs)]

        # Build sequence: k1 v1 k2 v2 ... SEP query_key PREDICT
        tokens: list[int] = []
        for k, v in zip(keys, values, strict=True):
            tokens.append(k)
            tokens.append(v)
        tokens.append(self.sep_id)

        # Pick a random key to query
        query_idx = rng.randint(0, self.num_pairs - 1)
        tokens.append(keys[query_idx])
        tokens.append(self.predict_id)

        return tokens, values[query_idx]


class IndirectIndexingDataset(IterableDataset[dict[str, Tensor]]):
    """Streaming dataset for indirect indexing task."""

    def __init__(
        self,
        config: IndirectIndexingConfig,
        n_examples: int,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.n_examples = n_examples
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self.n_examples

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        for _ in range(self.n_examples):
            tokens, answer = self.config.generate_example(rng)
            # Input: all tokens except last (PREDICT is the position we predict AT)
            # Target: shift by 1 as usual, but we only care about the last position
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            # Target: the answer at the PREDICT position (last input position)
            target = torch.tensor(tokens[1:], dtype=torch.long)
            # Overwrite the target at the PREDICT position with the correct answer
            target[-1] = answer
            # The query_key token is the last input token (index len-2 in the
            # full sequence; PREDICT at len-1 was dropped from input). The model
            # predicts the answer at this position (next-token = the value).
            answer_position = len(input_ids) - 1
            yield {
                "input_ids": input_ids,
                "targets": target,
                "answer": torch.tensor(answer, dtype=torch.long),
                "answer_position": torch.tensor(
                    answer_position,
                    dtype=torch.long,
                ),
            }


def collate_indirect_indexing(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Collate function for indirect indexing dataset."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
        "answer": torch.stack([b["answer"] for b in batch]),
        "answer_position": torch.stack([b["answer_position"] for b in batch]),
    }


@torch.no_grad()
def evaluate_indirect_indexing(
    model: torch.nn.Module,
    dataset: IndirectIndexingDataset,
    device: torch.device,
    n_eval: int = 1000,
    batch_size: int = 64,
) -> dict[str, float]:
    """Evaluate accuracy on indirect indexing task.

    Returns dict with accuracy and loss at the answer position.
    """
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_indirect_indexing,
    )

    correct = 0
    total = 0
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        answer = batch["answer"].to(device)
        answer_pos = batch["answer_position"].to(device)

        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(input_ids)  # (B, S, V)

        # Get logits at the answer position for each example
        batch_idx = torch.arange(logits.shape[0], device=device)
        answer_logits = logits[batch_idx, answer_pos]  # (B, V)

        # Accuracy
        preds = answer_logits.argmax(dim=-1)
        correct += (preds == answer).sum().item()
        total += answer.shape[0]

        # Loss at answer position
        import torch.nn.functional as F

        loss = F.cross_entropy(answer_logits.float(), answer, reduction="sum")
        total_loss += loss.item()

        if total >= n_eval:
            break

    model.train()
    accuracy = correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return {"indirect_indexing/accuracy": accuracy, "indirect_indexing/loss": avg_loss}
