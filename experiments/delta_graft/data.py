"""Streaming FineWeb-Edu data pipeline for the delta-graft experiment.

Streams documents, tokenizes with the base model's tokenizer, packs them into
fixed-length windows, and yields ``{"input_ids", "labels"}`` where labels are a
copy of input_ids (the HF model shifts internally for next-token loss).

The first ``val_docs`` documents of the stream are reserved for validation; the
training stream skips them, so train and val never share a document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import PreTrainedTokenizerBase


class PackedStream(IterableDataset[dict[str, torch.Tensor]]):
    """Yields packed token windows from a streaming text dataset.

    Documents are tokenized and concatenated (separated by EOS) into a running
    buffer that is sliced into ``context_len``-token windows.
    """

    def __init__(
        self,
        *,
        dataset: str,
        dataset_config: str | None,
        text_column: str,
        tokenizer: PreTrainedTokenizerBase,
        context_len: int,
        split: str,
        skip_docs: int,
        take_docs: int | None,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.split = split
        self.skip_docs = skip_docs
        self.take_docs = take_docs
        self.seed = seed
        eos = tokenizer.eos_token_id
        assert isinstance(eos, int), "tokenizer must define an integer eos_token_id"
        self.eos_id: int = eos

    def _docs(self) -> Iterator[str]:
        ds = load_dataset(
            self.dataset,
            self.dataset_config,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)
        if self.skip_docs:
            ds = ds.skip(self.skip_docs)
        if self.take_docs is not None:
            ds = ds.take(self.take_docs)
        # Shard across DataLoader workers so each tokenizes a disjoint slice of
        # documents (parallel tokenization, no duplication). With num_workers=0
        # get_worker_info() is None and the full stream is used.
        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers > 1:
            ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
        for row in ds:
            text = row[self.text_column]
            if text:
                yield text

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        window = self.context_len
        buffer: list[int] = []
        for text in self._docs():
            buffer.extend(self.tokenizer(text).input_ids)
            buffer.append(self.eos_id)
            while len(buffer) >= window:
                chunk = buffer[:window]
                buffer = buffer[window:]
                ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": ids, "labels": ids.clone()}


def make_streams(
    *,
    dataset: str,
    dataset_config: str | None,
    text_column: str,
    tokenizer: PreTrainedTokenizerBase,
    context_len: int,
    val_docs: int,
    seed: int,
) -> tuple[PackedStream, PackedStream]:
    """Build (train, val) packed streams with no document overlap."""
    train = PackedStream(
        dataset=dataset,
        dataset_config=dataset_config,
        text_column=text_column,
        tokenizer=tokenizer,
        context_len=context_len,
        split="train",
        skip_docs=val_docs,
        take_docs=None,
        seed=seed,
    )
    val = PackedStream(
        dataset=dataset,
        dataset_config=dataset_config,
        text_column=text_column,
        tokenizer=tokenizer,
        context_len=context_len,
        split="train",
        skip_docs=0,
        take_docs=val_docs,
        seed=seed,
    )
    return train, val
