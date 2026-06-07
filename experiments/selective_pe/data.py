"""Data pipeline for next-token prediction on text corpora.

Supports multiple datasets (SimpleStories, CodeParrot) and two tokenization
modes (GPT-2 BPE, raw UTF-8 bytes). Concatenates documents with EOS
separators and produces fixed-length chunks for autoregressive training.

Tokenized corpora are cached to data/selective_pe/corpus_cache/.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from experiments.selective_pe.config import BYTE_EOS_ID, GPT2_EOS_ID

CACHE_DIR = Path("data/selective_pe/corpus_cache")

# Source: (hf_path, text_column, split_name, hf_config_name). hf_config_name
# is the dataset subset/config (or None). A split may map to a single source
# or a list of sources (interleaved round-robin, e.g. bilingual corpora).
SourceSpec = tuple[str, str, str, str | None]
DATASET_CONFIGS: dict[str, dict[str, SourceSpec | list[SourceSpec]]] = {
    "simplestories": {
        "train": ("SimpleStories/SimpleStories", "story", "train", None),
        "test": ("SimpleStories/SimpleStories", "story", "test", None),
    },
    "codeparrot": {
        "train": ("codeparrot/codeparrot-clean-train", "content", "train", None),
        "test": ("codeparrot/codeparrot-clean-valid", "content", "train", None),
    },
    # Chinese Wikipedia: no whitespace between words -> a clean test of whether
    # learned deltas discover real word boundaries vs just detecting spaces.
    # Only a "train" split exists (train==test source); train.py reserves the
    # first val_max_docs for val and skips them in train (no leakage).
    "chinese": {
        "train": ("wikimedia/wikipedia", "text", "train", "20231101.zh"),
        "test": ("wikimedia/wikipedia", "text", "train", "20231101.zh"),
    },
    "english": {
        "train": ("wikimedia/wikipedia", "text", "train", "20231101.en"),
        "test": ("wikimedia/wikipedia", "text", "train", "20231101.en"),
    },
    # Bilingual: interleave English + Chinese wiki so one model sees both,
    # letting us compare learned deltas on en (whitespace-marked) vs zh
    # (no word boundaries) within a single model.
    "bilingual": {
        "train": [
            ("wikimedia/wikipedia", "text", "train", "20231101.en"),
            ("wikimedia/wikipedia", "text", "train", "20231101.zh"),
        ],
        "test": [
            ("wikimedia/wikipedia", "text", "train", "20231101.en"),
            ("wikimedia/wikipedia", "text", "train", "20231101.zh"),
        ],
    },
}


def get_tokenizer() -> PreTrainedTokenizerBase:
    """Load GPT-2 BPE tokenizer."""
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer


def _cache_key(
    split: str,
    dataset_name: str,
    max_docs: int | None,
    skip_docs: int = 0,
) -> str:
    """Compute a cache key for a tokenized corpus."""
    key = f"{dataset_name}:{split}:{max_docs}:skip{skip_docs}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def train_test_share_source(dataset: str) -> bool:
    """True if a dataset's train and test splits stream the same source.

    For such datasets (e.g. Wikipedia, which has no separate test split), the
    train loader must skip the val region to avoid train/test leakage.
    """
    cfg = DATASET_CONFIGS[dataset]
    return cfg["train"] == cfg["test"]


def _load_texts(
    dataset: str,
    split: Literal["train", "test"],
    max_docs: int | None = None,
    skip_docs: int = 0,
) -> list[str]:
    """Load text documents from a HuggingFace dataset.

    Args:
        dataset: Dataset key (e.g. "simplestories", "codeparrot")
        split: "train" or "test"
        max_docs: Cap number of documents (for debugging)
        skip_docs: Skip the first N interleaved docs (reserve a held-out region
            for the val split when train/test share a source)

    Returns:
        List of text strings.
    """
    from datasets import load_dataset

    entry = DATASET_CONFIGS[dataset][split]
    sources = entry if isinstance(entry, list) else [entry]

    # Build a (streaming iterator, text_column) for each source
    iters: list[tuple[object, str]] = []
    for hf_path, text_col, hf_split, hf_config in sources:
        if hf_config is not None:
            ds = load_dataset(hf_path, hf_config, split=hf_split, streaming=True)
        else:
            ds = load_dataset(hf_path, split=hf_split, streaming=True)
        iters.append((iter(ds), text_col))

    # Round-robin interleave documents across sources, skipping the first
    # skip_docs (the reserved val region) before collecting.
    texts: list[str] = []
    active = list(range(len(iters)))
    seen = 0
    while active and (max_docs is None or len(texts) < max_docs):
        for i in list(active):
            ds_iter, text_col = iters[i]
            try:
                example = next(ds_iter)  # type: ignore[arg-type]
            except StopIteration:
                active.remove(i)
                continue
            seen += 1
            if seen <= skip_docs:
                continue
            texts.append(example[text_col])  # type: ignore[index]
            if len(texts) % 100_000 == 0:
                print(f"    loaded {len(texts):,} documents...")
            if max_docs is not None and len(texts) >= max_docs:
                break

    return texts


def prepare_corpus(
    split: Literal["train", "test"],
    dataset: str = "simplestories",
    max_docs: int | None = None,
    skip_docs: int = 0,
) -> Tensor:
    """Load dataset, tokenize with GPT-2 BPE, concatenate with EOS.

    Returns a 1D int32 tensor of token IDs. Documents are separated by
    the GPT-2 EOS token (50256).

    Results are cached to disk so subsequent runs skip tokenization.
    """
    cache_name = _cache_key(split, f"bpe:{dataset}", max_docs, skip_docs)
    cache_path = CACHE_DIR / f"{split}_{cache_name}.pt"
    if cache_path.exists():
        corpus = torch.load(cache_path, weights_only=True)
        assert isinstance(corpus, Tensor)
        print(f"  Loaded cached {split} corpus: {len(corpus):,} tokens")
        return corpus

    texts = _load_texts(dataset, split, max_docs, skip_docs)
    tokenizer = get_tokenizer()

    # Per-doc int32 arrays concatenated once (avoids list[int] memory blowup).
    eos = np.array([GPT2_EOS_ID], dtype=np.int32)
    parts: list[np.ndarray] = []
    n_docs = len(texts)
    for i in range(n_docs):
        tokens = tokenizer.encode(texts[i])  # type: ignore[union-attr]
        parts.append(np.asarray(tokens, dtype=np.int32))
        parts.append(eos)
        texts[i] = ""  # free the string as we go
        if (i + 1) % 100_000 == 0:
            print(f"    tokenized {i + 1:,} documents...")

    corpus_np = np.concatenate(parts) if parts else np.array([], dtype=np.int32)
    corpus = torch.from_numpy(corpus_np)
    print(f"  Prepared {split} corpus: {n_docs:,} docs, {len(corpus):,} tokens")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, cache_path)
    print(f"  Cached corpus to {cache_path}")

    return corpus


def prepare_byte_corpus(
    split: Literal["train", "test"],
    dataset: str = "simplestories",
    max_docs: int | None = None,
    skip_docs: int = 0,
) -> Tensor:
    """Load dataset, encode as raw UTF-8 bytes, concatenate with EOS.

    Returns a 1D int32 tensor. Byte values 0-255 are token IDs 0-255,
    EOS (document separator) is token ID 256.

    Results are cached to disk.
    """
    cache_name = _cache_key(split, f"byte:{dataset}", max_docs, skip_docs)
    cache_path = CACHE_DIR / f"{split}_{cache_name}.pt"
    if cache_path.exists():
        corpus = torch.load(cache_path, weights_only=True)
        assert isinstance(corpus, Tensor)
        print(f"  Loaded cached byte {split} corpus: {len(corpus):,} bytes")
        return corpus

    texts = _load_texts(dataset, split, max_docs, skip_docs)

    # Build per-doc uint16 arrays (EOS=256 needs >8 bits), concatenate once.
    # A Python list[int] would use ~28 bytes/token and OOM on large corpora.
    eos = np.array([BYTE_EOS_ID], dtype=np.uint16)
    parts: list[np.ndarray] = []
    n_docs = len(texts)
    for i in range(n_docs):
        b = np.frombuffer(texts[i].encode("utf-8"), dtype=np.uint8).astype(np.uint16)
        parts.append(b)
        parts.append(eos)
        texts[i] = ""  # free the string as we go
        if (i + 1) % 100_000 == 0:
            print(f"    encoded {i + 1:,} documents...")

    corpus_np = np.concatenate(parts) if parts else np.array([], dtype=np.uint16)
    corpus = torch.from_numpy(corpus_np.astype(np.int32))
    print(f"  Prepared byte {split} corpus: {n_docs:,} docs, {len(corpus):,} bytes")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(corpus, cache_path)
    print(f"  Cached corpus to {cache_path}")

    return corpus


class FixedChunkDataset(Dataset[dict[str, Tensor]]):
    """Non-overlapping fixed-length chunks from a flat token corpus.

    Input is tokens[start:start+context_len], target is
    tokens[start+1:start+context_len+1] (next-token prediction).
    """

    def __init__(self, corpus: Tensor, context_len: int) -> None:
        self.corpus = corpus
        self.context_len = context_len
        self.n_chunks = (len(corpus) - 1) // context_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        start = idx * self.context_len
        chunk = self.corpus[start : start + self.context_len + 1].long()
        return {"input_ids": chunk[:-1], "targets": chunk[1:]}


class StreamingChunkDataset(IterableDataset[dict[str, Tensor]]):
    """Streaming dataset that randomly samples chunks from a token corpus.

    Each chunk starts at a random position, so every training step sees
    unique byte alignments. This follows the CLAUDE.md guideline of
    streaming unique data rather than repeating epochs.
    """

    def __init__(
        self,
        corpus: Tensor,
        context_len: int,
        n_examples: int,
        seed: int = 42,
    ) -> None:
        self.corpus = corpus
        self.context_len = context_len
        self.n_examples = n_examples
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self.n_examples

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        # Mix in the worker id so multiple DataLoader workers don't all yield
        # the same random chunks (the classic IterableDataset duplication trap).
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + self._epoch * 1000 + worker_id)
        self._epoch += 1
        max_start = len(self.corpus) - self.context_len - 1
        for _ in range(self.n_examples):
            start = rng.randint(0, max_start)
            chunk = self.corpus[start : start + self.context_len + 1].long()
            yield {"input_ids": chunk[:-1], "targets": chunk[1:]}


def collate_chunks(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Collate function for chunk datasets."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
    }
