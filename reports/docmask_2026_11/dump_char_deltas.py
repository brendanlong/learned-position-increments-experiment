"""Dump raw per-character learned increments for the report's spacing figures.

Usage: python -m reports...dump_char_deltas OUT.json name=ckpt [name=ckpt ...]
Each character's increment is the sum of its bytes' deltas, per layer (one row
for shared-delta models), with an EOS prepended as in plot_increments.py.
"""

import json
import sys

import torch

from experiments.selective_pe.config import BYTE_EOS_ID
from experiments.selective_pe.model import load_checkpoint

TEXTS = {
    "en": "Marie Curie, a Polish-born physicist, won two Nobel Prizes.",
    "en2": "Marie Curie won two Nobel Prizes. She was the first woman to win one.",
    "zh": "人工智能是计算机科学的一个分支。",
}


@torch.no_grad()
def main() -> None:
    out = {"texts": TEXTS, "models": {}}
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        model = load_checkpoint(path)
        rows = {}
        for key, text in TEXTS.items():
            ids = torch.tensor([BYTE_EOS_ID, *text.encode()])[None]
            deltas = torch.stack(model.run(ids)[2])[:, 0, 1:]  # (L, bytes)
            if not model.config.per_layer_delta:
                deltas = deltas[:1]
            per_char, off = [], 0
            for ch in text:
                n = len(ch.encode())
                per_char.append(deltas[:, off : off + n].sum(1))
                off += n
            rows[key] = torch.stack(per_char, 1).tolist()  # (L, chars)
        out["models"][name] = rows
        print(name, "done", flush=True)
    with open(sys.argv[1], "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
