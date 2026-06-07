"""Do learned position increments group multi-word entities? (bilingual byte model)

A multi-word named entity like "New York" is two whitespace-separated words, so
the byte-level model sees a space between them and would normally treat that
space as a word boundary (large increment). If the model has learned that the
span is a single unit, the *internal* space should get a smaller increment than
a normal word boundary.

To control for confounds (capital-to-capital spaces, byte identity of the space,
sentence position) we use a **reversed-order control**: for each entity "A B" we
compare the increment at the internal space in a carrier like "I visited A B"
against the reversed "I visited B A". Both contain the exact same two capitalized
words separated by a space in the same carrier; the only difference is whether
the order forms a real entity. A lower increment for the correct order means the
model is grouping the entity.

The increment at the internal space is causal, so only the text *up to* that
space matters -- the second word and anything after it do not affect it. We
therefore use short carriers (just a prefix + "A B") and average the increment
over a set of them (CARRIERS), so the result isn't an artifact of one phrasing.
(An unavoidable wrinkle: no single natural carrier fits every entity type -- "I
visited World War" is as odd as "the New York" -- which is another reason to
average over several rather than trust one sentence.)

The shared model (increment = a function of byte identity alone) is a negative
control: a space is a space, so it cannot distinguish the two orders.

**What a positive result shows -- and what it does NOT.** The increment at the
internal space is computed *causally*, before the second word's bytes are read,
so it measures whether the first word leads the model to *expect a continuation*
-- not whether the two words are bound into a single unit. "New" is a
high-continuation prefix (New York/Delhi/Orleans...); "York" is not. The
reversed-order control holds capitalization, the space's byte identity, and
sentence position constant, but it does NOT hold *bigram frequency* constant
("New York" is far more frequent than "York New"). So a significant effect shows
the (context-aware) mid-layer increment encodes learned forward-continuation
expectations that track entity bigrams -- a weaker, more honest claim than "the
model groups entities." Cleanly separating the two would need a
frequency-matched non-entity bigram control, which we do not do here.

Usage:
    uv run python -m experiments.selective_pe.entity_grouping_analysis \
        --shared <shared_ckpt> --per-layer <per_layer_ckpt>
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import wilcoxon

from experiments.selective_pe.config import BYTE_EOS_ID, SelectivePEModelConfig
from experiments.selective_pe.model import SelectivePETransformer

# Two-word, ASCII, capitalized named entities (countries, cities, people,
# institutions, places). Reversed order is used as the within-pair control.
ENTITIES: list[tuple[str, str]] = [
    ("United", "States"), ("United", "Kingdom"), ("South", "Korea"),
    ("North", "Korea"), ("South", "Africa"), ("Saudi", "Arabia"),
    ("Sri", "Lanka"), ("Costa", "Rica"), ("New", "Zealand"), ("Sierra", "Leone"),
    ("El", "Salvador"), ("Czech", "Republic"), ("Burkina", "Faso"),
    ("Dominican", "Republic"), ("San", "Marino"), ("Great", "Britain"),
    ("Ivory", "Coast"), ("Cape", "Verde"), ("Hong", "Kong"), ("Puerto", "Rico"),
    ("New", "York"), ("Los", "Angeles"), ("San", "Francisco"), ("Las", "Vegas"),
    ("New", "Orleans"), ("San", "Diego"), ("Santa", "Monica"), ("Buenos", "Aires"),
    ("Mexico", "City"), ("New", "Delhi"), ("Tel", "Aviv"), ("Saint", "Petersburg"),
    ("Cape", "Town"), ("Kansas", "City"), ("Albert", "Einstein"),
    ("Isaac", "Newton"), ("Marie", "Curie"), ("Charles", "Darwin"),
    ("William", "Shakespeare"), ("Abraham", "Lincoln"), ("George", "Washington"),
    ("Winston", "Churchill"), ("Nelson", "Mandela"), ("Steve", "Jobs"),
    ("Bill", "Gates"), ("Mark", "Twain"), ("Julius", "Caesar"),
    ("Thomas", "Edison"), ("Martin", "Luther"), ("World", "War"), ("Cold", "War"),
    ("Wall", "Street"), ("Silicon", "Valley"), ("White", "House"),
    ("Supreme", "Court"), ("United", "Nations"), ("World", "Bank"),
    ("Red", "Cross"), ("Roman", "Empire"), ("Middle", "Ages"), ("Big", "Bang"),
    ("Black", "Sea"), ("Red", "Sea"), ("Mount", "Everest"), ("Niagara", "Falls"),
    ("Pacific", "Ocean"), ("Atlantic", "Ocean"), ("Indian", "Ocean"),
    ("Great", "Wall"), ("Solar", "System"), ("Milky", "Way"), ("North", "America"),
    ("South", "America"), ("Star", "Wars"), ("Harry", "Potter"),
    ("New", "Testament"), ("Old", "Testament"), ("Holy", "Land"), ("Far", "East"),
]

# Short carriers (prefix + "{a} {b}"); only the prefix and the first word affect
# the causal increment at the internal space, so the suffix is dropped. We
# average over the set so the effect isn't tied to one phrasing.
CARRIERS: list[str] = [
    "{a} {b}",
    "the {a} {b}",
    "I visited {a} {b}",
    "near {a} {b}",
    "they went to {a} {b}",
]


def load_model(path: str) -> tuple[SelectivePETransformer, SelectivePEModelConfig]:
    ckpt = torch.load(path, weights_only=True)
    cfg = SelectivePEModelConfig(**ckpt["model_config"])
    model = SelectivePETransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def internal_space_delta(
    model: SelectivePETransformer,
    cfg: SelectivePEModelConfig,
    w1: str,
    w2: str,
) -> np.ndarray:
    """Per-layer increment at the w1/w2 space, averaged over CARRIERS.

    Returns shape (n_rows,): n_rows is 1 (shared) or n_layers (per-layer).
    """
    per_carrier: list[np.ndarray] = []
    for carrier in CARRIERS:
        text = carrier.format(a=w1, b=w2)
        ids = torch.tensor([BYTE_EOS_ID, *text.encode()]).unsqueeze(0)
        deltas = model.get_deltas(ids)
        assert deltas is not None
        rows = (deltas[:, 0, :] if cfg.per_layer_delta else deltas).cpu().numpy()
        rows = rows[:, 1:]  # drop the prepended-EOS column -> aligned to text
        # Byte index of the internal space. rows is aligned to text *bytes*, so
        # compute the offset from byte lengths (robust to non-ASCII and to the
        # entity words appearing in the carrier prefix).
        prefix = carrier.split("{a}")[0].format(a=w1, b=w2)
        sp = len(prefix.encode()) + len(w1.encode())
        assert text.encode()[sp] == ord(" ")
        per_carrier.append(rows[:, sp])
    return np.mean(per_carrier, axis=0)


def analyze(
    model: SelectivePETransformer, cfg: SelectivePEModelConfig, name: str
) -> None:
    correct = np.array(
        [internal_space_delta(model, cfg, a, b) for a, b in ENTITIES]
    )  # (n_entities, n_rows)
    reversed_ = np.array(
        [internal_space_delta(model, cfg, b, a) for a, b in ENTITIES]
    )
    n_rows = correct.shape[1]
    # Two-sided Wilcoxon (the direction is the thing being claimed, so report it
    # honestly rather than pre-committing to it). Family of n_rows tests; the
    # surviving effects are well below any Bonferroni threshold (alpha/n_rows).
    print(
        f"\n{name} ({len(ENTITIES)} entities x {len(CARRIERS)} carriers; "
        f"two-sided Wilcoxon, {n_rows} tests)"
    )
    header = f"{'layer':<8}{'correct':>9}{'reversed':>10}{'corr<rev':>10}"
    print(header + f"{'p (2-sided)':>14}")
    for r in range(n_rows):
        c, v = correct[:, r], reversed_[:, r]
        frac = float((c < v).mean())
        diff = v - c
        if np.allclose(diff, 0):  # noqa: SIM108
            # Byte-identity controls (shared model, per-layer L0) give identical
            # increments for both orders; guard the all-zero-difference case.
            p = 1.0
        else:
            p = float(wilcoxon(diff).pvalue)  # type: ignore[attr-defined]
        layer = "shared" if not cfg.per_layer_delta else f"L{r}"
        print(f"{layer:<8}{c.mean():>9.2f}{v.mean():>10.2f}{frac:>10.2f}{p:>14.1e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared", help="shared-delta checkpoint (negative control)")
    p.add_argument("--per-layer", help="per-layer-delta checkpoint")
    args = p.parse_args()
    if args.shared:
        model, cfg = load_model(args.shared)
        analyze(model, cfg, "SHARED (byte identity -> should not distinguish)")
    if args.per_layer:
        model, cfg = load_model(args.per_layer)
        analyze(model, cfg, "PER-LAYER")


if __name__ == "__main__":
    main()
