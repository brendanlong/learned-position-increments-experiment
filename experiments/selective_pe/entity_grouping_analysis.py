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

As a partial check, the script also measures the model's own forward
expectation at the space (next-byte entropy, and log-prob of the second word)
and repeats the test on the part of log δ that a linear fit on those two
doesn't explain.

Usage:
    uv run python -m experiments.selective_pe.entity_grouping_analysis \
        --shared <shared_ckpt> --per-layer <per_layer_ckpt>
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import wilcoxon

from experiments.selective_pe.config import BYTE_EOS_ID
from experiments.selective_pe.model import SelectivePETransformer, load_checkpoint

# Two-word, ASCII, capitalized named entities (countries, cities, people,
# institutions, places). Reversed order is used as the within-pair control.
ENTITIES: list[tuple[str, str]] = [
    ("United", "States"),
    ("United", "Kingdom"),
    ("South", "Korea"),
    ("North", "Korea"),
    ("South", "Africa"),
    ("Saudi", "Arabia"),
    ("Sri", "Lanka"),
    ("Costa", "Rica"),
    ("New", "Zealand"),
    ("Sierra", "Leone"),
    ("El", "Salvador"),
    ("Czech", "Republic"),
    ("Burkina", "Faso"),
    ("Dominican", "Republic"),
    ("San", "Marino"),
    ("Great", "Britain"),
    ("Ivory", "Coast"),
    ("Cape", "Verde"),
    ("Hong", "Kong"),
    ("Puerto", "Rico"),
    ("New", "York"),
    ("Los", "Angeles"),
    ("San", "Francisco"),
    ("Las", "Vegas"),
    ("New", "Orleans"),
    ("San", "Diego"),
    ("Santa", "Monica"),
    ("Buenos", "Aires"),
    ("Mexico", "City"),
    ("New", "Delhi"),
    ("Tel", "Aviv"),
    ("Saint", "Petersburg"),
    ("Cape", "Town"),
    ("Kansas", "City"),
    ("Albert", "Einstein"),
    ("Isaac", "Newton"),
    ("Marie", "Curie"),
    ("Charles", "Darwin"),
    ("William", "Shakespeare"),
    ("Abraham", "Lincoln"),
    ("George", "Washington"),
    ("Winston", "Churchill"),
    ("Nelson", "Mandela"),
    ("Steve", "Jobs"),
    ("Bill", "Gates"),
    ("Mark", "Twain"),
    ("Julius", "Caesar"),
    ("Thomas", "Edison"),
    ("Martin", "Luther"),
    ("World", "War"),
    ("Cold", "War"),
    ("Wall", "Street"),
    ("Silicon", "Valley"),
    ("White", "House"),
    ("Supreme", "Court"),
    ("United", "Nations"),
    ("World", "Bank"),
    ("Red", "Cross"),
    ("Roman", "Empire"),
    ("Middle", "Ages"),
    ("Big", "Bang"),
    ("Black", "Sea"),
    ("Red", "Sea"),
    ("Mount", "Everest"),
    ("Niagara", "Falls"),
    ("Pacific", "Ocean"),
    ("Atlantic", "Ocean"),
    ("Indian", "Ocean"),
    ("Great", "Wall"),
    ("Solar", "System"),
    ("Milky", "Way"),
    ("North", "America"),
    ("South", "America"),
    ("Star", "Wars"),
    ("Harry", "Potter"),
    ("New", "Testament"),
    ("Old", "Testament"),
    ("Holy", "Land"),
    ("Far", "East"),
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


@torch.no_grad()
def internal_space(
    model: SelectivePETransformer, w1: str, w2: str
) -> tuple[np.ndarray, float, float]:
    """Per-layer delta at the w1/w2 space, averaged over CARRIERS, plus the
    model's own forward expectation there: the entropy of its next-byte
    prediction at the space and its log-prob of w2 given everything before.

    Deltas have shape (n_rows,): n_rows is 1 (shared) or n_layers (per-layer).
    """
    device = next(model.parameters()).device
    per_carrier: list[np.ndarray] = []
    entropies: list[float] = []
    w2_logprobs: list[float] = []
    for carrier in CARRIERS:
        text = carrier.format(a=w1, b=w2)
        ids = torch.tensor([BYTE_EOS_ID, *text.encode()], device=device)
        logits, _, deltas = model.run(ids[None])
        d = torch.stack(deltas)[:, 0] if model.config.per_layer_delta else deltas[0]
        rows = d.cpu().numpy()[:, 1:]  # drop the prepended-EOS column
        # Byte index of the internal space. rows is aligned to text *bytes*, so
        # compute the offset from byte lengths (robust to non-ASCII and to the
        # entity words appearing in the carrier prefix).
        prefix = carrier.split("{a}")[0].format(a=w1, b=w2)
        sp = len(prefix.encode()) + len(w1.encode())
        assert text.encode()[sp] == ord(" ")
        per_carrier.append(rows[:, sp])
        # Logits at index t (of ids, which has the EOS) predict ids[t + 1]
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        at_space = sp + 1
        entropies.append(float(-(logp[at_space].exp() * logp[at_space]).sum()))
        n2 = len(w2.encode())
        targets = ids[at_space + 1 : at_space + 1 + n2]
        w2_lp = logp[at_space : at_space + n2].gather(1, targets[:, None]).sum()
        w2_logprobs.append(float(w2_lp))
    return (
        np.mean(per_carrier, axis=0),
        float(np.mean(entropies)),
        float(np.mean(w2_logprobs)),
    )


def paired_p(correct: np.ndarray, reversed_: np.ndarray) -> float:
    """Two-sided Wilcoxon p for reversed - correct (1.0 if all differences 0)."""
    diff = reversed_ - correct
    # Byte-identity controls (shared model, per-layer L0) give identical
    # increments for both orders; guard the all-zero-difference case.
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(diff).pvalue)  # type: ignore[attr-defined]


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of a least-squares fit of y on x (with intercept)."""
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def analyze(model: SelectivePETransformer, name: str) -> None:
    n = len(ENTITIES)
    results = [internal_space(model, a, b) for a, b in ENTITIES] + [
        internal_space(model, b, a) for a, b in ENTITIES
    ]
    deltas = np.array([r[0] for r in results])  # (2n, n_rows)
    uncertainty = np.array([[r[1], r[2]] for r in results])  # (2n, 2)
    n_rows = deltas.shape[1]
    # Two-sided Wilcoxon (the direction is the thing being claimed, so report it
    # honestly rather than pre-committing to it). Family of n_rows tests.
    print(
        f"\n{name} ({n} entities x {len(CARRIERS)} carriers; "
        f"two-sided Wilcoxon, {n_rows} tests)"
    )
    for col, label in [(0, "next-byte entropy"), (1, "log p(second word)")]:
        real, rev = uncertainty[:n, col], uncertainty[n:, col]
        print(
            f"At the space, {label}: real {real.mean():.2f}, reversed "
            f"{rev.mean():.2f} (p={paired_p(real, rev):.1e})"
        )
    print(
        "'p beyond uncertainty' repeats the test on the part of delta not "
        "explained by a linear fit on those two."
    )
    header = f"{'layer':<8}{'correct':>9}{'reversed':>10}{'corr<rev':>10}"
    print(header + f"{'p (2-sided)':>14}{'p beyond uncertainty':>22}")
    for r in range(n_rows):
        c, v = deltas[:n, r], deltas[n:, r]
        resid = residualize(np.log(deltas[:, r]), uncertainty)
        p_beyond = paired_p(resid[:n], resid[n:]) if not np.allclose(c, v) else 1.0
        layer = "shared" if not model.config.per_layer_delta else f"L{r}"
        print(
            f"{layer:<8}{c.mean():>9.2f}{v.mean():>10.2f}{(c < v).mean():>10.2f}"
            f"{paired_p(c, v):>14.1e}{p_beyond:>22.1e}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shared", help="shared-delta checkpoint (negative control)")
    p.add_argument("--per-layer", help="per-layer-delta checkpoint")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.shared:
        model = load_checkpoint(args.shared).to(device)
        analyze(model, "SHARED (byte identity -> should not distinguish)")
    if args.per_layer:
        model = load_checkpoint(args.per_layer).to(device)
        analyze(model, "PER-LAYER")


if __name__ == "__main__":
    main()
