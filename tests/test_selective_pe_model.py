import pytest
import torch
from torch import Tensor

from experiments.selective_pe.config import BYTE_EOS_ID, selective_pe_model_config
from experiments.selective_pe.model import SelectivePETransformer


def _model(**kwargs: object) -> SelectivePETransformer:
    torch.manual_seed(0)
    config = selective_pe_model_config(
        dim=32, n_heads=4, n_layers=2, vocab_size=257, max_seq_len=64, **kwargs
    )
    model = SelectivePETransformer(config)
    # Random DeltaMLP outputs, so positions are not just ~integer
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return model.eval()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pos_encoding": "rope"},
        {"pos_encoding": "selective_rope"},
        {"pos_encoding": "selective_rope", "per_layer_delta": True},
        {"pos_encoding": "selective_rope", "decay": "learned"},
        {"pos_encoding": "rope", "decay": "alibi"},
    ],
)
def test_documents_do_not_see_each_other(kwargs: dict[str, object]) -> None:
    model = _model(**kwargs)
    doc_b = torch.randint(0, 256, (10,))
    with torch.no_grad():
        logits = [
            model(
                torch.cat(
                    [torch.randint(0, 256, (12,)), torch.tensor([BYTE_EOS_ID]), doc_b]
                )[None]
            )
            for _ in range(2)
        ]
    # EOS and everything after it only depend on the second document
    torch.testing.assert_close(logits[0][0, 12:], logits[1][0, 12:])
    assert not torch.allclose(logits[0][0, :12], logits[1][0, :12])


def test_unit_deltas_match_integer_rope() -> None:
    selective = _model(pos_encoding="selective_rope", per_layer_delta=True)
    rope = _model(pos_encoding="rope")
    rope.load_state_dict(selective.state_dict(), strict=False)
    ids = torch.randint(0, 257, (2, 40))

    def ones(deltas: Tensor, _layer_idx: int) -> Tensor:
        return torch.ones_like(deltas)

    with torch.no_grad():
        torch.testing.assert_close(
            selective.run(ids, delta_override=ones)[0], rope(ids)
        )
        assert not torch.allclose(selective(ids), rope(ids))


def test_deltas_are_causal() -> None:
    model = _model(pos_encoding="selective_rope", per_layer_delta=True)
    ids = torch.randint(0, 256, (1, 40))
    changed = ids.clone()
    changed[0, 30] = (ids[0, 30] + 1) % 256
    deltas, changed_deltas = model.get_deltas(ids), model.get_deltas(changed)
    assert deltas is not None and changed_deltas is not None
    assert deltas.shape == (2, 1, 40)
    assert (deltas > 0).all()
    torch.testing.assert_close(deltas[..., :30], changed_deltas[..., :30])
    assert not torch.allclose(deltas[..., 30:], changed_deltas[..., 30:])
