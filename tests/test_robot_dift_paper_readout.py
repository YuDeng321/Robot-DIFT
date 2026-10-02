"""CPU checks for the standalone candidate Robot-DIFT manuscript readout."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.encoders.robot_dift_paper_readout import (  # noqa: E402
    RobotDIFTPaperReadout,
    _rotate_2d_keys,
)


@pytest.fixture
def readout():
    torch.manual_seed(11)
    return RobotDIFTPaperReadout(
        {"us3": 4, "us6": 6, "us8": 8},
        fpn_dim=16,
        model_dim=16,
        output_dim=12,
        num_heads=4,
        mlp_hidden_dims=(32,),
        dropout=0.0,
    ).cpu()


def _views(batch=2, count=2, requires_grad=False):
    return [
        {
            "us3": torch.randn(batch, 4, 2, 3, requires_grad=requires_grad),
            "us6": torch.randn(batch, 6, 4, 6, requires_grad=requires_grad),
            "us8": torch.randn(batch, 8, 4, 6, requires_grad=requires_grad),
        }
        for _ in range(count)
    ]


def test_multiview_shape_and_trainable_readout_gradients(readout):
    views = _views(requires_grad=True)
    clip_tokens = torch.randn(2, 77, 512, requires_grad=True)
    mask = torch.zeros(2, 77, dtype=torch.bool)
    mask[:, :9] = True

    per_view = readout.encode_views(views, clip_tokens, mask)
    embedding = readout(views, clip_tokens, mask)

    assert per_view.shape == (2, 2, 77, 16)
    assert embedding.shape == (2, 12)
    assert torch.isfinite(embedding).all()
    torch.testing.assert_close(per_view[:, :, 9:], torch.zeros_like(per_view[:, :, 9:]))

    embedding.square().mean().backward()
    assert clip_tokens.grad is None  # Frozen external CLIP features.
    assert readout.lateral["us3"][0].weight.grad is not None
    assert readout.text_adapter[1].weight.grad is not None
    assert readout.cross_attention.k_proj.weight.grad is not None
    assert views[0]["us3"].grad is not None


def test_view_max_is_elementwise_and_view_order_invariant(readout):
    known = torch.tensor([[[[1.0, 4.0], [5.0, -2.0]], [[3.0, 2.0], [4.0, 6.0]]]])
    torch.testing.assert_close(
        readout.pool_views(known),
        torch.tensor([[[3.0, 4.0], [5.0, 6.0]]]),
    )

    views = _views(batch=1, count=3)
    clip_tokens = torch.randn(1, 77, 512)
    readout.eval()
    with torch.no_grad():
        original = readout(views, clip_tokens)
        permuted = readout([views[2], views[0], views[1]], clip_tokens)
    torch.testing.assert_close(original, permuted, rtol=1e-6, atol=1e-6)


def test_padding_mask_ignores_tokens_after_eot(readout):
    views = _views(batch=1, count=1)
    clip_tokens = torch.randn(1, 77, 512)
    mask = torch.zeros(1, 77, dtype=torch.bool)
    mask[:, :5] = True  # Includes EOT in the active prefix.
    changed_padding = clip_tokens.clone()
    changed_padding[:, 5:] = 1e6

    readout.eval()
    with torch.no_grad():
        original = readout(views, clip_tokens, mask)
        changed = readout(views, changed_padding, mask)
        unmasked = readout(views, clip_tokens)
    assert torch.isfinite(original).all() and torch.isfinite(unmasked).all()
    torch.testing.assert_close(original, changed, rtol=0, atol=0)


def test_accepts_half_precision_frozen_features_with_float32_readout(readout):
    views = [{key: value.half() for key, value in view.items()} for view in _views(batch=1, count=1)]
    clip_tokens = torch.randn(1, 77, 512).half()
    output = readout(views, clip_tokens)
    assert output.shape == (1, 12)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_2d_rope_preserves_norm_and_uses_both_axes():
    keys = torch.ones(1, 1, 4, 8)
    rotated = _rotate_2d_keys(keys, 2, 2, 10000.0)
    torch.testing.assert_close(rotated[..., 0, :], keys[..., 0, :], rtol=0, atol=0)
    torch.testing.assert_close(rotated.norm(dim=-1), keys.norm(dim=-1), rtol=1e-6, atol=1e-6)
    assert not torch.equal(rotated[..., 1, :], rotated[..., 2, :])


def test_rejects_wrong_clip_shape_and_out_of_order_maps(readout):
    views = _views(batch=1, count=1)
    with pytest.raises(ValueError, match=r"\[B, 77, 512\]"):
        readout(views, torch.randn(1, 76, 512))

    wrong_order = [dict(views[0], us8=torch.randn(1, 8, 1, 1))]
    with pytest.raises(ValueError, match="coarse to fine"):
        readout(wrong_order, torch.randn(1, 77, 512))

    with pytest.raises(ValueError, match="at least one active token"):
        readout(views, torch.randn(1, 77, 512), torch.zeros(1, 77, dtype=torch.bool))

    non_prefix_mask = torch.zeros(1, 77, dtype=torch.bool)
    non_prefix_mask[0, [0, 2]] = True
    with pytest.raises(ValueError, match="True prefix"):
        readout(views, torch.randn(1, 77, 512), non_prefix_mask)


def test_compact_eot_readout_reduces_parameters_and_preserves_padding_and_gradients(readout):
    torch.manual_seed(13)
    compact = RobotDIFTPaperReadout(
        {"us3": 4, "us6": 6, "us8": 8},
        fpn_dim=16,
        model_dim=16,
        output_dim=12,
        num_heads=4,
        mlp_hidden_dims=(32,),
        token_pooling="eot_attention",
    ).cpu()
    assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in readout.parameters())

    views = _views(batch=2, count=2, requires_grad=True)
    tokens = torch.randn(2, 77, 512)
    mask = torch.zeros(2, 77, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :12] = True
    changed_padding = tokens.clone()
    changed_padding[~mask] = 1e6
    compact.eval()
    original = compact(views, tokens, mask)
    changed = compact(views, changed_padding, mask)
    torch.testing.assert_close(original, changed, rtol=0, atol=0)
    assert original.shape == (2, 12)
    original.square().mean().backward()
    assert compact.token_pooler.in_proj_weight.grad is not None
    assert compact.lateral["us3"][0].weight.grad is not None
    assert views[0]["us3"].grad is not None
