"""Characterize the existing Robot-DIFT S2-FPN without loading diffusion models.

The production module imports and constructs several optional, heavyweight models.
Only the three local S2-FPN definitions are compiled here so this test stays on CPU
and makes no checkpoint or network requests. These tests describe legacy checkpoint
behavior; a future paper-aligned architecture should receive its own tests.
"""

import ast
import copy
from pathlib import Path
from typing import Dict, List

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F


SOURCE = Path(__file__).resolve().parents[1] / "agents/encoders/cleandift_img_encoder.py"
DEFINITIONS = {"_make_group_norm", "S2FPNFuseBlock", "S2FPNBidirectionalFusion"}


def _load_legacy_fusion_class():
    """Load the actual source definitions while avoiding diffusion dependencies."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS
    ]
    assert {node.name for node in selected} == DEFINITIONS
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "F": F, "Dict": Dict, "List": List}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["S2FPNBidirectionalFusion"]


@pytest.fixture
def legacy_fusion():
    torch.manual_seed(7)
    fusion_class = _load_legacy_fusion_class()
    module = fusion_class(
        feature_dims={"us3": 4, "us6": 6, "us8": 8},
        feature_keys=["us3", "us6", "us8"],
        fpn_dim=8,
        device="cpu",
    ).eval()
    maps = {
        "us3": torch.randn(2, 4, 2, 2),
        "us6": torch.randn(2, 6, 4, 4),
        "us8": torch.randn(2, 8, 8, 8),
    }
    return module, maps


def test_legacy_bottom_up_blocks_do_not_affect_output_or_receive_gradients(legacy_fusion):
    module, maps = legacy_fusion
    with torch.no_grad():
        expected = module(maps)
        for block in module.bu_fuse_blocks:
            for parameter in block.parameters():
                parameter.fill_(37.0)
        observed = module(maps)

    torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    module.zero_grad(set_to_none=True)
    module(maps).square().mean().backward()
    assert all(parameter.grad is None for block in module.bu_fuse_blocks for parameter in block.parameters())
    assert any(parameter.grad is not None for block in module.td_fuse_blocks for parameter in block.parameters())


def test_legacy_final_fusion_can_collapse_to_one_top_down_input(legacy_fusion):
    """The two final input halves are identical, so their conv weights can sum."""
    module, maps = legacy_fusion
    captured = []
    hook = module.final_fuse.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
    try:
        with torch.no_grad():
            expected = module(maps)
    finally:
        hook.remove()

    assert len(captured) == 1
    left, right = captured[0].chunk(2, dim=1)
    torch.testing.assert_close(left, right, rtol=0, atol=0)

    legacy_conv = module.final_fuse[0]
    width = left.shape[1]
    collapsed_conv = nn.Conv2d(width, width, kernel_size=1, bias=True)
    with torch.no_grad():
        collapsed_conv.weight.copy_(
            legacy_conv.weight[:, :width] + legacy_conv.weight[:, width:]
        )
        collapsed_conv.bias.copy_(legacy_conv.bias)
        collapsed = nn.Sequential(
            collapsed_conv,
            copy.deepcopy(module.final_fuse[1]),
            copy.deepcopy(module.final_fuse[2]),
        )(left)

    torch.testing.assert_close(collapsed, expected, rtol=1e-5, atol=1e-6)
