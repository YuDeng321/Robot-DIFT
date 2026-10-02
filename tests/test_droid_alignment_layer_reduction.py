"""The paper's map sum must have the intended scale and gradients."""

import pytest
import torch

from agents.encoders.cleandift_img_encoder import reduce_alignment_loss_terms


def test_paper_sum_uses_one_minus_cosine_for_each_map():
    first = torch.tensor(-0.8, requires_grad=True)
    second = torch.tensor(-0.6, requires_grad=True)
    loss = reduce_alignment_loss_terms(
        {"neg_cossim_us3": first, "neg_cossim_us6": second}, "sum"
    )
    assert loss.item() == pytest.approx(0.6)
    loss.backward()
    assert first.grad.item() == pytest.approx(1.0)
    assert second.grad.item() == pytest.approx(1.0)


def test_legacy_mean_preserves_checkpoint_training_scale():
    first = torch.tensor(-0.8, requires_grad=True)
    second = torch.tensor(-0.6, requires_grad=True)
    loss = reduce_alignment_loss_terms(
        {"neg_cossim_us3": first, "neg_cossim_us6": second}, "mean"
    )
    assert loss.item() == pytest.approx(-0.7)
    loss.backward()
    assert first.grad.item() == pytest.approx(0.5)
    assert second.grad.item() == pytest.approx(0.5)


def test_reduction_rejects_invalid_configuration_and_empty_maps():
    with pytest.raises(ValueError, match="Unsupported"):
        reduce_alignment_loss_terms({"neg_cossim_us3": torch.tensor(0.0)}, "median")
    with pytest.raises(ValueError, match="no feature-map"):
        reduce_alignment_loss_terms({}, "sum")
