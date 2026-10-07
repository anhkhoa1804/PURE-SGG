import numpy as np
import pytest

from tools.r2_nested_correction import (
    assert_image_disjoint,
    assert_same_population,
    derive_cal_partition,
    deterministic_vocabulary,
    fit_nested_alpha,
    metric,
    nested_logits,
)
from pathlib import Path


def test_zero_residual_recovers_parent_logits_exactly():
    parent = np.arange(12, dtype=np.float64).reshape(3, 4)
    residual = np.full((3, 4), -np.log(4.0))
    assert np.array_equal(nested_logits(parent, residual, 0.0), parent)


def test_nested_alpha_includes_parent_as_a_valid_optimum():
    parent = np.zeros((4, 3), dtype=np.float64)
    residual = np.zeros_like(parent)
    targets = np.asarray([0, 1, 2, 0])
    alpha, nested_loss, parent_loss = fit_nested_alpha(parent, residual, targets)
    assert alpha == 0.0
    assert nested_loss == parent_loss


def test_exact_identity_population_is_required_for_comparison():
    assert_same_population([("im1", 0, 1, "near")], [("im1", 0, 1, "near")])
    with pytest.raises(ValueError, match="populations differ"):
        assert_same_population([("im1", 0, 1, "near")], [("im2", 0, 1, "near")])
    with pytest.raises(ValueError, match="duplicate"):
        assert_same_population(["a", "a"], ["a"])


def test_split_and_vocabulary_are_deterministic_and_fit_only():
    cal_ids = [f"img-{i}" for i in range(10)]
    first = derive_cal_partition(cal_ids, seed=12, fit_count=7)
    assert first == derive_cal_partition(cal_ids, seed=12, fit_count=7)
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == set(cal_ids)
    assert deterministic_vocabulary([" Zebra ", "apple", "APPLE", "zebra"]) == ["apple", "zebra"]


def test_image_split_overlap_is_rejected_to_guard_label_leakage():
    assert_image_disjoint({"fit"}, {"cal-fit"}, {"cal-check"})
    with pytest.raises(ValueError, match="overlap"):
        assert_image_disjoint({"same-image"}, {"same-image"})


def test_nested_logits_rejects_invalid_parent_comparison_shapes():
    with pytest.raises(ValueError, match="identical"):
        nested_logits(np.zeros((2, 3)), np.zeros((2, 4)), 0.2)
    with pytest.raises(ValueError, match="nonnegative"):
        nested_logits(np.zeros((2, 3)), np.zeros((2, 3)), -0.1)


def test_saved_calibrated_pair_prior_reproduces_registered_validation_ll():
    path = (Path(__file__).resolve().parents[1] /
            "runs/paper_c_fullscale_residual_audit_20261003T084403Z/fullscale_residual_row_predictions.npz")
    if not path.exists():
        pytest.skip("frozen fullscale prediction artifact is not present")
    with np.load(path, allow_pickle=False) as saved:
        score = metric(saved["pair_prior_calibrated_logits"], saved["targets"])
    assert score["rows"] == 14991
    assert score["log_loss"] == pytest.approx(1.6323624327446515, abs=1e-7)


def test_nested_extended_arm_is_not_an_independent_parent_comparison():
    parent = np.asarray([[2.0, -1.0], [0.1, 0.3]])
    residual = np.asarray([[-0.1, -2.0], [-1.0, -0.2]])
    extended_zero = nested_logits(parent, residual, 0.0)
    assert np.array_equal(extended_zero, parent)
    # The residual API takes the fixed parent logits; there is no second,
    # independently fitted parent parameterization in this comparison.
    with pytest.raises(ValueError, match="identical"):
        nested_logits(parent, residual[:1], 0.25)
