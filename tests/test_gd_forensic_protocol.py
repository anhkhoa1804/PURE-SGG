import torch
import pytest

from tools.gd_forensic_protocol import (
    GD_STATUSES,
    classify_gd,
    planned_gpu_matrix,
    required_condition_fields,
    tensor_error_metrics,
)


def test_matrix_contains_repeats_batch_controls_and_checkpoint_p_control():
    rows = planned_gpu_matrix()
    names = {row["condition"] for row in rows}
    assert len(rows) == len(names)
    assert "X_R0_noP_9_repeat" in names
    assert "Z_R2_P_registered_9_repeat" in names
    assert "X_R0_noP_12padded" in names
    assert "Z_R2_P_registered_12padded" in names
    assert "Zp_R2_checkpointP_registered_9" in names
    assert all("resolves" in row and row["resolves"] for row in rows)


def test_tensor_error_metrics_reports_fraction_rms_and_relative_error():
    ref = torch.tensor([[1.0, 0.0], [2.0, -4.0]])
    actual = torch.tensor([[1.5, 0.0], [2.0, -2.0]])
    result = tensor_error_metrics(actual, ref)
    assert result["element_count"] == 4
    assert result["changed_element_count"] == 2
    assert result["changed_fraction"] == 0.5
    assert result["max_abs"] == 2.0
    assert result["rms_abs"] == pytest.approx(1.0625 ** 0.5)
    assert result["max_relative"] == 0.5


def test_classification_is_conservative_and_has_exactly_three_statuses():
    assert set(GD_STATUSES) == {
        "GD-EXPLAINED-INERT",
        "GD-SCIENTIFICALLY-MATERIAL",
        "GD-UNRESOLVED",
    }
    assert classify_gd(
        reproduction_complete=False,
        causal_explanation_proven=True,
        endpoint_operationally_irrelevant=True,
        endpoint_materially_changed=False,
    ) == "GD-UNRESOLVED"
    assert classify_gd(
        reproduction_complete=True,
        causal_explanation_proven=True,
        endpoint_operationally_irrelevant=True,
        endpoint_materially_changed=False,
    ) == "GD-EXPLAINED-INERT"
    assert classify_gd(
        reproduction_complete=True,
        causal_explanation_proven=True,
        endpoint_operationally_irrelevant=False,
        endpoint_materially_changed=True,
    ) == "GD-SCIENTIFICALLY-MATERIAL"


def test_required_fields_cover_input_state_runtime_and_endpoint():
    fields = set(required_condition_fields())
    for required in (
        "image_ids", "pixel_tensor_sha256", "parameter_hashes",
        "buffer_hashes", "prototype_read_events", "rng_state_before",
        "rel_feat_vs_baseline", "final_batch_wprd",
    ):
        assert required in fields


def test_nonfinite_and_shape_errors_cannot_appear_as_zero_difference():
    assert tensor_error_metrics(torch.tensor([float('nan')]), torch.ones(1))['finite'] is False
    assert tensor_error_metrics(torch.ones(2), torch.ones(1))['shape_mismatch'] is True
    good = tensor_error_metrics(torch.ones(2, 3), torch.ones(2, 3))
    assert good['changed_row_count'] == 0
    assert good['absolute_quantiles']['1.0'] == 0


def test_local_equality_or_unproven_cause_is_not_inert():
    assert classify_gd(reproduction_complete=True, causal_explanation_proven=False,
        endpoint_operationally_irrelevant=True, endpoint_materially_changed=False) == 'GD-UNRESOLVED'
    assert classify_gd(reproduction_complete=True, causal_explanation_proven=True,
        endpoint_operationally_irrelevant=True, endpoint_materially_changed=True) == 'GD-SCIENTIFICALLY-MATERIAL'
