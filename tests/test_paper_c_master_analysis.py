"""CPU regression checks for saved-prediction analysis guards."""
import numpy as np
import pytest

from tools.build_paper_c_master_analysis import check_identity, require_close, saved_metrics, validate_destinations


def test_saved_logloss_stable_with_large_logits_and_present_class_recall():
    logits = np.full((2, 51), -1000.0)
    logits[0, 0] = 1000.0
    logits[1, 1] = 1000.0
    result = saved_metrics(logits, np.array([0, 1]))
    assert result["log_loss"] == 0.0
    assert result["accuracy"] == 1.0
    assert result["macro_recall"] == 1.0
    assert result["classes_present"] == 2


def test_nonfinite_or_wrong_vocabulary_logits_rejected():
    with pytest.raises(ValueError):
        saved_metrics(np.full((1, 51), np.nan), np.array([0]))
    with pytest.raises(ValueError):
        saved_metrics(np.zeros((1, 50)), np.array([0]))


def test_identity_guard_rejects_duplicates_instead_of_ordinal_join():
    z = {"image_id": np.array(["a", "a"]), "subject_index": np.array([1, 1]),
         "object_index": np.array([2, 2]), "predicate": np.array(["on", "on"]),
         "targets": np.array([0, 0])}
    with pytest.raises(ValueError, match="Duplicate"):
        check_identity(z, images=1, rows=2)


def test_evidence_disagreement_fails_instead_of_overwriting_result():
    with pytest.raises(ValueError, match="Evidence mismatch"):
        require_close(0.1, 0.2, "frozen metric")


def test_existing_output_refused_without_explicit_verified_draft_refresh(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    with pytest.raises(FileExistsError):
        validate_destinations(tmp_path, out, tmp_path / "package", False)
