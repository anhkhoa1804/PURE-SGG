from tools.e2_validate_annotations import compatibility_errors


def test_single_relation_truth_requires_matching_fields():
    row = {"subject_visible": "yes", "object_visible": "yes", "truth_a": "valid", "truth_b": "valid", "pair_truth": "relation_a_only", "direction_a": "correct", "direction_b": "uncertain"}
    assert compatibility_errors(row)


def test_unambiguous_relation_truth_is_compatible():
    row = {"subject_visible": "yes", "object_visible": "yes", "truth_a": "valid", "truth_b": "invalid", "pair_truth": "relation_a_only", "direction_a": "correct", "direction_b": "uncertain"}
    assert compatibility_errors(row) == []


def test_both_valid_cannot_be_primary_truth():
    row = {"subject_visible": "yes", "object_visible": "yes", "truth_a": "valid", "truth_b": "valid", "pair_truth": "both_valid", "direction_a": "correct", "direction_b": "correct"}
    assert "ambiguous_pair_cannot_have_verified_direction" in compatibility_errors(row)
