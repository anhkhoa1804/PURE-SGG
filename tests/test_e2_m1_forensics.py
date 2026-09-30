import pytest
import torch

from tools.e2_score_models import pair_slot, score_m1


def test_cached_pair_lookup_is_ordered_and_unique():
    dump = {"pairs": [torch.tensor([[1, 2], [0, 1]])]}
    assert pair_slot(dump, 0, 1, 2) == 0
    assert pair_slot(dump, 0, 0, 1) == 1


def test_cached_pair_lookup_fails_on_duplicate_slot():
    dump = {"pairs": [torch.tensor([[1, 2], [1, 2]])]}
    with pytest.raises(ValueError):
        pair_slot(dump, 0, 1, 2)


def test_cached_pair_lookup_accepts_content_identical_duplicate_slots():
    dump = {
        "pairs": [torch.tensor([[1, 2], [1, 2]])],
        "rel_feat": [[torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])]],
        "text_logits": [[torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.2])]],
    }
    assert pair_slot(dump, 0, 1, 2) == 0


def test_m1_uses_exact_predicate_mapping():
    dump = {
        "pairs": [torch.tensor([[0, 1]])],
        "pred_vocab": ["riding", "holding"],
        "text_logits": [[torch.tensor([0.2, 0.9])]],
    }
    row = {"image_a_index": 0, "image_b_index": 0, "subject_index_a": 0, "object_index_a": 1, "subject_index_b": 0, "object_index_b": 1, "relation_a": "holding", "relation_b": "riding", "gold_relation_a": "holding", "gold_relation_b": "riding"}
    result = score_m1(dump, row)
    assert result["image_a_prediction"] == "holding"
    assert result["image_b_prediction"] == "holding"
