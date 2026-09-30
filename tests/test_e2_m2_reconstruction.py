import json

import torch

from tools.e2_m2_reconstruction import audit, build_fit_rows


def _dump():
    return {
        "image_id": ["dev", "train", "fit"],
        "pairs": [torch.tensor([[0, 1], [0, 1]]), torch.tensor([[0, 1]]), torch.tensor([[0, 1]])],
        "rel_feat": [torch.ones(2, 768), torch.full((1, 768), 2.0), torch.full((1, 768), 3.0)],
        "text_logits": [torch.zeros(2, 3), torch.zeros(1, 3), torch.zeros(1, 3)],
        "model_logits": [torch.zeros(2, 3), torch.zeros(1, 3), torch.zeros(1, 3)],
        "prior_rows": [torch.zeros(2, 3), torch.zeros(1, 3), torch.zeros(1, 3)],
        "cls_logits": [torch.zeros(2, 3), torch.zeros(1, 3), torch.zeros(1, 3)],
        "gt_subj_idx": [[0], [0], [0]],
        "gt_obj_idx": [[1], [1], [1]],
        "gt_pred": [["holding"], ["holding"], ["holding"]],
        "pred_vocab": ["holding", "riding", "relation"],
        "background_predicate_indices": [2],
    }


def test_fit_rows_exclude_all_frozen_e2_images():
    X, y, stats = build_fit_rows(_dump(), {"dev"})
    assert tuple(X.shape) == (2, 768)
    assert y.tolist() == [0, 0]
    assert stats["fit_image_count"] == 2


def test_audit_proves_candidate_resolution_and_heldout_fit(tmp_path):
    dump_path = tmp_path / "dump.pt"
    packet = tmp_path / "packet"
    packet.mkdir()
    torch.save(_dump(), dump_path)
    candidate = {
        "candidate_id": "c1", "image_a": "dev", "image_b": "train",
        "subject_index_a": 0, "object_index_a": 1,
        "subject_index_b": 0, "object_index_b": 1,
        "relation_a": "holding", "relation_b": "holding",
    }
    (packet / "development_candidates.jsonl").write_text(json.dumps(candidate) + "\n")
    result = audit(dump_path, packet, tmp_path / "audit.json")
    assert result["status"] == "AVAILABLE_CLEAN_VALIDATION_HELDOUT"
    assert result["fit_rows"]["fit_image_count"] == 1
    assert result["candidate_cache_check"]["candidate_sides_resolved"] == 2
