from __future__ import annotations

import copy

import torch

from tools.forensic_compare_rel_feat import compare_dumps


def _dump():
    return {
        "image_id": ["a", "b"],
        "pairs": [torch.tensor([[0, 1]]), torch.tensor([[0, 1]])],
        "subj_label": [["person"], ["person"]],
        "obj_label": [["bike"], ["bike"]],
        "obj_labels": [["person", "bike"], ["person", "bike"]],
        "gt_subj_idx": [[0], [0]], "gt_obj_idx": [[1], [1]],
        "gt_pred": [["on"], ["on"]],
        "rel_feat": [torch.zeros(1, 3), torch.ones(1, 3)],
        "text_logits": [torch.zeros(1, 2), torch.ones(1, 2)],
        "model_logits": [torch.zeros(1, 2), torch.ones(1, 2)],
        "cls_logits": [torch.zeros(1, 2), torch.ones(1, 2)],
        "prior_rows": [torch.zeros(1, 2), torch.ones(1, 2)],
    }


def test_dump_forensics_reports_localized_change(tmp_path):
    a = _dump()
    b = copy.deepcopy(a)
    b["rel_feat"][1][0, 0] = 0.25
    pa, pb = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(a, pa); torch.save(b, pb)

    result = compare_dumps(str(pa), str(pb), batch_size=2)
    assert result["population_identity"]["status"] == "EXACT_MATCH"
    assert result["fields"]["rel_feat"]["changed_entries"] == 1
    assert result["summary"]["n_changed_images"] == 1
    assert result["changed_images"][0]["image_id"] == "b"
    assert result["changed_images"][0]["batch_index"] == 0
