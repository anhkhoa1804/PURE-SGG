"""Small CPU-only end-to-end fixture for the human-gold boundary."""

import json

from tools.e2_adjudicate import adjudicate
from tools.e2_human_analysis import analyze
from tools.e2_score_models import block_correct
from tools.e2_validate_annotations import validate


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))


def test_synthetic_packet_runs_to_block_level_analysis(tmp_path):
    packet = tmp_path / "packet"
    annotations = packet / "annotations"
    annotations.mkdir(parents=True)
    candidate = {"candidate_id": "c1", "image_a": "1", "image_b": "2", "subject_label": "person", "object_label": "bike", "relation_a": "holding", "relation_b": "riding"}
    _jsonl(packet / "development_candidates.jsonl", [candidate])
    import hashlib
    (packet / "FROZEN_CANDIDATE_MANIFEST.json").write_text(json.dumps({"candidate_file": {"sha256": hashlib.sha256((packet / "development_candidates.jsonl").read_bytes()).hexdigest()}}))
    _jsonl(packet / "human_annotation_tasks.jsonl", [{"candidate_id": "c1", "block_side": "A", "image_id": "1"}, {"candidate_id": "c1", "block_side": "B", "image_id": "2"}])
    _jsonl(packet / "human_block_tasks.jsonl", [{"candidate_id": "c1"}])
    rows = []
    for annotator in ("annotator_1", "annotator_2", "annotator_3"):
        rows.extend([
            {"kind":"image_truth", "task_key":"c1:A", "candidate_id":"c1", "block_side":"A", "image_id":"1", "annotator_id":annotator, "task_version":"paper-c-e2-development-v1", "timestamp_utc":"now", "subject_visible":"yes", "object_visible":"yes", "truth_a":"valid", "truth_b":"invalid", "pair_truth":"relation_a_only", "direction_a":"correct", "direction_b":"uncertain", "confidence_1_to_5":4},
            {"kind":"image_truth", "task_key":"c1:B", "candidate_id":"c1", "block_side":"B", "image_id":"2", "annotator_id":annotator, "task_version":"paper-c-e2-development-v1", "timestamp_utc":"now", "subject_visible":"yes", "object_visible":"yes", "truth_a":"invalid", "truth_b":"valid", "pair_truth":"relation_b_only", "direction_a":"uncertain", "direction_b":"correct", "confidence_1_to_5":4},
            {"kind":"block_solvability", "task_key":"c1", "candidate_id":"c1", "annotator_id":annotator, "task_version":"paper-c-e2-development-v1", "timestamp_utc":"now", "image_a_choice":"description_a", "image_b_choice":"description_b", "confidence_1_to_5":4},
        ])
    for annotator in ("annotator_1", "annotator_2", "annotator_3"):
        _jsonl(annotations / f"{annotator}.jsonl", [r for r in rows if r["annotator_id"] == annotator])
    assert validate(packet, annotations)["status"] == "VALID"
    assert adjudicate(packet, annotations)["accepted_count"] == 1
    assert analyze(packet, annotations)["both_correct_accuracy"] == 1.0
    assert block_correct({"gold_relation_a":"holding", "gold_relation_b":"riding"}, {"holding":1.0,"riding":0.0}, {"holding":0.0,"riding":1.0})["both_correct"]
