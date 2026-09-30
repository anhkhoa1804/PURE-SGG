import json

from tools.e2_adjudicate import adjudicate
from tools.e2_human_analysis import analyze as analyze_humans
from tools.e2_score_models import choose
from tools.e2_validate_annotations import validate


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_choose_ties_are_explicit():
    assert choose({"holding": 1.0, "riding": 1.0}) == "tie"
    assert choose({"holding": 1.0, "riding": 0.9}) == "holding"


def test_annotation_validator_reports_pending_without_fabricating_answers(tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    candidate_path = packet / "development_candidates.jsonl"
    candidate_path.write_text(json.dumps({"candidate_id": "c1"}) + "\n")
    import hashlib
    (packet / "FROZEN_CANDIDATE_MANIFEST.json").write_text(json.dumps({
        "candidate_file": {"sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest()}
    }))
    (packet / "human_annotation_tasks.jsonl").write_text(json.dumps({"candidate_id": "c1", "block_side": "A", "image_id": "1"}) + "\n")
    (packet / "human_block_tasks.jsonl").write_text(json.dumps({"candidate_id": "c1"}) + "\n")
    result = validate(packet, packet / "annotations")
    assert result["status"] == "ANNOTATION_PENDING"


def test_adjudication_requires_unanimity_and_writes_gold(tmp_path):
    packet = tmp_path / "packet"
    annotations = packet / "annotations"
    annotations.mkdir(parents=True)
    candidate = {
        "candidate_id": "c1", "image_a": "1", "image_b": "2",
        "subject_label": "person", "object_label": "bike",
        "relation_a": "holding", "relation_b": "riding",
    }
    (packet / "development_candidates.jsonl").write_text(json.dumps(candidate) + "\n")
    (packet / "annotation_validation.json").write_text(json.dumps({"status": "VALID"}))
    rows = []
    for aid in ("annotator_1", "annotator_2", "annotator_3"):
        for side, image_id, ta, tb, pt, da, db in (
            ("A", "1", "valid", "invalid", "relation_a_only", "correct", "uncertain"),
            ("B", "2", "invalid", "valid", "relation_b_only", "uncertain", "correct"),
        ):
            rows.append({
                "kind": "image_truth", "task_key": f"c1:{side}", "candidate_id": "c1",
                "block_side": side, "image_id": image_id, "annotator_id": aid,
                "task_version": "paper-c-e2-development-v1", "timestamp_utc": "now",
                "subject_visible": "yes", "object_visible": "yes", "truth_a": ta,
                "truth_b": tb, "pair_truth": pt, "direction_a": da, "direction_b": db,
            })
        rows.append({
            "kind": "block_solvability", "task_key": "c1", "candidate_id": "c1",
            "annotator_id": aid, "task_version": "paper-c-e2-development-v1", "timestamp_utc": "now",
            "image_a_choice": "description_a", "image_b_choice": "description_b",
        })
    _write_jsonl(annotations / "annotator_1.jsonl", [x for x in rows if x["annotator_id"] == "annotator_1"])
    _write_jsonl(annotations / "annotator_2.jsonl", [x for x in rows if x["annotator_id"] == "annotator_2"])
    _write_jsonl(annotations / "annotator_3.jsonl", [x for x in rows if x["annotator_id"] == "annotator_3"])
    result = adjudicate(packet, annotations)
    assert result["accepted_count"] == 1
    accepted = [json.loads(x) for x in (packet / "accepted_development_blocks.jsonl").read_text().splitlines()]
    assert accepted[0]["gold_relation_a"] == "holding"
    human = analyze_humans(packet, annotations)
    assert human["both_correct_accuracy"] == 1.0
