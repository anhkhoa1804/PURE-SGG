import json
from tools.e2_annotation_app import _html, state_payload


def test_state_exposes_only_current_annotator_answers(tmp_path):
    packet = tmp_path / "packet"
    (packet / "annotations").mkdir(parents=True)
    (packet / "images").mkdir()
    (packet / "overlays").mkdir()
    (packet / "human_annotation_tasks.jsonl").write_text(json.dumps({"annotator_id": "annotator_1", "candidate_id": "c", "block_side": "A", "image_id": "1", "subject_label": "person", "object_label": "bike", "relation_a": "holding", "relation_b": "riding"}) + "\n")
    (packet / "human_block_tasks.jsonl").write_text(json.dumps({"annotator_id": "annotator_1", "candidate_id": "c", "image_a": "1", "image_b": "2", "subject_label": "person", "object_label": "bike", "relation_a": "holding", "relation_b": "riding", "description_a": "a", "description_b": "b"}) + "\n")
    (packet / "annotations" / "annotator_2.jsonl").write_text(json.dumps({"kind": "image_truth", "task_key": "c:A", "truth_a": "valid"}) + "\n")
    payload = state_payload(packet.resolve(), "annotator_1", packet / "annotations" / "annotator_1.jsonl")
    body = json.dumps(payload)
    assert "annotator_2" not in body
    assert "truth_a" not in body


def test_annotation_html_does_not_reference_other_sessions():
    html = _html()
    assert "other annotations" in html
    assert "model scores" in html
