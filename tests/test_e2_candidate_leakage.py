import json

from tools.e2_check_candidate_leakage import audit


def test_candidate_leakage_passes_clean_packet(tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    row = {"candidate_id": "c1", "image_a": "1", "image_b": "2"}
    for name, rows in (("development_candidates.jsonl", [row]), ("random_cohort_candidates.jsonl", [{"candidate_id": "r1", "image_a": "3", "image_b": "4"}])):
        (packet / name).write_text("".join(json.dumps(x) + "\n" for x in rows))
    (packet / "selection_flow.json").write_text(json.dumps({"seed": 1}) + "\n")
    assert audit(packet)["status"] == "PASS"


def test_candidate_leakage_rejects_model_field(tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "development_candidates.jsonl").write_text(json.dumps({"candidate_id": "c", "score": 1}) + "\n")
    (packet / "random_cohort_candidates.jsonl").write_text("\n")
    (packet / "selection_flow.json").write_text("{}\n")
    assert audit(packet)["status"] == "FAIL"
