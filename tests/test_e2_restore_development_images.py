import json

from tools.e2_restore_development_images import image_ids_from_packets


def test_image_ids_from_packets_is_unique_and_sorted(tmp_path):
    path = tmp_path / "packet.jsonl"
    rows = [
        {"image_a": "20", "image_b": "3"},
        {"image_a": "3", "image_b": "100"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    assert image_ids_from_packets([path]) == ["3", "20", "100"]
