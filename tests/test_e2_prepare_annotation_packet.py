import json

from PIL import Image

from tools.e2_prepare_annotation_packet import prepare


def test_prepare_creates_three_independent_truth_and_block_tasks(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (100, 80), "white").save(image_dir / "1.jpg")
    Image.new("RGB", (100, 80), "white").save(image_dir / "2.jpg")
    dev = tmp_path / "development.jsonl"
    dev.write_text(json.dumps({
        "candidate_id": "c1",
        "image_a": "1",
        "image_b": "2",
        "subject_label": "person",
        "object_label": "bike",
        "relation_a": "holding",
        "relation_b": "riding",
        "subject_box_a": [10, 10, 40, 60],
        "object_box_a": [45, 20, 80, 70],
        "subject_box_b": [15, 5, 50, 60],
        "object_box_b": [40, 15, 90, 75],
    }) + "\n")
    out = tmp_path / "packet"
    result = prepare(dev, image_dir, out)
    assert result == {"blocks": 1, "images": 2, "image_tasks": 6, "block_tasks": 3}
    image_tasks = [json.loads(x) for x in (out / "human_annotation_tasks.jsonl").read_text().splitlines()]
    block_tasks = [json.loads(x) for x in (out / "human_block_tasks.jsonl").read_text().splitlines()]
    assert {x["annotator_id"] for x in image_tasks} == {"annotator_1", "annotator_2", "annotator_3"}
    assert all("truth_a" in x and "truth_b" in x and "pair_truth" in x for x in image_tasks)
    assert len(block_tasks) == 3
    assert all((out / "overlays" / f"{i}.jpg").is_file() for i in ("1", "2"))
