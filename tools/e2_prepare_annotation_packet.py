"""Prepare reproducible, model-blind human annotation tasks for E2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageDraw


ANNOTATORS = ("annotator_1", "annotator_2", "annotator_3")
TRUTH_VALUES = ("valid", "invalid", "uncertain", "not_visible")
PAIR_VALUES = ("relation_a_only", "relation_b_only", "both_valid", "neither_visible", "uncertain")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _draw_overlay(image_path: Path, output_path: Path, subject_box: Iterable[float], object_box: Iterable[float]) -> None:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        s = [int(round(float(x))) for x in subject_box]
        o = [int(round(float(x))) for x in object_box]
        draw.rectangle(s, outline=(220, 30, 30), width=max(3, image.width // 300))
        draw.rectangle(o, outline=(30, 80, 220), width=max(3, image.width // 300))
        draw.rectangle((s[0], s[1], s[0] + 28, s[1] + 24), fill=(220, 30, 30))
        draw.rectangle((o[0], o[1], o[0] + 28, o[1] + 24), fill=(30, 80, 220))
        draw.text((s[0] + 5, s[1] + 3), "S", fill="white")
        draw.text((o[0] + 5, o[1] + 3), "O", fill="white")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)


def prepare(development_path: Path, image_dir: Path, output_dir: Path) -> Dict[str, int]:
    blocks = _read_jsonl(development_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    image_tasks: List[Dict[str, Any]] = []
    block_tasks: List[Dict[str, Any]] = []
    seen_images = set()

    for block in blocks:
        for side in ("a", "b"):
            image_id = str(block[f"image_{side}"])
            image_path = image_dir / f"{image_id}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"missing image for annotation packet: {image_path}")
            if image_id in seen_images:
                raise ValueError(f"development packet reuses image {image_id}")
            seen_images.add(image_id)
            overlay_path = overlay_dir / f"{image_id}.jpg"
            _draw_overlay(
                image_path,
                overlay_path,
                block[f"subject_box_{side}"],
                block[f"object_box_{side}"],
            )
            for annotator_id in ANNOTATORS:
                image_tasks.append({
                    "annotator_id": annotator_id,
                    "candidate_id": block["candidate_id"],
                    "block_side": side.upper(),
                    "image_id": image_id,
                    "image_path": str(image_path.resolve()),
                    "overlay_path": str(overlay_path.resolve()),
                    "subject_label": block["subject_label"],
                    "object_label": block["object_label"],
                    "relation_a": block["relation_a"],
                    "relation_b": block["relation_b"],
                    "subject_visible": "pending",
                    "object_visible": "pending",
                    "truth_a": "pending",
                    "truth_b": "pending",
                    "pair_truth": "pending",
                    "direction_a": "pending",
                    "direction_b": "pending",
                    "confidence_1_to_5": None,
                    "notes": "",
                })
        for annotator_id in ANNOTATORS:
            block_tasks.append({
                "annotator_id": annotator_id,
                "candidate_id": block["candidate_id"],
                "image_a": str(block["image_a"]),
                "image_b": str(block["image_b"]),
                "subject_label": block["subject_label"],
                "object_label": block["object_label"],
                "relation_a": block["relation_a"],
                "relation_b": block["relation_b"],
                "description_a": f"The highlighted {block['subject_label']} is {block['relation_a']} the highlighted {block['object_label']}.",
                "description_b": f"The highlighted {block['subject_label']} is {block['relation_b']} the highlighted {block['object_label']}.",
                "image_a_choice": "pending",
                "image_b_choice": "pending",
                "both_correct": "pending",
                "confidence_1_to_5": None,
                "notes": "",
            })

    schema = {
        "status": "annotation_pending",
        "required_annotators": 3,
        "annotator_ids": list(ANNOTATORS),
        "image_truth_values": list(TRUTH_VALUES),
        "pair_truth_values": list(PAIR_VALUES),
        "block_choice_values": ["description_a", "description_b", "uncertain"],
        "primary_inclusion": [
            "visible_subject_and_object",
            "truth_a_and_truth_b_are_unambiguous",
            "pair_truth_is_relation_a_only_or_relation_b_only",
            "direction_agreement",
            "human_adjudication_complete",
        ],
        "independence_rule": "Annotators receive the same frozen task packet independently and do not see other annotations before adjudication.",
        "truth_rule": "Record truth_a and truth_b separately; pair_truth is relation_a_only, relation_b_only, both_valid, neither_visible, or uncertain.",
    }
    (output_dir / "annotation_schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, rows in (("human_annotation_tasks.jsonl", image_tasks), ("human_block_tasks.jsonl", block_tasks)):
        with (output_dir / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    return {"blocks": len(blocks), "images": len(seen_images), "image_tasks": len(image_tasks), "block_tasks": len(block_tasks)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.development, args.image_dir, args.out_dir), sort_keys=True))


if __name__ == "__main__":
    main()
