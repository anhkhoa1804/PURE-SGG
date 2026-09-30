"""Validate independent E2 annotation files against the frozen packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


TASK_VERSION = "paper-c-e2-development-v1"
ANNOTATORS = ("annotator_1", "annotator_2", "annotator_3")
TRUTH = {"valid", "invalid", "uncertain", "not_visible"}
PAIR_TRUTH = {"relation_a_only", "relation_b_only", "both_valid", "neither_visible", "uncertain"}
VISIBILITY = {"yes", "no", "uncertain"}
DIRECTION = {"correct", "reversed", "uncertain"}
CHOICES = {"description_a", "description_b", "uncertain"}


def compatibility_errors(row: dict) -> list[str]:
    """Return logical contradictions in one image-truth annotation.

    This does not decide semantic truth.  It only prevents an internally
    contradictory record from reaching adjudication as if it were valid.
    """
    errors = []
    subject = row.get("subject_visible")
    obj = row.get("object_visible")
    ta, tb, pt = row.get("truth_a"), row.get("truth_b"), row.get("pair_truth")
    da, db = row.get("direction_a"), row.get("direction_b")
    if subject != "yes" or obj != "yes":
        if pt in {"relation_a_only", "relation_b_only"}:
            errors.append("single_relation_requires_both_objects_visible")
    if pt == "relation_a_only" and (ta, tb) != ("valid", "invalid"):
        errors.append("pair_truth_relation_a_only_must_match_truth_fields")
    if pt == "relation_b_only" and (ta, tb) != ("invalid", "valid"):
        errors.append("pair_truth_relation_b_only_must_match_truth_fields")
    if pt == "both_valid" and (ta, tb) != ("valid", "valid"):
        errors.append("both_valid_requires_two_valid_truth_fields")
    if pt == "neither_visible" and (ta, tb) not in {("not_visible", "not_visible"), ("invalid", "invalid")}:
        errors.append("neither_visible_requires_nonvalid_truth_fields")
    if pt in {"uncertain", "both_valid", "neither_visible"} and (da == "correct" or db == "correct"):
        errors.append("ambiguous_pair_cannot_have_verified_direction")
    return errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(packet: Path, annotations: Path) -> dict:
    frozen_path = packet / "FROZEN_CANDIDATE_MANIFEST.json"
    candidates_path = packet / "development_candidates.jsonl"
    tasks_path = packet / "human_annotation_tasks.jsonl"
    blocks_path = packet / "human_block_tasks.jsonl"
    errors = []
    warnings = []
    if not frozen_path.exists():
        errors.append("missing FROZEN_CANDIDATE_MANIFEST.json")
        result = {"status": "INVALID", "errors": errors}
        (packet / "annotation_validation.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if sha256(candidates_path) != frozen["candidate_file"]["sha256"]:
        errors.append("development_candidates.jsonl hash differs from frozen manifest")
    candidates = read_jsonl(candidates_path)
    candidate_ids = {str(x["candidate_id"]) for x in candidates}
    expected_images = {(str(x["candidate_id"]), str(x["block_side"]), str(x["image_id"])) for x in read_jsonl(tasks_path)}
    expected_blocks = {str(x["candidate_id"]) for x in read_jsonl(blocks_path)}
    complete = True
    counts = {}
    for annotator in ANNOTATORS:
        path = annotations / f"{annotator}.jsonl"
        if not path.exists():
            complete = False
            counts[annotator] = {"image_truth": 0, "block_solvability": 0}
            warnings.append(f"missing annotation file: {path}")
            continue
        rows = read_jsonl(path)
        seen = set()
        counts[annotator] = {"image_truth": 0, "block_solvability": 0}
        for row in rows:
            kind = row.get("kind")
            key = (kind, str(row.get("task_key")))
            if key in seen:
                errors.append(f"duplicate {annotator} task {key}")
            seen.add(key)
            if row.get("annotator_id") != annotator:
                errors.append(f"cross-annotator record in {annotator}: {row.get('annotator_id')}")
            if row.get("task_version") != TASK_VERSION:
                errors.append(f"wrong task version in {annotator}")
            if not row.get("timestamp_utc"):
                errors.append(f"missing timestamp in {annotator}: {key}")
            if kind == "image_truth":
                counts[annotator][kind] += 1
                expected = (str(row.get("candidate_id")), str(row.get("block_side")), str(row.get("image_id")))
                if expected not in expected_images:
                    errors.append(f"unknown image task in {annotator}: {expected}")
                if row.get("subject_visible") not in VISIBILITY or row.get("object_visible") not in VISIBILITY:
                    errors.append(f"invalid visibility in {annotator}: {key}")
                if row.get("truth_a") not in TRUTH or row.get("truth_b") not in TRUTH:
                    errors.append(f"invalid relation truth in {annotator}: {key}")
                if row.get("pair_truth") not in PAIR_TRUTH:
                    errors.append(f"invalid pair truth in {annotator}: {key}")
                if row.get("direction_a") not in DIRECTION or row.get("direction_b") not in DIRECTION:
                    errors.append(f"invalid direction in {annotator}: {key}")
                if row.get("confidence_1_to_5") not in {1, 2, 3, 4, 5}:
                    errors.append(f"invalid confidence in {annotator}: {key}")
                errors.extend(f"inconsistent annotation in {annotator}: {key}: {e}" for e in compatibility_errors(row))
            elif kind == "block_solvability":
                counts[annotator][kind] += 1
                if str(row.get("candidate_id")) not in expected_blocks:
                    errors.append(f"unknown block task in {annotator}: {key}")
                if row.get("image_a_choice") not in CHOICES or row.get("image_b_choice") not in CHOICES:
                    errors.append(f"invalid block choice in {annotator}: {key}")
                if row.get("confidence_1_to_5") not in {1, 2, 3, 4, 5}:
                    errors.append(f"invalid confidence in {annotator}: {key}")
            else:
                errors.append(f"invalid task kind in {annotator}: {kind}")
        if counts[annotator]["image_truth"] != len(expected_images) or counts[annotator]["block_solvability"] != len(expected_blocks):
            complete = False
            warnings.append(f"incomplete task counts for {annotator}: {counts[annotator]}")
    status = "VALID" if complete and not errors else ("ANNOTATION_PENDING" if not errors else "INVALID")
    result = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_version": TASK_VERSION,
        "candidate_sha256": sha256(candidates_path),
        "expected_candidate_count": len(candidate_ids),
        "counts": counts,
        "errors": errors,
        "warnings": warnings,
    }
    (packet / "annotation_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.packet, args.annotations)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "INVALID":
        sys.exit(1)
    if result["status"] == "ANNOTATION_PENDING":
        sys.exit(2)


if __name__ == "__main__":
    main()
