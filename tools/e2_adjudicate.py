"""Adjudicate human E2 annotations without silently majority-voting ambiguity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.e2_validate_annotations import compatibility_errors


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def consensus(rows, field, override=None):
    if override is not None and field in override:
        return override[field], "adjudicated"
    values = [row.get(field) for row in rows]
    if values and all(value == values[0] for value in values):
        return values[0], "unanimous"
    return "unresolved", "disagreement"


def adjudicate(packet: Path, annotations: Path, adjudication_path: Path | None = None) -> dict:
    validation_path = packet / "annotation_validation.json"
    if not validation_path.exists():
        raise RuntimeError("run e2_validate_annotations.py before adjudication")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "VALID":
        raise RuntimeError(f"annotation validation status is {validation.get('status')}")

    candidates = {str(x["candidate_id"]): x for x in read_jsonl(packet / "development_candidates.jsonl")}
    image_groups = defaultdict(list)
    block_groups = defaultdict(list)
    for annotator in ("annotator_1", "annotator_2", "annotator_3"):
        for row in read_jsonl(annotations / f"{annotator}.jsonl"):
            if row["kind"] == "image_truth":
                image_groups[(str(row["candidate_id"]), str(row["block_side"]))].append(row)
            else:
                block_groups[str(row["candidate_id"])].append(row)

    overrides = {}
    adjudicator_ids = {}
    if adjudication_path and adjudication_path.exists():
        for row in read_jsonl(adjudication_path):
            overrides[(str(row["candidate_id"]), str(row.get("side", "")))] = row.get("decision", row)
            adjudicator_ids[(str(row["candidate_id"]), str(row.get("side", "")))] = str(row.get("adjudicator_id", ""))

    truths = []
    accepted = []
    unresolved = []
    for candidate_id, candidate in candidates.items():
        summary = {"candidate_id": candidate_id, "adjudication_occurred": False}
        image_values = {}
        reasons = []
        for side in ("A", "B"):
            rows = image_groups[(candidate_id, side)]
            override = overrides.get((candidate_id, side))
            fields = {}
            sources = {}
            for field in ("subject_visible", "object_visible", "truth_a", "truth_b", "pair_truth", "direction_a", "direction_b"):
                fields[field], sources[field] = consensus(rows, field, override)
                if sources[field] == "adjudicated":
                    summary["adjudication_occurred"] = True
            image_values[side] = fields
            if "unresolved" in fields.values():
                reasons.append(f"{side.lower()}_annotator_disagreement")
            reasons.extend(f"{side.lower()}_{reason}" for reason in compatibility_errors(fields))
            if fields["subject_visible"] != "yes" or fields["object_visible"] != "yes":
                reasons.append(f"{side.lower()}_visibility")
            if fields["pair_truth"] not in {"relation_a_only", "relation_b_only"}:
                reasons.append(f"{side.lower()}_pair_truth_{fields['pair_truth']}")
        a, b = image_values["A"], image_values["B"]
        if a["pair_truth"] == "relation_a_only" and (a["truth_a"], a["truth_b"]) != ("valid", "invalid"):
            reasons.append("image_a_relation_fields")
        if b["pair_truth"] == "relation_b_only" and (b["truth_a"], b["truth_b"]) != ("invalid", "valid"):
            reasons.append("image_b_relation_fields")
        if a["direction_a"] != "correct" or b["direction_b"] != "correct":
            reasons.append("direction_not_verified")
        if a["pair_truth"] != "relation_a_only" or b["pair_truth"] != "relation_b_only":
            reasons.append("candidate_orientation_not_verified")
        status = "accepted" if not reasons else "excluded"
        if any("disagreement" in reason or "unresolved" in reason for reason in reasons):
            status = "requires_adjudication"
            unresolved.append(candidate_id)
        record = {
            **candidate,
            "status": status,
            "exclusion_reasons": sorted(set(reasons)),
            "image_truth": image_values,
            "adjudication_occurred": summary["adjudication_occurred"],
            "adjudicator_ids": {
                side: adjudicator_ids.get((candidate_id, side), "")
                for side in ("A", "B")
                if (candidate_id, side) in adjudicator_ids
            },
            "annotation_summary": {
                "annotator_count": len(image_groups[(candidate_id, "A")]),
                "block_annotator_count": len(block_groups[candidate_id]),
            },
        }
        truths.append(record)
        if status == "accepted":
            accepted.append({
                **candidate,
                "gold_relation_a": candidate["relation_a"],
                "gold_relation_b": candidate["relation_b"],
                "human_truth_status": "accepted_unambiguous",
            })

    (packet / "adjudicated_truth.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in truths), encoding="utf-8")
    (packet / "accepted_development_blocks.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in accepted), encoding="utf-8")
    template = [x for x in truths if x["status"] == "requires_adjudication"]
    (packet / "adjudication_template.jsonl").write_text("".join(json.dumps({"candidate_id": x["candidate_id"], "status": x["status"], "image_truth": x["image_truth"], "decision": "pending", "adjudicator_id": "", "notes": ""}, sort_keys=True) + "\n" for x in template), encoding="utf-8")
    result = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "excluded_count": sum(x["status"] == "excluded" for x in truths),
        "requires_adjudication_count": len(unresolved),
        "accepted_path": str((packet / "accepted_development_blocks.jsonl").resolve()),
    }
    (packet / "adjudication_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    args = parser.parse_args()
    try:
        result = adjudicate(args.packet, args.annotations, args.adjudication)
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, indent=2))
        sys.exit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
