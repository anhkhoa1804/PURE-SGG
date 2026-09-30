"""Compute agreement and human solvability for the E2 development packet."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def block_bootstrap(values: list[float], seed: int = 11, replicates: int = 5000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        samples.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    samples.sort()
    return [samples[int(0.025 * replicates)], samples[int(0.975 * replicates) - 1]]


def pairwise_agreement(groups: dict, fields: list[str]) -> dict:
    output = {}
    for field in fields:
        pairs = total_pairs = unanimous = 0
        for rows in groups.values():
            values = [row.get(field) for row in rows]
            if len(values) != 3:
                continue
            pair_count = sum(values[i] == values[j] for i in range(3) for j in range(i + 1, 3))
            pairs += pair_count
            total_pairs += 3
            unanimous += int(len(set(values)) == 1)
        output[field] = {
            "pairwise_agreement": pairs / total_pairs if total_pairs else None,
            "unanimous_rate": unanimous / len(groups) if groups else None,
            "groups": len(groups),
        }
    return output


def optional_alpha(groups: dict, field: str):
    try:
        import krippendorff  # type: ignore
    except ImportError:
        return {"status": "unavailable", "value": None}
    units = [list(row.get(field) for row in rows) for rows in groups.values() if len(rows) == 3]
    if not units:
        return {"status": "no_data", "value": None}
    return {"status": "computed", "value": float(krippendorff.alpha(reliability_data=units, level_of_measurement="nominal"))}


def analyze(packet: Path, annotations: Path) -> dict:
    accepted = read_jsonl(packet / "accepted_development_blocks.jsonl")
    image_rows = defaultdict(list)
    block_rows = defaultdict(list)
    for annotator in ("annotator_1", "annotator_2", "annotator_3"):
        for row in read_jsonl(annotations / f"{annotator}.jsonl"):
            if row["kind"] == "image_truth":
                image_rows[(str(row["candidate_id"]), str(row["block_side"]))].append(row)
            else:
                block_rows[str(row["candidate_id"])].append(row)

    agreement = {
        "status": "complete" if accepted else "pending_no_accepted_blocks",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_truth": pairwise_agreement(image_rows, ["subject_visible", "object_visible", "truth_a", "truth_b", "pair_truth", "direction_a", "direction_b"]),
        "block_solvability": pairwise_agreement(block_rows, ["image_a_choice", "image_b_choice"]),
        "krippendorff_alpha": {
            "pair_truth": optional_alpha(image_rows, "pair_truth"),
            "image_a_choice": optional_alpha(block_rows, "image_a_choice"),
            "image_b_choice": optional_alpha(block_rows, "image_b_choice"),
        },
    }
    (packet / "human_agreement.json").write_text(json.dumps(agreement, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    observations = []
    per_pair = defaultdict(list)
    block_values = []
    pair_block_values = defaultdict(list)
    for block in accepted:
        rows = block_rows[str(block["candidate_id"])]
        values_for_block = []
        for row in rows:
            correct = row.get("image_a_choice") == "description_a" and row.get("image_b_choice") == "description_b"
            values_for_block.append(int(correct))
            block_values.append(int(correct))
            observations.append(int(correct))
            per_pair[f"{block['relation_a']}||{block['relation_b']}"].append(int(correct))
        pair_block_values[f"{block['relation_a']}||{block['relation_b']}"].append(sum(values_for_block) / len(values_for_block))

    by_block = []
    for block in accepted:
        values = [int(row.get("image_a_choice") == "description_a" and row.get("image_b_choice") == "description_b") for row in block_rows[str(block["candidate_id"])] ]
        by_block.append((block, values))
    block_means = [sum(values) / len(values) for _, values in by_block if values]
    point = sum(block_means) / len(block_means) if block_means else None
    ci = block_bootstrap(block_means) if block_means else None
    family = {}
    for key, values in sorted(per_pair.items()):
        family[key] = {
            "n_annotator_block_observations": len(values),
            "n_blocks": len(pair_block_values[key]),
            "accuracy": sum(values) / len(values) if values else None,
            "response_ci95": wilson(sum(values), len(values)),
            "block_cluster_ci95": block_bootstrap(pair_block_values[key]),
        }
    gates = {"overall_preferred": point is not None and point >= 0.90, "overall_lower_bound": ci is not None and ci[0] >= 0.80,
             "each_relation_pair": bool(family) and all(v["accuracy"] is not None and v["accuracy"] >= 0.80 for v in family.values())}
    result = {
        "status": "complete" if observations else "pending_no_accepted_blocks",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_block_count": len(accepted),
        "annotator_block_observation_count": len(observations),
        "both_correct_accuracy": point,
        "ci95": ci,
        "response_wilson_ci95": wilson(sum(observations), len(observations)) if observations else None,
        "per_relation_pair": family,
        "development_gates": gates,
        "gate_pass": bool(observations) and all(gates.values()),
        "primary_unit": "accepted block; each block contributes the mean of its available independent annotator responses",
        "block_count_for_inference": len(block_means),
        "annotator_response_count_for_description": len(observations),
    }
    (packet / "human_solvability.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.packet, args.annotations), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
