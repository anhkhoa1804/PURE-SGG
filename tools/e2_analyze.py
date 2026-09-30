"""Block-level CPU analysis for the E2 development run."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap_delta(a, b, seed=11, replicates=5000):
    if not a:
        return None
    rng = random.Random(seed)
    values = [x - y for x, y in zip(a, b)]
    samples = []
    for _ in range(replicates):
        samples.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    samples.sort()
    return {
        "estimate": sum(values) / len(values),
        "ci95": [samples[int(0.025 * replicates)], samples[int(0.975 * replicates) - 1]],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def accuracy(values):
    return sum(values) / len(values) if values else None


def analyze(packet: Path, score_path: Path) -> dict:
    scores = read_jsonl(score_path)
    models = ["M1_cached_C1", "M3_geometry", "M4_object", "M5_frequency"]
    if scores and "M2_valheldout" in scores[0].get("models", {}):
        models.insert(1, "M2_valheldout")
    result = {model: {"n": len(scores), "accuracy": accuracy([int(row["models"][model]["both_correct"]) for row in scores])} for model in models}
    per_pair = defaultdict(lambda: defaultdict(list))
    for row in scores:
        for model in models:
            per_pair[row["relation_pair"]][model].append(int(row["models"][model]["both_correct"]))
    result["per_relation_pair"] = {pair: {model: {"n": len(vals), "accuracy": accuracy(vals)} for model, vals in values.items()} for pair, values in sorted(per_pair.items())}

    comparisons = {}
    for left, right in (("M2_valheldout", "M3_geometry"), ("M2_valheldout", "M4_object"), ("M1_cached_C1", "M3_geometry"), ("M1_cached_C1", "M4_object")):
        if left not in models:
            comparisons[f"{left}_vs_{right}"] = {"status": "unavailable_under_current_lineage", "reason": "No validation-held-out M2 artifact was supplied; canonical train-derived M2 remains unavailable."}
        else:
            comparisons[f"{left}_vs_{right}"] = {"status": "descriptive", **bootstrap_delta([int(x["models"][left]["both_correct"]) for x in scores], [int(x["models"][right]["both_correct"]) for x in scores])}
    model_comparison = {
        "status": "completed_descriptive_baselines; validation_heldout_M2_comparisons_if_supplied",
        "primary_comparisons": ["M2_valheldout_vs_M3_geometry", "M2_valheldout_vs_M4_object"],
        "holm_correction": "apply to the two predeclared M2_valheldout comparisons when inferential p-values are available",
        "comparisons": comparisons,
        "model_accuracy": result,
    }
    (packet / "model_comparison.json").write_text(json.dumps(model_comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    conflicts = []
    geometry_rows = []
    for row in scores:
        m3 = row["models"]["M3_geometry"]
        conflict = m3["image_a_prediction"] != row["gold_relation_a"] or m3["image_b_prediction"] != row["gold_relation_b"]
        if conflict:
            conflicts.append(row["candidate_id"])
        geometry_rows.append({"candidate_id": row["candidate_id"], "relation_pair": row["relation_pair"], "geometry_conflict": conflict, "image_a_prediction": m3["image_a_prediction"], "image_b_prediction": m3["image_b_prediction"], "gold_relation_a": row["gold_relation_a"], "gold_relation_b": row["gold_relation_b"]})
    geometry_analysis = {"status": "completed", "geometry_conflict_block_count": len(conflicts), "geometry_conflict_candidate_ids": conflicts, "definition": "frozen geometry-only baseline predicts a relation inconsistent with human gold on at least one image"}
    (packet / "geometry_analysis.json").write_text(json.dumps(geometry_analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    random_path = packet / "random_cohort_candidates.jsonl"
    random_rows = read_jsonl(random_path)
    random_analysis = {
        "status": "candidate_only_not_human_validated",
        "candidate_count": len(random_rows),
        "relation_pair_counts": {pair: sum(1 for row in random_rows if f"{row['relation_a']}||{row['relation_b']}" == pair) for pair in sorted({f"{row['relation_a']}||{row['relation_b']}" for row in random_rows})},
        "model_scores_used_in_construction": False,
    }
    (packet / "random_cohort_analysis.json").write_text(json.dumps(random_analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    development = {
        "status": "completed_cpu_baseline_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_block_count": len(scores),
        "primary_endpoint": "both_correct_block_accuracy",
        "bootstrap": {"replicates": 5000, "seed": 11, "unit": "accepted block"},
        "human_gate_required_before_interpretation": True,
        "model_scores": result,
        "m2_status": "validation_heldout_if_supplied; canonical_train_derived_unavailable",
        "m6_status": "not_run",
        "geometry_conflict_count": len(conflicts),
    }
    (packet / "development_result.json").write_text(json.dumps(development, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Paper C E2 development report",
        "",
        "Status: CPU baseline analysis completed after human gold freezing.",
        "",
        f"Accepted blocks: {len(scores)}.",
        "",
        "## Established",
        "",
        "- Block-level scoring and bootstrap analysis are reproducible.",
        "- M1 is labeled cached C1 readout; M3/M4/M5 are CPU nuisance baselines.",
        "- If supplied, M2_valheldout is a frozen validation-held-out probe; it is not a canonical train-derived model.",
        "",
        "## Not established",
        "",
        "- This development analysis does not establish general semantic understanding.",
        "- It does not establish causality, geometry removal, open-vocabulary understanding, or task-unseen transfer.",
        "- M6 was not run; task-seen controlled discrimination does not establish semantic understanding or open-vocabulary transfer.",
        "",
        "## Model accuracy",
        "",
    ]
    for model, values in result.items():
        if model == "per_relation_pair":
            continue
        report.append(f"- {model}: n={values['n']}, both-correct={values['accuracy']}")
    (packet / "final_development_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args()
    scores = args.scores or args.packet / "model_scores.jsonl"
    print(json.dumps(analyze(args.packet, scores), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
