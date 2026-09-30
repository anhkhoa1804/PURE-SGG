"""Audit that the frozen E2 candidate packet is model-blind.

The checker is intentionally conservative: it inspects candidate fields,
selection metadata, and the executable candidate-generator namespace.  It does
not infer intent from model scores after the fact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path


FORBIDDEN_FIELDS = {
    "score", "scores", "logit", "logits", "prediction", "predictions",
    "probability", "probabilities", "embedding", "embeddings", "loss",
    "model_error", "model_confidence", "confidence_from_model", "wprd",
    "decoder_output", "model_output",
}
FORBIDDEN_MODULES = {"e2_score_models", "e2_analyze", "e2_m6_runner"}


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recursive_keys(value, prefix=""):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_FIELDS:
                found.append(path)
            found.extend(recursive_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(recursive_keys(child, f"{prefix}[{index}]"))
    return found


def source_audit(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = []
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Name):
            names.add(node.id.lower())
    module_hits = sorted({m for m in imports for forbidden in FORBIDDEN_MODULES if m.endswith(forbidden)})
    executable_name_hits = sorted(n for n in names if n in {"logit", "prediction", "model_scores", "model_output", "wprd"})
    return {"path": str(path), "forbidden_imports": module_hits, "forbidden_executable_names": executable_name_hits}


def audit(packet: Path, generator: Path | None = None) -> dict:
    candidates_path = packet / "development_candidates.jsonl"
    random_path = packet / "random_cohort_candidates.jsonl"
    candidates = read_jsonl(candidates_path)
    random_rows = read_jsonl(random_path)
    hits = []
    for filename, rows in ((candidates_path.name, candidates), (random_path.name, random_rows)):
        for index, row in enumerate(rows):
            for hit in recursive_keys(row):
                hits.append({"file": filename, "row": index, "field": hit})
    flow = json.loads((packet / "selection_flow.json").read_text(encoding="utf-8"))
    flow_hits = recursive_keys(flow)
    source = source_audit(generator) if generator and generator.exists() else {"status": "not_checked"}
    passed = not hits and not flow_hits and not source.get("forbidden_imports") and not source.get("forbidden_executable_names")
    return {
        "status": "PASS" if passed else "FAIL",
        "model_derived_candidate_fields": hits,
        "model_derived_selection_fields": flow_hits,
        "candidate_generator_source": source,
        "candidate_sha256": sha256(candidates_path),
        "random_sha256": sha256(random_path),
        "rule": "candidate inclusion, ranking, pairing, and random-cohort selection must not consume model-derived quantities",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--generator", type=Path, default=Path("tools/e2_population_feasibility.py"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.packet, args.generator)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
