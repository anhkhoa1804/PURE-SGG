"""Freeze the model-blind E2 development candidate population."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def freeze(packet: Path) -> dict:
    output = packet / "FROZEN_CANDIDATE_MANIFEST.json"
    candidate_path = packet / "development_candidates.jsonl"
    random_path = packet / "random_cohort_candidates.jsonl"
    image_manifest_path = packet / "images" / "image_manifest.json"
    if output.exists():
        raise FileExistsError(f"frozen manifest already exists: {output}")
    candidates = read_jsonl(candidate_path)
    random = read_jsonl(random_path)
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate development candidate IDs")
    images = [str(x) for row in candidates for x in (row["image_a"], row["image_b"])]
    random_images = {str(x) for row in random for x in (row["image_a"], row["image_b"])}
    if len(images) != len(set(images)):
        raise ValueError("development population reuses an image")
    if set(images) & random_images:
        raise ValueError("development and random cohort overlap")
    forbidden_fields = {"score", "scores", "prediction", "model_error", "wprd", "logit"}
    leaked = sorted(forbidden_fields & set().union(*(row.keys() for row in candidates)))
    if leaked:
        raise ValueError(f"model-score fields present: {leaked}")
    repo = packet.resolve().parents[1]
    selection = json.loads((packet / "selection_flow.json").read_text(encoding="utf-8"))
    manifest = json.loads(image_manifest_path.read_text(encoding="utf-8"))
    image_sha = sha256(image_manifest_path)
    candidate_bytes = candidate_path.read_bytes()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    order_sha = hashlib.sha256("\n".join(candidate_ids).encode("utf-8")).hexdigest()
    frozen = {
        "packet_version": "paper-c-e2-development-v1",
        "status": "frozen_model_blind_candidate_population",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git(repo, "rev-parse", "HEAD"),
        "git_branch": git(repo, "branch", "--show-current"),
        "candidate_file": {
            "path": str(candidate_path.resolve()),
            "size_bytes": candidate_path.stat().st_size,
            "sha256": candidate_sha,
            "candidate_order_sha256": order_sha,
        },
        "image_manifest": {
            "path": str(image_manifest_path.resolve()),
            "size_bytes": image_manifest_path.stat().st_size,
            "sha256": image_sha,
            "count": manifest["count"],
            "missing": manifest["missing"],
        },
        "development": {
            "candidate_count": len(candidates),
            "image_count": len(set(images)),
            "relation_pair_counts": dict(sorted(Counter(f"{x['relation_a']}||{x['relation_b']}" for x in candidates).items())),
            "selection_seed": selection["seed"],
            "relation_pair_whitelist": selection["relation_pair_whitelist"],
        },
        "random_cohort": {
            "candidate_count": len(random),
            "image_count": len(random_images),
            "sha256": sha256(random_path),
            "selection_seed": selection["random_cohort_seed"],
        },
        "model_scores_used_in_selection": False,
    }
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.packet), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
