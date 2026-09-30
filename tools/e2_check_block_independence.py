"""Check image and block independence for the frozen E2 packet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(packet: Path) -> dict:
    dev = read_jsonl(packet / "development_candidates.jsonl")
    random = read_jsonl(packet / "random_cohort_candidates.jsonl")
    dev_ids = [str(x["candidate_id"]) for x in dev]
    random_ids = [str(x["candidate_id"]) for x in random]
    dev_images = [str(x) for row in dev for x in (row["image_a"], row["image_b"])]
    random_images = [str(x) for row in random for x in (row["image_a"], row["image_b"])]
    pair_keys = [
        (str(row["image_a"]), str(row["image_b"]), str(row["subject_label"]), str(row["object_label"]), str(row["relation_a"]), str(row["relation_b"]))
        for row in dev + random
    ]
    duplicate_dev = [x for x, n in Counter(dev_images).items() if n > 1]
    duplicate_random = [x for x, n in Counter(random_images).items() if n > 1]
    result = {
        "status": "PASS" if len(dev_ids) == len(set(dev_ids)) and len(random_ids) == len(set(random_ids)) and not duplicate_dev and not duplicate_random and not (set(dev_images) & set(random_images)) and all(str(r["image_a"]) != str(r["image_b"]) for r in dev + random) and len(pair_keys) == len(set(pair_keys)) else "FAIL",
        "development_blocks": len(dev),
        "random_blocks": len(random),
        "development_images": len(set(dev_images)),
        "random_images": len(set(random_images)),
        "duplicate_development_images": duplicate_dev,
        "duplicate_random_images": duplicate_random,
        "development_random_overlap": sorted(set(dev_images) & set(random_images)),
        "duplicate_candidate_ids": [x for x, n in Counter(dev_ids + random_ids).items() if n > 1],
        "duplicate_relation_image_combinations": [list(x) for x, n in Counter(pair_keys).items() if n > 1],
        "within_block_same_image": [str(r["candidate_id"]) for r in dev + random if str(r["image_a"]) == str(r["image_b"])],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.packet)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
