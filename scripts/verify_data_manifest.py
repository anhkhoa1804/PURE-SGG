#!/usr/bin/env python3
"""Verify external artifact metadata and hashes from the migration manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(record: dict) -> Path:
    kind = record["root"]
    roots = {
        "repository": ROOT,
        "data": Path(os.environ.get("RESEARCH_NO1_DATA_ROOT", ROOT / "datasets_vg150_clean")),
        "checkpoints": Path(os.environ.get("RESEARCH_NO1_CHECKPOINT_ROOT", ROOT / "checkpoints")),
        "runs": Path(os.environ.get("RESEARCH_NO1_RUN_ROOT", ROOT / "runs")),
    }
    return roots[kind].expanduser() / record["relative_path"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/DATA_AND_CHECKPOINT_MANIFEST.json")
    parser.add_argument("--metadata-only", action="store_true", help="check existence and size without streaming hashes")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    failures = 0
    for record in manifest["artifacts"]:
        if record.get("verification") == "collection_metadata_only":
            print(f"SKIP collection hash: {record['logical_name']}")
            continue
        path = resolve_path(record)
        if not path.is_file():
            print(f"FAIL missing: {record['logical_name']} -> {path}")
            failures += 1
            continue
        size_ok = path.stat().st_size == record["size_bytes"]
        hash_ok = True
        actual_hash = "SKIPPED"
        if not args.metadata_only:
            actual_hash = sha256_file(path)
            hash_ok = actual_hash == record["sha256"]
        verdict = "PASS" if size_ok and hash_ok else "FAIL"
        print(f"{verdict} {record['logical_name']} size={path.stat().st_size} sha256={actual_hash}")
        failures += int(not (size_ok and hash_ok))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
