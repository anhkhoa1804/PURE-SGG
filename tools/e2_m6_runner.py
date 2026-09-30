"""Interface-only runner for a future frozen M6 visual pilot.

No model is downloaded or loaded here.  The runner validates the frozen block
interface and provides a deterministic synthetic scorer for tests.  A real
backend must be added by an additive protocol amendment with checkpoint and
prompt hashes before inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_blocks(path: Path, limit: int | None = None):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def synthetic_scores(block: dict) -> dict:
    """Deterministic fixture backend; it is never scientific M6 evidence."""
    return {
        "candidate_id": block["candidate_id"],
        "backend": "synthetic_fixture_only",
        "image_a": {block["relation_a"]: 1.0, block["relation_b"]: 0.0},
        "image_b": {block["relation_a"]: 0.0, block["relation_b"]: 1.0},
        "both_correct": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--synthetic-fixture", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_fixture:
        raise SystemExit("No real M6 backend is registered; use --synthetic-fixture only for interface tests")
    rows = [synthetic_scores(row) for row in load_blocks(args.blocks, args.limit)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"status": "synthetic_interface_only", "blocks": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
