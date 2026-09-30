"""Search accessible project storage for a clean M2 representation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def search(roots):
    hits = []
    names = ("rel_feat", "decoder", "pair_logits", "feature", "embedding")
    suffixes = {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npz", ".parquet"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or any(part == ".venv" for part in path.parts):
                continue
            name = path.name.lower()
            if path.suffix.lower() in suffixes or any(token in name for token in names):
                hits.append({"path": str(path), "size": path.stat().st_size})
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", default=[Path(".")])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    hits = search([p.resolve() for p in args.root])
    # The two accepted pair dumps are evaluation artifacts, not clean M2
    # training representations. The historical ladder code is source only.
    clean = [h for h in hits if Path(h["path"]).suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npz", ".parquet"} and "rel_feat" in Path(h["path"]).name.lower() and "eval" not in h["path"]]
    status = "AVAILABLE_CLEAN" if clean else "NOT_AVAILABLE_TRAIN_DERIVED"
    result = {
        "status": status,
        "canonical_train_derived_m2": status,
        "validation_heldout_m2": "AUDIT_WITH_TOOLS_E2_M2_RECONSTRUCTION",
        "searched_roots": [str(p.resolve()) for p in args.root],
        "hits": hits,
        "clean_candidate_hits": clean,
        "reason": "No standalone training rel_feat tensor, decoder head, or canonical train-derived M2 fit artifact was found in accessible project storage. The cached validation dump must be audited separately for a disjoint held-out probe." if not clean else "Candidate requires manual lineage inspection before use.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
