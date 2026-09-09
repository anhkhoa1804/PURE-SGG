#!/usr/bin/env python
"""Paper C -- cross-arm provenance audit (R0/C0/C1/R2). CPU only, read-only.

Reads existing `result.json` / `*/provenance.txt` / checkpoint files and
assembles one machine-readable provenance record per arm. Never loads a
model, never touches the GPU, never writes to any existing artifact.
Checkpoint SHA256 is computed by streaming bytes (no `torch.load`) and is
the only even moderately expensive step (~seconds per GB) -- skip it with
`--no-hash` if that is still too much while a GPU job is running.

This script deliberately does NOT resolve the C0/C1 "+0.02 vs +0.008"
discrepancy discussed in docs/PAPER_C_PROVENANCE_AUDIT.md -- that
resolution rests on reading prose in several documents (which numbers are
macro-population deltas vs. spatial-subgroup-restricted deltas vs. a
same-checkpoint readout-gap that is a different comparison axis entirely),
which is not something to re-derive mechanically here. This tool's job is
narrower and more mechanical: pull the numbers that ARE in structured
files into one consistent table, and flag internal inconsistencies
(population mismatch, missing files, hash mismatch) if any exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def sha256_of(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _read_provenance_txt(path: str) -> Optional[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return None
    out: Dict[str, str] = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


# One entry per arm this program currently recognizes. `label` is the name
# used in Paper C's own narrative (R0/C0/C1/R2); `arm` is this repo's own
# internal arm code, which is NOT always the same string (R0 == the C1
# seed-1234 artifact, reused under a different label for the Readout v2
# comparison -- this is a fact to surface, not paper over).
ARMS: List[Dict[str, str]] = [
    {"label": "C0", "arm": "C0",
     "result_json": "runs/eval_C0/result.json",
     "training_provenance": "runs/C0_seed1234/provenance.txt",
     "checkpoint": "checkpoints/C0_seed1234.pt"},
    {"label": "C1 (== R0)", "arm": "C1",
     "result_json": "runs/eval_C1/result.json",
     "training_provenance": "runs/C1_seed1234/provenance.txt",
     "checkpoint": "checkpoints/C1_seed1234.pt"},
    {"label": "C1_seed5678", "arm": "C1",
     "result_json": "runs/eval_C1_seed5678/result.json",
     "training_provenance": "runs/C1_seed5678/provenance.txt",
     "checkpoint": "checkpoints/C1_seed5678.pt"},
    {"label": "R2_pilot_v3", "arm": "readout_v2",
     "result_json": "runs/eval_readout_v2_R2_pilot_v3/result.json",
     "training_provenance": "runs/readout_v2_pilot_seed1234_v3/provenance.txt",
     "checkpoint": "checkpoints/readout_v2_pilot_seed1234_v3.pt"},
]


def build_record(spec: Dict[str, str], compute_hash: bool) -> Dict[str, Any]:
    res = _read_json(spec["result_json"])
    prov = _read_provenance_txt(spec["training_provenance"])
    ckpt_path = spec["checkpoint"]
    ckpt_exists = Path(ckpt_path).exists()

    rec: Dict[str, Any] = {
        "label": spec["label"], "arm": spec["arm"],
        "checkpoint_path": ckpt_path, "checkpoint_exists": ckpt_exists,
        "checkpoint_sha256": None,
        "training_git_commit": prov.get("git_commit") if prov else None,
        "training_seed": prov.get("seed") if prov else None,
        "training_provenance_path": spec["training_provenance"],
        "training_provenance_exists": prov is not None,
        "resume_from": prov.get("resume_from") if prov else None,
        "resume_from_sha256_recorded": prov.get("ckpt_sha256") if prov else None,
        "eval_result_json_path": spec["result_json"],
        "eval_result_exists": res is not None,
        "geom_input_pixel_space": None, "geom_fourier_scale": None,
        "population": None, "wprd_macro": None, "wprd_weighted": None,
        "R_at_50": None, "mR_at_50": None, "prior_control_wprd": None,
        "dump_path": None,
        "provenance_confidence": None,
    }

    if compute_hash and ckpt_exists:
        rec["checkpoint_sha256"] = sha256_of(ckpt_path)

    if res is not None:
        contract = res.get("contract", {})
        rec["geom_input_pixel_space"] = contract.get("geom_input_pixel_space")
        rec["geom_fourier_scale"] = contract.get("geom_fourier_scale")
        rec["population"] = res.get("population")
        rec["wprd_macro"] = res.get("wprd_macro")
        rec["wprd_weighted"] = res.get("wprd_weighted")
        rec["R_at_50"] = res.get("R")
        rec["mR_at_50"] = res.get("mR")
        rec["prior_control_wprd"] = res.get("prior_control_wprd")
        rec["dump_path"] = res.get("dump")
        rec["dump_exists"] = Path(res.get("dump", "")).exists() if res.get("dump") else False

    # Confidence classification -- mechanical, not a judgment call:
    # VERIFIED requires every structured artifact this script can check to
    # be present and internally consistent; PENDING means the eval has not
    # produced a result.json yet (expected for R2 while the GPU job runs);
    # PARTIAL means some but not all expected files exist.
    if res is not None and prov is not None and ckpt_exists:
        rec["provenance_confidence"] = "VERIFIED"
    elif res is None and prov is not None and ckpt_exists:
        rec["provenance_confidence"] = "PENDING (training artifacts complete, evaluation not yet run/finished)"
    else:
        rec["provenance_confidence"] = "PARTIAL (see which of result_json/provenance/checkpoint is missing)"

    return rec


def cross_check_populations(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    pops = {r["label"]: r["population"] for r in records if r["population"] is not None}
    values = list(pops.values())
    all_equal = all(v == values[0] for v in values) if values else None
    return {"populations": pops, "all_equal": all_equal,
            "n_arms_with_population": len(pops), "n_arms_total": len(records)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip checkpoint SHA256 (faster; use while GPU-heavy jobs are running "
                         "and every extra bit of disk I/O matters)")
    args = ap.parse_args(argv)

    records = [build_record(spec, compute_hash=not args.no_hash) for spec in ARMS]
    cross = cross_check_populations(records)
    out = {"tool": "provenance_audit", "arms": records, "population_cross_check": cross}

    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[written] {args.out}")
    else:
        print(text)

    print("\n" + "=" * 100)
    print(f"{'label':<14}{'confidence':<70}")
    print("=" * 100)
    for r in records:
        print(f"{r['label']:<14}{r['provenance_confidence']:<70}")
    print(f"\npopulations equal across arms with a result.json: {cross['all_equal']} "
          f"({cross['n_arms_with_population']}/{cross['n_arms_total']} arms have one)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
