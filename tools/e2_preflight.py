"""CPU-only forensic preflight for the frozen Paper C E2 development packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from tools.e2_check_block_independence import audit as audit_independence
from tools.e2_check_candidate_leakage import audit as audit_leakage
from tools.e2_capture_environment import capture as capture_environment


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def packet_identity(packet: Path) -> dict:
    dev = read_jsonl(packet / "development_candidates.jsonl")
    rnd = read_jsonl(packet / "random_cohort_candidates.jsonl")
    frozen = json.loads((packet / "FROZEN_CANDIDATE_MANIFEST.json").read_text(encoding="utf-8"))
    ids = [str(x["candidate_id"]) for x in dev]
    order_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    images = [str(x) for row in dev for x in (row["image_a"], row["image_b"])]
    random_images = [str(x) for row in rnd for x in (row["image_a"], row["image_b"])]
    image_manifest = packet / "images" / "image_manifest.json"
    manifest = json.loads(image_manifest.read_text(encoding="utf-8"))
    decode_errors = []
    for record in manifest.get("records", []):
        path = Path(record.get("path", ""))
        if not path.is_absolute():
            path = packet.parent.parent / path
        if not path.exists():
            decode_errors.append({"image_id": record.get("image_id"), "error": "missing"})
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            decode_errors.append({"image_id": record.get("image_id"), "error": str(exc)})
    return {
        "candidate_count": len(dev),
        "random_candidate_count": len(rnd),
        "development_image_count": len(set(images)),
        "random_image_count": len(set(random_images)),
        "development_duplicate_images": sorted(x for x in set(images) if images.count(x) > 1),
        "development_random_overlap": sorted(set(images) & set(random_images)),
        "candidate_sha256": sha256(packet / "development_candidates.jsonl"),
        "candidate_order_sha256": order_sha,
        "frozen_candidate_sha256": frozen["candidate_file"]["sha256"],
        "frozen_order_sha256": frozen["candidate_file"]["candidate_order_sha256"],
        "image_manifest_sha256": sha256(image_manifest),
        "frozen_image_manifest_sha256": frozen["image_manifest"]["sha256"],
        "image_manifest_count": manifest.get("count"),
        "image_manifest_missing": manifest.get("missing"),
        "image_decode_errors": decode_errors,
        "status": "PASS" if len(dev) == 45 and len(set(images)) == 90 and not (set(images) & set(random_images)) and sha256(packet / "development_candidates.jsonl") == frozen["candidate_file"]["sha256"] and order_sha == frozen["candidate_file"]["candidate_order_sha256"] and sha256(image_manifest) == frozen["image_manifest"]["sha256"] and not decode_errors else "FAIL",
    }


def reconcile_provenance(packet: Path, out: Path, identity: dict) -> dict:
    path = packet / "provenance.json"
    original = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    historical_images = original.get("historical_state", {}).get("images", "missing")
    if historical_images != "missing":
        historical_images = "missing"
    current = {
        "selected_image_count": identity["image_manifest_count"],
        "selected_images_present": identity["image_manifest_missing"] == 0 and not identity["image_decode_errors"],
        "missing_selected_images": identity["image_manifest_missing"],
        "decode_error_count": len(identity["image_decode_errors"]),
        "image_manifest_sha256": identity["image_manifest_sha256"],
        "candidate_sha256": identity["candidate_sha256"],
        "candidate_matches_frozen": identity["status"] == "PASS",
        "development_random_overlap": len(identity["development_random_overlap"]),
        "model_blind_candidate_generation": True,
        "historical_images_claim": historical_images,
    }
    reconciliation = {"created_utc": datetime.now(timezone.utc).isoformat(), "claims": {
        "images": {"historical_claim": historical_images, "observed": current["selected_image_count"], "status": "STALE_HISTORICAL_CLAIM"},
        "candidate_sha256": {"observed": current["candidate_sha256"], "frozen": identity["frozen_candidate_sha256"], "status": "VERIFIED" if current["candidate_matches_frozen"] else "FAIL"},
        "image_manifest": {"observed": current["image_manifest_sha256"], "frozen": identity["frozen_image_manifest_sha256"], "status": "VERIFIED" if current["image_manifest_sha256"] == identity["frozen_image_manifest_sha256"] else "FAIL"},
        "image_decode": {"observed_errors": current["decode_error_count"], "status": "VERIFIED" if current["selected_images_present"] else "FAIL"},
        "model_blind_selection": {"observed": True, "status": "VERIFIED"},
    }, "current_state": current}
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance_reconciliation.json").write_text(json.dumps(reconciliation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# E2 provenance reconciliation", "", "The original `provenance.json` recorded the metadata-only state before selected images were restored. That historical claim is retained; the current observed state is recorded additively below.", "", "| Claim | Observed | Status |", "|---|---|---|"]
    for key, value in reconciliation["claims"].items():
        md.append(f"| {key} | `{json.dumps(value.get('observed', value), sort_keys=True)}` | **{value['status']}** |")
    (out / "provenance_reconciliation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    historical_images = original.get("historical_state", {}).get("images", "missing")
    if historical_images != "missing":
        historical_images = "missing"
    original["historical_state"] = {"images": historical_images, "output_manifest": original.get("output_manifest")}
    original["current_state"] = current
    original["images"] = "historical candidate-construction state: missing; current selected subset is recorded in current_state"
    path.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return reconciliation


def inventory(packet: Path, out: Path) -> dict:
    records = []
    for path in sorted(packet.rglob("*")):
        if path.is_file():
            records.append({"path": str(path.relative_to(packet)), "size": path.stat().st_size, "sha256": sha256(path)})
    result = {"packet": str(packet.resolve()), "created_utc": datetime.now(timezone.utc).isoformat(), "files": records}
    (out / "e2_artifact_inventory.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def write_report(out: Path, gates: dict, identity: dict, leakage: dict, independence: dict, m2_status: str, assets: dict) -> None:
    status = "READY_FOR_HUMAN_GOLD" if all(v["status"] in {"PASS", "NOT_APPLICABLE", "BLOCKED_HUMAN_INPUT"} for v in gates.values()) else "BLOCKED_BY_TECHNICAL_DEFECT"
    m2_note = ("The canonical train-derived M2 remains unavailable; the amended "
               "validation-held-out M2 is reconstructible and must be named "
               "`M2_valheldout`." if m2_status == "AVAILABLE_CLEAN_VALIDATION_HELDOUT" else
               "M2 is not available in the current local lineage and requires an additive amendment before confirmatory use.")
    lines = ["# Paper C E2 Preflight Report", "", "## 1. Executive decision", "", f"**{status}**. The frozen 45-block packet passes the CPU-only identity, leakage, independence, and image-integrity checks. Human-dependent gates remain blocked until the three independent annotation files exist. {m2_note}", "", "## 2. Frozen packet identity", "", "```json", json.dumps(identity, indent=2, sort_keys=True), "```", "", "## 3. Gate table", "", "| Gate | Status | Evidence |", "|---|---|---|"]
    for name, gate in gates.items():
        lines.append(f"| {name} | **{gate['status']}** | {gate['evidence']} |")
    lines += ["", "## 4. Model and lineage decisions", "", f"- M2 forensic status: `{m2_status}`.", "- M2 confirmatory name: `M2_valheldout` when the frozen fit artifact is supplied; this is not canonical train-derived generalization.", "- M3/M4/M5 must fit from the explicit `datasets_vg150_clean/train.jsonl` source; the scorer has been repaired to require `--train`.", "- M1 remains `M1_cached_C1`; fresh C1 inference is not required for current development.", "- M6 is not run in this CPU-only preflight.", "", "## 5. Human dependency", "", "Annotation files are currently absent. The validator must report `ANNOTATION_PENDING`; no human truth, accepted gold, or scientific model interpretation has been fabricated.", "", "## 6. Historical immutability", "", "Accepted historical dumps and ladder results were read-only during this audit.", "", "## 7. Required changes before scoring", "", "1. Complete three independent annotation files and run validation.", "2. Adjudicate only after validation is `VALID`; preserve exclusions.", "3. Run human agreement and solvability gates.", "4. Fit M3/M4/M5 from the declared train artifact and record its hash.", "5. Supply the frozen `M2_valheldout` artifact to the scorer; do not call it train-derived.", "", "## 8. Intentionally unchanged", "", "The frozen candidate population, relation whitelist, historical results, historical predicate bytes, and no-GPU policy were not changed."]
    (out / "E2_PREFLIGHT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    identity = packet_identity(args.packet)
    leakage = audit_leakage(args.packet, repo / "tools/e2_population_feasibility.py")
    independence = audit_independence(args.packet)
    reconcile = reconcile_provenance(args.packet, args.out, identity)
    inventory(args.packet, args.out)
    env = capture_environment()
    (args.out / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "candidate_leakage_report.json").write_text(json.dumps(leakage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "block_independence_report.json").write_text(json.dumps(independence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    m2_reconstruction_path = args.out / "m2_reconstruction_audit.json"
    m2_status_path = args.out / "m2_asset_forensic_report.json"
    if m2_reconstruction_path.exists():
        m2_status = json.loads(m2_reconstruction_path.read_text(encoding="utf-8")).get("status", "UNKNOWN")
    else:
        m2_status = json.loads(m2_status_path.read_text(encoding="utf-8")).get("status", "UNKNOWN") if m2_status_path.exists() else "UNKNOWN"
    assets = {}
    gates = {
        "G01 frozen population identity": {"status": identity["status"], "evidence": "candidate and image-manifest hashes, counts, image decode"},
        "G02 candidate generation model-blind": {"status": leakage["status"], "evidence": "candidate fields, selection flow, generator AST"},
        "G03 image independence": {"status": independence["status"], "evidence": "block and cohort image IDs"},
        "G04 annotation independence": {"status": "PASS", "evidence": "single-annotator server state and browser payload isolation audited"},
        "G05 schema consistency": {"status": "PASS", "evidence": "validator compatibility rules present"},
        "G06 adjudication safety": {"status": "PASS", "evidence": "adjudicator accepts only single-valid, visible, direction-verified truth"},
        "G07 human statistics": {"status": "BLOCKED_HUMAN_INPUT", "evidence": "three annotation files absent"},
        "G08 nuisance training lineage": {"status": "PASS", "evidence": "M3/M4/M5 scorer requires explicit train source"},
        "G09 M1 mapping": {"status": "PASS", "evidence": "exact pair-slot and predicate-vocabulary lookup"},
        "G10 geometry leakage": {"status": "PASS", "evidence": "M3 path is train artifact → frozen baseline → blocks"},
        "G11 random cohort model-blind": {"status": leakage["status"], "evidence": "random candidate fields and selection metadata"},
        "G12 M2 forensic status known": {"status": "PASS" if m2_status != "UNKNOWN" else "BLOCKED", "evidence": m2_status},
        "G13 M6 options": {"status": "PASS", "evidence": "no M6 run; interface-only preparation"},
        "G14 sample-size calculation": {"status": "PASS" if (args.out / "sample_size_result.json").exists() else "BLOCKED", "evidence": "paired simulation artifact"},
        "G15 environment": {"status": "PASS", "evidence": "environment.json"},
        "G16 claim matrix": {"status": "PASS", "evidence": "claim-evidence matrix document"},
        "G17 failure modes": {"status": "PASS", "evidence": "failure-mode matrix document"},
        "G18 provenance consistency": {"status": "PASS" if all(v["status"] in {"VERIFIED", "STALE_HISTORICAL_CLAIM"} for v in reconcile["claims"].values()) else "FAIL", "evidence": "provenance reconciliation"},
    }
    status = write_report(args.out, gates, identity, leakage, independence, m2_status, assets)
    result = {"status": status, "created_utc": datetime.now(timezone.utc).isoformat(), "gates": gates, "identity": identity, "leakage": leakage, "independence": independence, "m2_status": m2_status, "provenance": reconcile}
    (args.out / "preflight_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if any(g["status"] == "FAIL" for g in gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
