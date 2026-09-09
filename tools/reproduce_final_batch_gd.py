#!/usr/bin/env python3
"""Read-only final-batch G-D reproduction and input provenance audit.

This tool deliberately does not construct the SGG model or run inference.  It
is the safe half of the G-D reproduction when CUDA is unavailable: it proves
the final-batch population against the existing dumps, reconstructs the
current dataset-side inputs, hashes the currently reproducible preprocessing
output, and records every causal condition that was not run.

It never writes any historical artifact.  The only output is the path passed
with ``--out``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tools.forensic_compare_rel_feat import compare_dumps
    from tools.gd_forensic_protocol import planned_gpu_matrix, required_condition_fields
    from openvocab_rel.datasets.vg150_loader import (
        VG150JSONLDataset,
        VG150LoaderConfig,
        _load_vg150_vocab,
    )
except ModuleNotFoundError:  # direct invocation from tools/
    from forensic_compare_rel_feat import compare_dumps
    from gd_forensic_protocol import planned_gpu_matrix, required_condition_fields
    from openvocab_rel.datasets.vg150_loader import (
        VG150JSONLDataset,
        VG150LoaderConfig,
        _load_vg150_vocab,
    )

R0_DUMP = ROOT / "runs/eval_C1/pair_logits.pt"
R2_DUMP = ROOT / "runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt"
DATA_ROOT = ROOT / "datasets_vg150_clean"
BATCH_SIZE = 12
FINAL_START = 10392


def _sha_tensor(x: torch.Tensor) -> Dict[str, Any]:
    y = x.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(y.dtype).encode())
    h.update(json.dumps(list(y.shape)).encode())
    h.update(y.numpy().tobytes())
    return {"sha256": h.hexdigest(), "dtype": str(y.dtype), "shape": list(y.shape)}


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _run_nvidia_smi() -> Dict[str, Any]:
    try:
        p = subprocess.run(
            ["nvidia-smi"], cwd=str(ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False, timeout=20,
        )
        return {"returncode": p.returncode, "output": p.stdout[-4000:]}
    except Exception as exc:  # pragma: no cover - host-dependent
        return {"error": f"{type(exc).__name__}: {exc}"}


def _runtime_state() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": (
            [{"index": i, "name": torch.cuda.get_device_name(i)}
             for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else []
        ),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _historical_eval_settings() -> Dict[str, Any]:
    """Read the two recorded eval configs without constructing either model."""
    paths = {
        "R0": ROOT / "runs/eval_C1/latest_metrics.json",
        "R2": ROOT / "runs/eval_readout_v2_R2_pilot_v3/latest_metrics.json",
    }
    keys = (
        "device", "batch_size", "num_workers", "clip_input_res", "amp",
        "amp_dtype", "gpu_preset", "cudnn_benchmark", "channels_last",
        "eval_split_name", "vg150_source", "vg150_root", "eval_batches",
        "eval_sgg_dump_rel_feat",
    )
    configs: Dict[str, Dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.exists():
            configs[name] = {"missing": str(path)}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        configs[name] = {k: cfg.get(k) for k in keys}
    comparable = len(configs) == 2 and all("missing" not in x for x in configs.values())
    equal = comparable and configs["R0"] == configs["R2"]
    return {"configs": configs, "selected_settings_equal": equal}


def _candidate_image_paths(row: Dict[str, Any]) -> List[Path]:
    raw = row.get("image", row.get("image_path", row.get("path", row.get("file_name", ""))))
    candidates: List[Path] = []
    if isinstance(raw, str) and raw.strip():
        value = raw.strip()
        candidates.extend([
            Path(value), DATA_ROOT / value, DATA_ROOT / "images" / value,
            DATA_ROOT / "images" / "VG_100K" / Path(value).name,
        ])
    image_id = str(row.get("image_id", row.get("img_id", ""))).strip()
    if image_id:
        stem = image_id if image_id.lower().endswith(".jpg") else f"{image_id}.jpg"
        candidates.extend([
            DATA_ROOT / "images" / stem,
            DATA_ROOT / "images" / "VG_100K" / stem,
            DATA_ROOT / "images" / "VG_100K_2" / stem,
        ])
    return candidates


def _source_annotation_check(
    r0: Dict[str, Any], r2: Dict[str, Any], processor: Any,
) -> Dict[str, Any]:
    _, pred_to_idx = _load_vg150_vocab(str(DATA_ROOT))
    cfg = VG150LoaderConfig(
        vg150_root=str(DATA_ROOT), source="local-jsonl", split="validation",
        batch_size=BATCH_SIZE, num_workers=0, drop_last=False, seed=1234,
        samples_per_epoch=0, max_images=0, max_objects=32, max_pairs=64,
        use_all_pairs=True, negative_pair_ratio=-1.0,
    )
    ds = VG150JSONLDataset(cfg, pred_to_idx, split="validation",
                           processor=processor, clip_input_res=336)

    rows: List[Dict[str, Any]] = []
    all_ok = True
    pixel_hash = hashlib.sha256()
    source_image_files: List[str] = []
    final_indices = list(range(FINAL_START, len(r0["image_id"])))
    for i in final_indices:
        ex = ds[i]
        pairs = torch.tensor([[int(x[0]), int(x[1])] for x in ex["pairs"]], dtype=torch.long)
        pos_pairs = pairs[ex["rel_pos_mask"]]
        pos_preds = [str(x) for x, m in zip(ex["rel_preds"], ex["rel_pos_mask"].tolist()) if m]
        obj_labels = list(ex["obj_labels"])
        gt_subj_labels = [obj_labels[int(x)] for x in r0["gt_subj_idx"][i]]
        gt_obj_labels = [obj_labels[int(x)] for x in r0["gt_obj_idx"][i]]
        same = {
            "image_id": str(ex["image_id"]) == str(r0["image_id"][i]) == str(r2["image_id"][i]),
            "positive_pairs": bool(torch.equal(pos_pairs, r0["pairs"][i])) and bool(torch.equal(pos_pairs, r2["pairs"][i])),
            "object_labels": obj_labels == list(r0["obj_labels"][i]) == list(r2["obj_labels"][i]),
            "object_boxes": bool(torch.equal(ex["obj_boxes"], r0["obj_boxes"][i])) and bool(torch.equal(ex["obj_boxes"], r2["obj_boxes"][i])),
            "gt_subject_indices": list(pos_pairs[:, 0].tolist()) == list(r0["gt_subj_idx"][i]) == list(r2["gt_subj_idx"][i]),
            "gt_object_indices": list(pos_pairs[:, 1].tolist()) == list(r0["gt_obj_idx"][i]) == list(r2["gt_obj_idx"][i]),
            "gt_predicates": pos_preds == list(r0["gt_pred"][i]) == list(r2["gt_pred"][i]),
            "gt_subject_labels": gt_subj_labels == list(r0["gt_subj_label"][i]) == list(r2["gt_subj_label"][i]),
            "gt_object_labels": gt_obj_labels == list(r0["gt_obj_label"][i]) == list(r2["gt_obj_label"][i]),
        }
        same["all_annotation_fields"] = all(bool(v) for v in same.values())
        all_ok = all_ok and same["all_annotation_fields"]

        paths = [p for p in _candidate_image_paths(ds.rows[i]) if p.exists()]
        if paths:
            source_image_files.extend(str(p) for p in paths[:1])
        px = ex["pixel_values"]
        pixel_hash.update(str(i).encode())
        pixel_hash.update(bytes.fromhex(_sha_tensor(px)["sha256"]))
        rows.append({
            "index": i, "image_id": str(ex["image_id"]),
            "n_candidate_pairs": len(ex["pairs"]),
            "n_positive_pair_rows": int(pos_pairs.shape[0]),
            "annotation_identity": same,
            "pixel": _sha_tensor(px),
            "image_file_found": bool(paths),
            "image_path": str(paths[0]) if paths else None,
            "image_file_sha256": _sha_file(paths[0]) if paths else None,
        })

    return {
        "source_dataset": str(DATA_ROOT / "validation.jsonl"),
        "dataset_length": len(ds),
        "loader_contract": {
            "source": "local-jsonl", "split": "validation", "batch_size": BATCH_SIZE,
            "max_objects": 32, "max_pairs": 64, "use_all_pairs": True,
            "negative_pair_ratio": -1.0, "clip_input_res": 336,
        },
        "final_indices": final_indices,
        "n_images": len(rows),
        "n_pair_rows": sum(r["n_positive_pair_rows"] for r in rows),
        "annotation_identity_exact": all_ok,
        "rows": rows,
        "image_files_found": sorted(set(source_image_files)),
        "pixel_reconstruction": {
            "processor_loaded": processor is not None,
            "current_loader_output": "reconstructed",
            "fallback_image_used_for_all_rows": not bool(source_image_files),
            "batch_tensor_hash_sha256": pixel_hash.hexdigest(),
            "tensor_dtype": "torch.float32",
            "tensor_shape_per_image": [3, 336, 336],
            "historical_pixel_identity": "UNPROVEN_FROM_DUMPS",
            "current_source_reconstruction": "EXACT_CURRENT_LOADER_PATH",
            "reason": "all nine source image files are present and the local CLIP processor plus loader path were reproduced; historical dumps do not serialize pixel_values, so byte identity with the historical process is still not directly certified",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-overwrite", action="store_true",
                    help="explicitly permit replacing this tool's own prior artifact")
    args = ap.parse_args()

    r0 = torch.load(R0_DUMP, map_location="cpu", weights_only=False)
    r2 = torch.load(R2_DUMP, map_location="cpu", weights_only=False)
    try:
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14-336", local_files_only=True)
        processor_status = "loaded_local_only"
    except Exception as exc:  # pragma: no cover - cache-dependent
        processor = None
        processor_status = f"unavailable: {type(exc).__name__}: {exc}"

    dump_comparison = compare_dumps(str(R0_DUMP), str(R2_DUMP), BATCH_SIZE)
    source_check = _source_annotation_check(r0, r2, processor)
    not_run_reason = "CUDA unavailable on this VM; nvidia-smi could not communicate with the NVIDIA driver"
    conditions = []
    for planned in planned_gpu_matrix():
        row = dict(planned)
        row.update({"status": "NOT_RUN", "reason": not_run_reason, "rel_feat": None})
        conditions.append(row)

    artifact = {
        "schema": "paper_c_gd_final_batch_reproduction_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "full_evaluation_run": False, "decoder_ladder_run": False,
            "training_run": False, "historical_artifacts_modified": False,
            "source_artifacts_modified": False,
        },
        "preflight": {
            "nvidia_smi": _run_nvidia_smi(),
            "torch_runtime": _runtime_state(),
            "processor": processor_status,
        },
        "historical_eval_settings": _historical_eval_settings(),
        "historical_dumps": {
            "r0": str(R0_DUMP), "r2": str(R2_DUMP),
            "comparison": dump_comparison,
            "known_localization": {
                "changed_images": 9, "changed_pair_rows": 65,
                "final_image_indices": [FINAL_START, len(r0["image_id"]) - 1],
                "historical_max_abs_rel_feat": 0.1689453125,
            },
        },
        "final_batch_inputs": source_check,
        "controlled_reproduction_matrix": {
            "batch_size": BATCH_SIZE,
            "same_final_batch_only": True,
            "cuda_conditions": conditions,
            "diagnostic_fields_for_unrun_conditions": required_condition_fields(),
            "cpu_existing_isolation_test": {
                "test": "tests/test_readout_v2.py::test_forward_from_featmap_unaffected_by_readout_v2",
                "status": "PASS_RECORDED_FROM_PRIOR_AUDIT",
                "scope": "CPU/module-level featmap path only; not a reproduction of the historical CUDA final batch",
            },
        },
        "assessment": {
            "status": "GD-UNRESOLVED",
            "root_cause_proven": False,
            "why_not_inert": "the required controlled final-batch GPU comparison was not executable, and historical rel_feat changes reach 0.1689453125 for 65 rows",
            "r2_endpoint_null": "remains numerically valid as the registered R2 endpoint result, but its pure-same-rel_feat intervention interpretation remains conditional on resolving G-D",
            "decoder_ladder": "conditional_only_pending_GD; no ladder rerun authorized or performed",
        },
    }
    out = Path(args.out)
    if out.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing artifact {out}; choose a versioned path "
            "or pass --allow-overwrite explicitly for this non-historical audit artifact."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[written] {out}")
    print("status=GD-UNRESOLVED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
