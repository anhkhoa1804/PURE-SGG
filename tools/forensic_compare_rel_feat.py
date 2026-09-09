#!/usr/bin/env python3
"""Read-only forensic comparison for two relational-feature evaluations.

This tool does not construct a model, run inference, or modify either input.
It combines the repository's exact population verifier with per-image tensor
comparisons, so a large global max cannot hide whether a discrepancy is local
to one batch or present throughout the split.  Checkpoint comparison uses
streaming SHA-256 digests of serialized model/CLIP tensors and reports the
exact differing key set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    from tools.verify_population_identity import verify_population_identity
except ModuleNotFoundError:  # direct ``python tools/forensic_compare_rel_feat.py``
    from verify_population_identity import verify_population_identity


def _tensor_digest(tensor: torch.Tensor) -> Dict[str, Any]:
    x = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    flat = x.view(-1)
    step = max(1, min(int(flat.numel()), 1 << 20))
    for start in range(0, int(flat.numel()), step):
        h.update(flat[start:start + step].numpy().tobytes())
    return {"sha256": h.hexdigest(), "shape": list(x.shape),
            "dtype": str(x.dtype), "numel": int(x.numel())}


def compare_checkpoint_tensors(path_a: str, path_b: str) -> Dict[str, Any]:
    """Compare model/CLIP tensors without writing or re-saving checkpoints."""
    def load(path: str) -> Dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:  # older torch without mmap
            return torch.load(path, map_location="cpu", weights_only=False)

    out: Dict[str, Any] = {"checkpoint_a": path_a, "checkpoint_b": path_b,
                           "sections": {}}
    a, b = load(path_a), load(path_b)
    for section in ("model", "clip"):
        sa, sb = a.get(section, {}), b.get(section, {})
        keys_a, keys_b = set(sa), set(sb)
        differing: List[str] = []
        for key in sorted(keys_a & keys_b):
            va, vb = sa[key], sb[key]
            if not isinstance(va, torch.Tensor) or not isinstance(vb, torch.Tensor):
                if repr(va) != repr(vb):
                    differing.append(key)
                continue
            if _tensor_digest(va) != _tensor_digest(vb):
                differing.append(key)
        out["sections"][section] = {
            "keys_a": len(keys_a), "keys_b": len(keys_b),
            "only_a": sorted(keys_a - keys_b),
            "only_b": sorted(keys_b - keys_a),
            "different_common_tensors": differing,
        }
    return out


def _field_summary(a: List[torch.Tensor], b: List[torch.Tensor]) -> Dict[str, Any]:
    changed_entries = 0
    max_abs = 0.0
    mean_abs_sum = 0.0
    n_values = 0
    for xa, xb in zip(a, b):
        if not isinstance(xa, torch.Tensor) or not isinstance(xb, torch.Tensor):
            continue
        if tuple(xa.shape) != tuple(xb.shape):
            return {"shape_mismatch": True}
        d = (xa.float() - xb.float()).abs()
        changed_entries += int(bool(d.numel() and (d > 0).any()))
        if d.numel():
            max_abs = max(max_abs, float(d.max()))
            mean_abs_sum += float(d.sum())
            n_values += int(d.numel())
    return {"entries": len(a), "changed_entries": changed_entries,
            "max_abs": max_abs,
            "mean_abs": mean_abs_sum / max(1, n_values)}


def compare_dumps(path_a: str, path_b: str, batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Compare aligned dump fields and report where nonzero differences occur."""
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    identity = verify_population_identity(path_a, path_b)
    out: Dict[str, Any] = {"dump_a": path_a, "dump_b": path_b,
                           "population_identity": identity,
                           "fields": {}, "changed_images": []}

    for field in ("rel_feat", "text_logits", "model_logits", "cls_logits", "prior_rows"):
        if isinstance(a.get(field), list) and isinstance(b.get(field), list):
            out["fields"][field] = _field_summary(a[field], b[field])

    ra, rb = a.get("rel_feat"), b.get("rel_feat")
    if isinstance(ra, list) and isinstance(rb, list) and len(ra) == len(rb):
        for index, (xa, xb) in enumerate(zip(ra, rb)):
            if not isinstance(xa, torch.Tensor) or not isinstance(xb, torch.Tensor):
                continue
            if tuple(xa.shape) != tuple(xb.shape):
                out["changed_images"].append({"index": index, "image_id": str(a["image_id"][index]),
                                               "shape_a": list(xa.shape), "shape_b": list(xb.shape)})
                continue
            d = (xa.float() - xb.float()).abs()
            if d.numel() and bool((d > 0).any()):
                item: Dict[str, Any] = {
                    "index": index, "image_id": str(a["image_id"][index]),
                    "n_rows": int(xa.shape[0]), "max_abs": float(d.max()),
                    "mean_abs": float(d.mean()),
                    "changed_rows": int((d.max(dim=1).values > 0).sum()),
                }
                if batch_size is not None and int(batch_size) > 0:
                    item["batch_index"] = index // int(batch_size)
                out["changed_images"].append(item)
    out["summary"] = {
        "n_images": len(a.get("image_id", [])),
        "n_changed_images": len(out["changed_images"]),
        "all_changes_in_final_batch": bool(
            out["changed_images"] and batch_size and
            all(x.get("batch_index") == (len(a.get("image_id", [])) - 1) // int(batch_size)
                for x in out["changed_images"])
        ),
    }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dumps")
    d.add_argument("dump_a")
    d.add_argument("dump_b")
    d.add_argument("--batch-size", type=int, default=0)
    d.add_argument("--out", default="")
    c = sub.add_parser("checkpoints")
    c.add_argument("checkpoint_a")
    c.add_argument("checkpoint_b")
    c.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.command == "dumps":
        result = compare_dumps(args.dump_a, args.dump_b, args.batch_size or None)
    else:
        result = compare_checkpoint_tensors(args.checkpoint_a, args.checkpoint_b)
    rendered = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"[written] {args.out}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
