#!/usr/bin/env python
"""Paper C -- full-split evaluation of one trained C0/C1 arm.

Pre-registered in docs/PAPER_C_C0_C1_PREREGISTRATION.md.

Runs the SAME evaluator configuration every existing WPRD anchor uses
(ensemble_alpha = 0.0, GT pairs, freq prior alpha 3.75, full VG150 validation),
dumps the pair logits, and computes:

  PRIMARY    WPRD (model term, cap 64, macro over cells)
  SECONDARY  R@50 / mR@50 under the checkpoint's own composition
  MECHANISM  geometry channel stats, fusion_gate stats, prior agreement

The arm's two geometry flags are set from --arm, so an arm can never be
evaluated under the other arm's contract.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openvocab_rel.models.relational_model import ProgressiveRelationalDecoder  # noqa: E402
from openvocab_rel import geometry as geometry_mod  # noqa: E402
import openvocab_rel.models.relational_model as rel_mod  # noqa: E402

COLS = ["dx", "dy", "rw", "rh", "ar1", "ar2", "a1", "a2"]
_GEOM: List[torch.Tensor] = []
_GATE_ACC: Dict[str, Any] = {"n": 0, "s": 0.0, "s2": 0.0, "min": None, "max": None,
                             "sub": [], "pair_means": []}
_FLAGS: Dict[str, Any] = {}
_orig_fp = ProgressiveRelationalDecoder.forward_pairs
_orig_geom = geometry_mod.geom_feats_torch


def _patched_fp(self, visual_tokens, sub_feat, obj_feat, geom_feat_raw):
    if not _FLAGS.get("_seen"):
        _FLAGS["_seen"] = True
        _FLAGS["decoder.geom_fourier_scale"] = float(getattr(self, "geom_fourier_scale", float("nan")))
        _FLAGS["decoder.img_res"] = int(getattr(self, "img_res", -1))
        gb = getattr(self, "geom_B", None)
        if gb is not None:
            _FLAGS["geom_B_std"] = float(gb.detach().float().std())
    out = _orig_fp(self, visual_tokens, sub_feat, obj_feat, geom_feat_raw)
    g = getattr(self, "last_vector_gate", None)
    if isinstance(g, torch.Tensor) and g.numel() > 0:
        f = g.detach().float().cpu()
        a = _GATE_ACC
        a["n"] += f.numel(); a["s"] += float(f.sum()); a["s2"] += float((f * f).sum())
        mn, mx = float(f.min()), float(f.max())
        a["min"] = mn if a["min"] is None else min(a["min"], mn)
        a["max"] = mx if a["max"] is None else max(a["max"], mx)
        a["sub"].append(f.flatten()[::503].clone())        # deterministic thinning
        a["pair_means"].append(f.mean(dim=-1).clone())
    return out


def _patched_geom(b1, b2):
    feat = _orig_geom(b1, b2)
    if len(_GEOM) < 2000:
        _GEOM.append(feat.detach().float().cpu())
    return feat


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["C0", "C1", "C1a", "C1b"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--fourier_scale", type=float, default=0.01)
    ap.add_argument("--eval_batches", type=int, default=0)   # 0 = full split
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=3.75)
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    args = ap.parse_args()

    pixel = args.arm in ("C1", "C1a")
    scale = args.fourier_scale if args.arm in ("C1", "C1b") else 1.0
    out_dir = Path(args.out_dir or f"runs/eval_{args.arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = out_dir / "pair_logits.pt"

    ProgressiveRelationalDecoder.forward_pairs = _patched_fp
    geometry_mod.geom_feats_torch = _patched_geom
    rel_mod.geom_feats_torch = _patched_geom
    from openvocab_rel import train as train_mod

    argv = [
        "--stage", "3", "--gpu_preset", "l4_24gb",
        "--eval_only", "true", "--epochs", "0", "--resume", "true",
        "--resume_from", args.ckpt,
        "--vg150_root", "datasets_vg150_clean",
        "--vg150_enabled", "true", "--vg150_source", "local-jsonl",
        "--device", "cuda", "--batch_size", str(args.batch_size),
        "--num_workers", "4", "--clip_input_res", "336",
        "--eval_batches", str(args.eval_batches),
        "--eval_fast_mode", "false",
        "--explicit_spoa_enabled", "false",
        "--text_conditioned_projection_enabled", "false",
        "--relationness_enabled", "false",
        "--eval_sgg_use_relationness", "false",
        "--eval_sgg_predicate_score_mode", "ensemble",
        "--eval_sgg_predicate_ensemble_alpha", "0.0",
        "--adaptive_calibration_enabled", "true",
        "--bayes_calibration_weight", "0.0",
        "--freq_bias_enabled", "true",
        "--freq_bias_path", args.prior,
        "--freq_bias_alpha", str(args.alpha), "--freq_bias_smoothing", "1.0",
        "--eval_sgg_use_gt_pairs", "true",
        "--eval_sgg_grounding_dino_enabled", "false",
        "--eval_sgg_dump_pair_logits_path", str(dump),
        "--eval_sgg_dump_rel_feat", "true",
        "--geom_input_pixel_space", "true" if pixel else "false",
        "--geom_fourier_scale", str(scale),
        "--run_name", f"eval_{args.arm}", "--out_dir", str(out_dir),
        "--save_metrics_json", str(out_dir / "metrics.jsonl"),
    ]
    try:
        train_mod.main(argv)
    except SystemExit as exc:
        if int(exc.code or 0) != 0:
            raise
    finally:
        ProgressiveRelationalDecoder.forward_pairs = _orig_fp
        geometry_mod.geom_feats_torch = _orig_geom
        rel_mod.geom_feats_torch = _orig_geom

    # ---------------- offline: WPRD and deployed metrics -------------------
    MECH = _load("cprime_mechanism"); WPD = _load("within_pair_discrimination")
    B = MECH.Mech(str(dump), args.prior, "raw50")
    Gs = WPD.Groups(B)
    mt = B.fixed_ensemble(0.0)
    r = WPD.wprd(Gs, mt[B.gt_row], args.cap)
    comp = B.score(0.0, args.alpha, mt)
    m = B.metrics(comp)
    pred = B.predict(comp)
    prior_pred = B.col_to_class[B.prior[B.gt_row].argmax(-1)]
    prior_wprd = WPD.wprd(Gs, B.prior[B.gt_row], args.cap)["wprd_macro"]

    G = torch.cat(_GEOM, 0) if _GEOM else torch.zeros(0, 8)
    chan = {c: {"std": float(G[:, k].std()),
                "n_distinct": int(torch.unique(torch.round(G[:, k] * 1e6)).numel()),
                "min": float(G[:, k].min()), "max": float(G[:, k].max())}
            for k, c in enumerate(COLS)} if G.numel() else {}
    n_const = sum(1 for c in COLS if chan.get(c, {}).get("n_distinct") == 1)

    a = _GATE_ACC
    sub = torch.cat(a["sub"]) if a["sub"] else torch.zeros(0)
    pm = torch.cat(a["pair_means"]) if a["pair_means"] else torch.zeros(0)
    mean = a["s"] / max(1, a["n"])
    var = max(0.0, a["s2"] / max(1, a["n"]) - mean * mean)
    gate = {"n_elements": a["n"], "mean": mean, "std": var ** 0.5,
            "min": a["min"], "max": a["max"],
            "q25": float(sub.quantile(0.25)) if sub.numel() else None,
            "median": float(sub.quantile(0.5)) if sub.numel() else None,
            "q75": float(sub.quantile(0.75)) if sub.numel() else None,
            "per_pair_mean_std": float(pm.std()) if pm.numel() else None,
            "per_pair_mean_min": float(pm.min()) if pm.numel() else None,
            "per_pair_mean_max": float(pm.max()) if pm.numel() else None,
            "frac_in_0.45_0.55": float(((sub > 0.45) & (sub < 0.55)).float().mean())
            if sub.numel() else None}

    res = {
        "tool": "c0_c1_evaluate",
        "prereg": "docs/PAPER_C_C0_C1_PREREGISTRATION.md",
        "arm": args.arm, "ckpt": args.ckpt,
        "contract": {"geom_input_pixel_space": pixel, "geom_fourier_scale": scale},
        "observed_in_live_model": _FLAGS,
        "population": {"images": B.n_images, "pairs": int(B.prior.shape[0]),
                       "gt_rows": B.n_gt, "cells": r["n_cells"]},
        "wprd_macro": r["wprd_macro"], "wprd_weighted": r["wprd_weighted"],
        "frac_cells_above_half": r["frac_cells_above_half"],
        "cell_values": r["_vals"],
        "R": m["R"], "mR": m["mR"], "head_mR": m["head_mR"],
        "body_mR": m["body_mR"], "tail_mR": m["tail_mR"],
        "top1_equals_prior_argmax": float((pred == prior_pred).float().mean()),
        "model_term_std": float(mt.std()),
        "prior_control_wprd": prior_wprd,
        "geom_channels": chan, "n_constant_channels": n_const,
        "fusion_gate": gate,
        "dump": str(dump),
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print(f"  {args.arm}   pixel_space={pixel}  fourier_scale={scale}")
    print("=" * 96)
    print(f"  population {B.n_images:,} images / {B.prior.shape[0]:,} pairs / "
          f"{r['n_cells']:,} cells")
    print(f"  WPRD {r['wprd_macro']:.4f}   weighted {r['wprd_weighted']:.4f}   "
          f"R@50 {m['R']:.4f}   mR@50 {m['mR']:.4f}")
    print(f"  prior control {prior_wprd:.6f}   pred==prior argmax "
          f"{res['top1_equals_prior_argmax']:.4f}   model_term_std {res['model_term_std']:.4f}")
    print(f"  geometry: {n_const} of 8 constant   gate mean {gate['mean']:.5f}  "
          f"std {gate['std']:.5f}  per-pair std {gate['per_pair_mean_std']:.6f}")
    print(f"  wrote {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
