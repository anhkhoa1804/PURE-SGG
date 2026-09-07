#!/usr/bin/env python
"""Paper C -- the paired C1 vs C0 comparison and the registered decision rule.

Pre-registered in docs/PAPER_C_C0_C1_PREREGISTRATION.md. CPU only.

Reads two runs/eval_*/result.json produced by tools/c0_c1_evaluate.py and
applies the thresholds fixed in advance. Because both arms score the SAME cells
(same split, same GT rows, same Groups construction), the primary test is a
PAIRED bootstrap over cells, not a comparison of independent CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def _log(m: str = "") -> None:
    print(m, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", default="runs/eval_C0/result.json")
    ap.add_argument("--c1", default="runs/eval_C1/result.json")
    ap.add_argument("--label_c0", default="C0")
    ap.add_argument("--label_c1", default="C1")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/paper_c_c0_c1_comparison.json")
    args = ap.parse_args()

    A = json.loads(Path(args.c0).read_text())
    B = json.loads(Path(args.c1).read_text())
    vA = torch.tensor(A["cell_values"], dtype=torch.float64)
    vB = torch.tensor(B["cell_values"], dtype=torch.float64)

    gates: List[Dict[str, Any]] = []
    gates.append({"gate": "D1 both arms scored the same population",
                  "pass": bool(A["population"] == B["population"]),
                  "detail": f"{A['population']} vs {B['population']}"})
    gates.append({"gate": "D2 cells are paired (identical count)",
                  "pass": bool(vA.numel() == vB.numel()),
                  "detail": f"{vA.numel()} vs {vB.numel()}"})
    gates.append({"gate": "D3 prior control reads exactly 0.5000 in both arms",
                  "pass": bool(abs(A["prior_control_wprd"] - 0.5) < 1e-6
                               and abs(B["prior_control_wprd"] - 0.5) < 1e-6),
                  "detail": f"{A['prior_control_wprd']:.6f} / {B['prior_control_wprd']:.6f}"})
    gates.append({"gate": "D4 each arm ran under its own registered contract",
                  "pass": bool(A["contract"]["geom_input_pixel_space"] is False
                               and A["contract"]["geom_fourier_scale"] == 1.0
                               and B["contract"] != A["contract"]),
                  "detail": f"{A['contract']} vs {B['contract']}"})
    if not (gates[1]["pass"] and gates[0]["pass"]):
        _log("*** PAIRING GATE FAILED -- the arms are not comparable. ***")
        for g in gates:
            _log(f"  {'PASS' if g['pass'] else 'FAIL'}  {g['gate']}  |  {g['detail']}")
        return 1

    d = vB - vA
    g = torch.Generator().manual_seed(11)
    idx = torch.randint(len(d), (args.boot, len(d)), generator=g)
    bs = d[idx].mean(dim=1)
    lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    delta = float(d.mean())
    p_neg = float((bs < 0).double().mean())

    dR = B["R"] - A["R"]
    dmR = B["mR"] - A["mR"]
    d_prior = B["top1_equals_prior_argmax"] - A["top1_equals_prior_argmax"]

    primary = ("MATERIAL" if delta >= 0.010 else
               "WEAK-POSITIVE" if delta >= 0.005 else
               "NULL" if delta > -0.005 else "HARMFUL")
    ci_excludes_zero = bool(lo > 0.0 or hi < 0.0)
    # Registered: a WPRD-flat / recall-positive result is NOT a Paper C success.
    calibration_only = bool(abs(delta) < 0.005 and (dR > 0.005 or dmR > 0.005))

    res = {
        "tool": "c0_c1_compare",
        "prereg": "docs/PAPER_C_C0_C1_PREREGISTRATION.md",
        "arms": {args.label_c0: args.c0, args.label_c1: args.c1},
        "wprd": {args.label_c0: A["wprd_macro"], args.label_c1: B["wprd_macro"]},
        "delta_wprd": delta, "ci95": [lo, hi], "p_delta_negative": p_neg,
        "n_cells": int(vA.numel()),
        "secondary": {"R@50": {args.label_c0: A["R"], args.label_c1: B["R"], "delta": dR},
                      "mR@50": {args.label_c0: A["mR"], args.label_c1: B["mR"], "delta": dmR}},
        "calibration": {
            "top1_equals_prior_argmax": {args.label_c0: A["top1_equals_prior_argmax"],
                                         args.label_c1: B["top1_equals_prior_argmax"],
                                         "delta": d_prior},
            "model_term_std": {args.label_c0: A["model_term_std"],
                               args.label_c1: B["model_term_std"]},
        },
        "mechanism": {
            "n_constant_geometry_channels": {args.label_c0: A["n_constant_channels"],
                                             args.label_c1: B["n_constant_channels"]},
            "fusion_gate": {args.label_c0: A["fusion_gate"], args.label_c1: B["fusion_gate"]},
            "geom_channels": {args.label_c0: A["geom_channels"], args.label_c1: B["geom_channels"]},
        },
        "gates": gates,
        "verdict": {
            "primary": primary,
            "ci_excludes_zero": ci_excludes_zero,
            "calibration_only_effect": calibration_only,
            "meets_registered_success_threshold": bool(delta >= 0.010 and ci_excludes_zero
                                                       and not calibration_only),
            "gates_all_pass": bool(all(x["pass"] for x in gates)),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    _log("=" * 96)
    _log(f"  PAPER C -- {args.label_c1} vs {args.label_c0}   "
         f"({vA.numel():,} paired cells, {A['population']['images']:,} images)")
    _log("=" * 96)
    _log(f"  PRIMARY   WPRD  {args.label_c0} {A['wprd_macro']:.4f}   "
         f"{args.label_c1} {B['wprd_macro']:.4f}")
    _log(f"            delta {delta:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
         f"P(d<0) {p_neg:.3f}   -> {primary}")
    _log(f"  SECONDARY R@50  {A['R']:.4f} -> {B['R']:.4f}  ({dR:+.4f})")
    _log(f"            mR@50 {A['mR']:.4f} -> {B['mR']:.4f}  ({dmR:+.4f})")
    _log(f"  CALIBRAT  pred==prior argmax {A['top1_equals_prior_argmax']:.4f} -> "
         f"{B['top1_equals_prior_argmax']:.4f}  ({d_prior:+.4f})")
    _log(f"  MECHANISM constant geom channels "
         f"{A['n_constant_channels']} -> {B['n_constant_channels']}")
    ga, gb = A["fusion_gate"], B["fusion_gate"]
    _log(f"            gate mean {ga['mean']:.5f} -> {gb['mean']:.5f}   "
         f"std {ga['std']:.5f} -> {gb['std']:.5f}   "
         f"per-pair std {ga['per_pair_mean_std']:.6f} -> {gb['per_pair_mean_std']:.6f}")
    _log(f"            gate range [{ga['min']:.4f},{ga['max']:.4f}] -> "
         f"[{gb['min']:.4f},{gb['max']:.4f}]")
    _log("\n  gates: " + "  ".join(
        f"{x['gate'].split()[0]}={'PASS' if x['pass'] else 'FAIL'}" for x in gates))
    v = res["verdict"]
    _log(f"\n  REGISTERED SUCCESS THRESHOLD (delta >= +0.010, CI excludes 0, not "
         f"calibration-only): {'MET' if v['meets_registered_success_threshold'] else 'NOT MET'}")
    if calibration_only:
        _log("  *** WPRD flat while recall moved: classified as a "
             "CALIBRATION/COMPOSED-METRIC EFFECT, not a Paper C success. ***")
    _log(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
