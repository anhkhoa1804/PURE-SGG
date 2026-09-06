#!/usr/bin/env python
"""p70 analysis -- WPRD and deployed metrics for each geometry-ablation arm.

Pre-registered in docs/GEOMETRY_CAUSAL_ABLATION_PREREGISTRATION.md.
CPU only. Reads runs/p70_geometry_causal_ablation/{pair_logits_armA.pt,
arm_rel_feats.pt}. No GPU, no historical artifact touched.

Every arm's deployed model term is rebuilt from its rel_feat by the identity the
evaluator itself uses at ensemble_alpha = 0.0:

    model_term = _normalize_eval_logits( normalize(rel_feat) @ normalize(pred_emb).T )[fg]

Arm A is gated against the cache's own stored model term, so the reconstruction
is proved rather than assumed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MECH = _load("cprime_mechanism")
WPD = _load("within_pair_discrimination")
STRAT = _load("wprd_stratified")
Mech = MECH.Mech

P33_VAL_MODEL_TERM_WPRD = 0.5542   # p33 anchor, reproduced exactly this session


def _log(m: str = "") -> None:
    print(m, flush=True)


def _model_term(rel_feats: List[torch.Tensor], pred_emb: torch.Tensor,
                fg: List[int], tt: float) -> torch.Tensor:
    pe = F.normalize(pred_emb.float(), dim=-1)
    out = []
    for rf in rel_feats:
        t = F.normalize(rf.float(), dim=-1) @ pe.t()
        out.append(Mech._norm(t)[:, fg])
    return torch.cat(out, 0) / tt


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/p70_geometry_causal_ablation")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=3.75)
    args = ap.parse_args(argv)
    run = Path(args.run)
    out_path = Path(args.out) if args.out else run / "ablation_wprd.json"

    _log("=" * 108)
    _log("p70 GEOMETRY CAUSAL ABLATION -- WPRD per arm. CPU only, NO GPU")
    _log("=" * 108)

    B = Mech(str(run / "pair_logits_armA.pt"), args.prior, "raw50")
    Gs = WPD.Groups(B)
    arms_raw = torch.load(run / "arm_rel_feats.pt", map_location="cpu", weights_only=False)
    pe = B.meta["pred_emb"]
    tt = float(B.meta.get("text_temperature", 1.0)) or 1.0
    gates: List[Dict[str, Any]] = []

    _log(f"  images {B.n_images:,}  pairs {B.prior.shape[0]:,}  GT rows {B.n_gt:,}  "
         f"groups {Gs.G:,}")
    g3 = (B.n_images == 10401) and (B.prior.shape[0] == 132556)
    gates.append({"gate": "G3 same population as the p36/p60/p69 anchors",
                  "pass": bool(g3),
                  "detail": f"{B.n_images} images / {B.prior.shape[0]} pairs"})

    mt: Dict[str, torch.Tensor] = {}
    for name, rfs in arms_raw.items():
        mt[name] = _model_term(rfs, pe, B.fg_cols, tt)

    stored = B.fixed_ensemble(0.0)
    e = float((mt["A_full"] - stored).abs().max())
    # The dump stores rel_feat in fp16 while `stored` was built from the fp32
    # tensor, and Bench._norm divides by the per-row std (~0.03 for cosine
    # similarities), amplifying that storage error by ~30x. The tolerance is
    # therefore expressed RELATIVE to the model term's own scale. This error is
    # common-mode across arms -- identical rel_feat precision for all six -- so
    # it cancels out of every paired delta, which is where the science lives.
    rel = e / float(stored.std())
    gates.append({"gate": "G1b arm A reconstructs the cache's own model term",
                  "pass": bool(rel < 0.02),
                  "detail": f"max abs diff {e:.3e} = {rel:.4f} of the model term's std"})
    _log(f"  G1b model-term reconstruction  max|A - stored| = {e:.3e} "
         f"({rel:.4f} of its std)  {'PASS' if rel < 0.02 else 'FAIL'}")

    diffs = {}
    for name in mt:
        if name == "A_full":
            continue
        diffs[name] = float((mt[name] - mt["A_full"]).abs().max())
    g6 = all(v > 1e-4 for v in diffs.values())
    gates.append({"gate": "G6 every ablation arm actually differs from A",
                  "pass": bool(g6),
                  "detail": {k: round(v, 6) for k, v in diffs.items()}})

    g5 = all(bool(torch.isfinite(v).all()) for v in mt.values())
    gates.append({"gate": "G5 all arms finite", "pass": bool(g5),
                  "detail": f"{len(mt)} arms"})

    res: Dict[str, Any] = {
        "tool": "geometry_ablation_wprd",
        "prereg": "docs/GEOMETRY_CAUSAL_ABLATION_PREREGISTRATION.md",
        "run": str(run), "cap": args.cap, "boot": args.boot,
        "freq_bias_alpha": args.alpha, "arms": {}, "paired": {}}

    _log(f"\n{'-'*108}")
    _log(f"  {'arm':>20} {'WPRD':>8} {'weighted':>9} {'cells>.5':>9} "
         f"{'R':>7} {'mR':>7} {'agree@1':>8} {'prior@1':>8}")
    _log(f"{'-'*108}")
    cellvals: Dict[str, torch.Tensor] = {}
    predA = None
    prior_pred = B.col_to_class[B.prior[B.gt_row].argmax(-1)]
    for name in mt:
        s_gt = mt[name][B.gt_row]
        r = WPD.wprd(Gs, s_gt, args.cap)
        cellvals[name] = torch.tensor(r["_vals"], dtype=torch.float64)
        comp = B.score(0.0, args.alpha, mt[name])
        m = B.metrics(comp)
        pred = B.predict(comp)
        if predA is None:
            predA = pred
        agree = float((pred == predA).float().mean())
        prior_agree = float((pred == prior_pred).float().mean())
        res["arms"][name] = {
            "wprd_macro": r["wprd_macro"], "wprd_weighted": r["wprd_weighted"],
            "n_cells": r["n_cells"],
            "frac_cells_above_half": r["frac_cells_above_half"],
            "R": m["R"], "mR": m["mR"], "head_mR": m["head_mR"],
            "body_mR": m["body_mR"], "tail_mR": m["tail_mR"],
            "top1_agreement_with_A": agree,
            "top1_equals_prior_argmax": prior_agree,
            "model_term_std": float(mt[name].std()),
        }
        _log(f"  {name:>20} {r['wprd_macro']:8.4f} {r['wprd_weighted']:9.4f} "
             f"{r['frac_cells_above_half']*100:8.1f}% {m['R']:7.4f} {m['mR']:7.4f} "
             f"{agree:8.4f} {prior_agree:8.4f}")

    g2 = abs(res["arms"]["A_full"]["wprd_macro"] - P33_VAL_MODEL_TERM_WPRD) < 0.005
    gates.insert(0, {"gate": f"G2 arm A reproduces p33's {P33_VAL_MODEL_TERM_WPRD} +-0.005",
                     "pass": bool(g2),
                     "detail": f"{res['arms']['A_full']['wprd_macro']:.4f} "
                               f"(delta {res['arms']['A_full']['wprd_macro']-P33_VAL_MODEL_TERM_WPRD:+.4f})"})

    prior_wprd = WPD.wprd(Gs, B.prior[B.gt_row], args.cap)["wprd_macro"]
    gates.append({"gate": "G4 prior control exactly 0.5000",
                  "pass": bool(abs(prior_wprd - 0.5) < 1e-6),
                  "detail": f"{prior_wprd:.6f}"})

    _log(f"\n{'-'*108}")
    _log("  PAIRED deltas vs A_full (bootstrap over the SAME cells, both arms)")
    _log(f"{'-'*108}")
    _log(f"  {'arm':>20} {'dWPRD':>9} {'95% CI':>22} {'P(d<0)':>8} {'dR':>9} {'dmR':>9}")
    vA = cellvals["A_full"]
    g = torch.Generator().manual_seed(11)
    idx = torch.randint(len(vA), (args.boot, len(vA)), generator=g)
    for name in mt:
        if name == "A_full":
            continue
        d = cellvals[name] - vA
        bs = d[idx].mean(dim=1)
        lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975],
                                                 dtype=torch.float64)).tolist()
        res["paired"][name] = {
            "delta_wprd": float(d.mean()), "ci95": [lo, hi],
            "p_delta_negative": float((bs < 0).double().mean()),
            "delta_R": res["arms"][name]["R"] - res["arms"]["A_full"]["R"],
            "delta_mR": res["arms"][name]["mR"] - res["arms"]["A_full"]["mR"],
        }
        p = res["paired"][name]
        _log(f"  {name:>20} {p['delta_wprd']:+9.4f} [{lo:+.4f},{hi:+.4f}] "
             f"{p['p_delta_negative']:8.3f} {p['delta_R']:+9.4f} {p['delta_mR']:+9.4f}")

    dD = res["paired"]["D_no_geom"]["delta_wprd"]
    dB = res["paired"]["B_no_fusion_geom"]["delta_wprd"]
    dC = res["paired"]["C_no_edge_geom"]["delta_wprd"]
    primary = ("LIVE" if dD <= -0.010 else
               "WEAK" if dD <= -0.003 else
               "INVERTED" if dD >= 0.003 else "INERT")
    allpass = all(x["pass"] for x in gates)
    res["gates"] = gates
    res["verdict"] = {
        "delta_D": dD, "delta_B": dB, "delta_C": dC,
        "additivity_residual": dD - (dB + dC),
        "primary": primary, "gates_all_pass": bool(allpass)}

    _log(f"\n  PRIMARY   delta_D = WPRD(D_no_geom) - WPRD(A_full) = {dD:+.4f}  -> {primary}")
    _log(f"  PATHS     fusion-only removal {dB:+.4f}   edge-only removal {dC:+.4f}   "
         f"additivity residual {dD-(dB+dC):+.4f}")
    _log("\n  gates: " + "  ".join(
        f"{x['gate'].split()[0]}={'PASS' if x['pass'] else 'FAIL'}" for x in gates))
    if not allpass:
        _log("  *** A GATE FAILED. By the registration, NO NUMBER HERE IS REPORTABLE. ***")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
