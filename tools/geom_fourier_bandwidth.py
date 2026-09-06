#!/usr/bin/env python
"""p72 -- does PURE's frozen random-Fourier geometry encoder destroy the
geometry information that p69 showed exists?

Pre-registered in docs/GEOM_FOURIER_BANDWIDTH_PREREGISTRATION.md.

CPU only. Reads the p36 cache and the historical checkpoint's geom_B (read
only). Reuses p60/p69's estimator, folds and reporting path verbatim, so every
number here is comparable to p60/p69 by construction.

The question. p69 established that the six geometry channels PURE cannot see
carry +0.0341 WPRD. C1a would restore them. But they would be delivered through
`fourier = [sin, cos](2*pi * x @ geom_B)` with geom_B frozen at std 9.91 --
a phase rate of ~41 rad per unit of input, i.e. a full 2*pi cycle every 0.155
of dx. If that map is information-destroying, C1a alone cannot work and C1b is
mandatory; if it is not, C1a is the right first intervention and C1b is a
distraction. This tool measures which.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
GEO = _load("wprd_geometry_control")
OBJ = _load("objective_ablation_relfeat")
Mech = MECH.Mech
N_FOLDS = 5

from openvocab_rel.geometry import geom_feats_torch  # noqa: E402


def _log(m: str = "") -> None:
    print(m, flush=True)


def _geom8(B: Mech, normalise: bool) -> torch.Tensor:
    """The 8-vector geom_feats_torch produces, per GT row.

    normalise=False -> boxes in their native pixel units: the domain
    geom_feats_torch was written for and clamp_min(1.0) never binds. This is the
    C1a ("units fixed") input contract.

    normalise=True  -> boxes divided by their own per-image extent so every
    width/height is <= 1: clamp_min(1.0) binds on all four, exactly reproducing
    production's 6-of-8-constant degeneracy (p68). Proxy for the current
    contract; identical in kind, differing from PURE's fixed /336 only in the
    units of the surviving dx, dy (registered caveat, same as p69's).
    """
    feats: List[torch.Tensor] = []
    for i in range(B.n_images):
        boxes = B.meta["obj_boxes"][i].float()
        pairs = B.meta["pairs"][i]
        if boxes.numel() == 0 or len(pairs) == 0:
            continue
        b = boxes
        if normalise:
            W = float(boxes[:, 2].max() - boxes[:, 0].min()) or 1.0
            H = float(boxes[:, 3].max() - boxes[:, 1].min()) or 1.0
            b = boxes / torch.tensor([W, H, W, H], dtype=torch.float32)
        s = b[pairs[:, 0].long()]
        o = b[pairs[:, 1].long()]
        feats.append(geom_feats_torch(s, o))
    return torch.cat(feats, 0)[B.gt_row]


def _fourier(x: torch.Tensor, Bmat: torch.Tensor) -> torch.Tensor:
    """Exactly relational_model.py:441-447, with geom_B scaled."""
    xc = torch.nan_to_num(x[..., :8], nan=0.0, posinf=1.0, neginf=-1.0)
    xc = torch.clamp(xc, -10.0, 10.0)
    proj = (2.0 * math.pi * xc) @ Bmat
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _std_bias(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(0, keepdim=True)
    sd = x.std(0, keepdim=True).clamp_min(1e-6)
    return torch.cat([(x - mu) / sd, torch.ones(x.shape[0], 1)], 1)


def _kernel_profile(x: torch.Tensor, Bmat: torch.Tensor, n: int = 20000,
                    seed: int = 3) -> Dict[str, Any]:
    """Empirical cosine similarity of Fourier features vs input distance."""
    g = torch.Generator().manual_seed(seed)
    idx1 = torch.randint(x.shape[0], (n,), generator=g)
    idx2 = torch.randint(x.shape[0], (n,), generator=g)
    a, b = x[idx1], x[idx2]
    d = (a - b).norm(dim=-1)
    fa = torch.nn.functional.normalize(_fourier(a, Bmat), dim=-1)
    fb = torch.nn.functional.normalize(_fourier(b, Bmat), dim=-1)
    cs = (fa * fb).sum(-1)
    edges = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if int(m.sum()) < 30:
            continue
        out.append({"dist_lo": lo, "dist_hi": hi, "n": int(m.sum()),
                    "cos_mean": float(cs[m].mean()), "cos_std": float(cs[m].std())})
    return {"buckets": out,
            "input_pair_dist_median": float(d.median()),
            "overall_cos_mean": float(cs.mean())}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--ckpt", default="checkpoints/demo_best/pure_best_adapt_light_mR50.pt")
    ap.add_argument("--out", default="runs/p72_geom_fourier_bandwidth/est.json")
    ap.add_argument("--scales", default="1.0,0.25,0.1,0.05")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args(argv)

    _log("=" * 108)
    _log("p72 GEOMETRY FOURIER BANDWIDTH, matched estimator -- CPU only, NO GPU")
    _log("=" * 108)

    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    fold = OBJ.folds_of(B, 0)
    y = B.gt_y
    C = B.n_classes
    gates: List[Dict[str, Any]] = []

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    Bmat = ck["model"]["decoder.geom_B"].float()
    del ck
    _log(f"  geom_B {tuple(Bmat.shape)}  std {Bmat.std():.4f}  |max| {Bmat.abs().max():.3f}")

    G_fixed = _geom8(B, normalise=False)
    G_prod = _geom8(B, normalise=True)
    assert G_fixed.shape[0] == B.n_gt and G_prod.shape[0] == B.n_gt

    cols = ["dx", "dy", "rw", "rh", "ar1", "ar2", "a1", "a2"]
    chan = {}
    for nm, Gm in (("fixed", G_fixed), ("prod", G_prod)):
        chan[nm] = {c: {"min": float(Gm[:, k].min()), "max": float(Gm[:, k].max()),
                        "mean": float(Gm[:, k].mean()), "std": float(Gm[:, k].std()),
                        "n_distinct": int(torch.unique(
                            torch.round(Gm[:, k] * 1e6)).numel())}
                    for k, c in enumerate(cols)}
    n_const_prod = sum(1 for c in cols if chan["prod"][c]["n_distinct"] == 1)
    n_const_fixed = sum(1 for c in cols if chan["fixed"][c]["n_distinct"] == 1)
    gates.append({"gate": "G1 prod contract reproduces p68's 6-of-8 degeneracy",
                  "pass": bool(n_const_prod == 6), "detail": f"{n_const_prod} constant"})
    gates.append({"gate": "G2 fixed contract has 0 constant channels",
                  "pass": bool(n_const_fixed == 0), "detail": f"{n_const_fixed} constant"})

    scales = [float(s) for s in args.scales.split(",") if s.strip()]
    feats: Dict[str, torch.Tensor] = {
        "R_raw_fixed": _std_bias(G_fixed),
        "R_raw_prod": _std_bias(G_prod),
    }
    for s in scales:
        feats[f"F_s{s:g}_fixed"] = _std_bias(_fourier(G_fixed, Bmat * s))
    feats["F_s1_prod"] = _std_bias(_fourier(G_prod, Bmat))

    kern = {f"fixed_s{s:g}": _kernel_profile(G_fixed, Bmat * s) for s in scales}
    kern["prod_s1"] = _kernel_profile(G_prod, Bmat)

    _log(f"  rows {B.n_gt:,}   " + "  ".join(
        f"{k}{tuple(v.shape)}" for k, v in feats.items()))

    scores: Dict[str, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(0)

    def fit(name: str, X: torch.Tensor, labels: torch.Tensor) -> None:
        out = torch.zeros(X.shape[0], C)
        for f in range(N_FOLDS):
            te = fold == f
            tr = torch.nonzero(~te, as_tuple=True)[0]
            net = OBJ.mlp(X.shape[1], args.hidden, C, 0)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.l2)
            Xtr, ytr = X[tr], labels[tr]
            n, bs = Xtr.shape[0], 8192
            for _ in range(args.epochs):
                perm = torch.randperm(n, generator=gen)
                for i in range(0, n, bs):
                    ix = perm[i:i + bs]
                    opt.zero_grad()
                    torch.nn.functional.cross_entropy(net(Xtr[ix]), ytr[ix]).backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                out[te] = net(X[te])
        scores[name] = out
        _log(f"    fitted {name}")

    _log(f"\n  fitting (cross-fitted, {N_FOLDS} folds, hidden={args.hidden}, "
         f"epochs={args.epochs}, lr={args.lr}, l2={args.l2})")
    for k, v in feats.items():
        fit(k, v, y)
    ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    fit("N_shuffled", feats["R_raw_fixed"], ysh)
    scores["P_prior"] = B.prior

    g6 = all(bool(torch.isfinite(v).all()) and v.shape[0] == B.n_gt
             for v in scores.values())
    gates.append({"gate": "G5 every arm scores every row", "pass": bool(g6),
                  "detail": f"{len(scores)} arms x {B.n_gt} rows"})

    res: Dict[str, Any] = {
        "tool": "geom_fourier_bandwidth",
        "prereg": "docs/GEOM_FOURIER_BANDWIDTH_PREREGISTRATION.md",
        "geom_B": {"shape": list(Bmat.shape), "std": float(Bmat.std()),
                   "absmax": float(Bmat.abs().max())},
        "channel_stats": chan, "kernel": kern,
        "estimator": {"hidden": args.hidden, "epochs": args.epochs, "lr": args.lr,
                      "l2": args.l2, "loss": "softmax CE", "opt": "AdamW",
                      "regime": "5-fold CV on validation, salt 0"},
        "gates": gates, "arms": {}}

    _log(f"\n{'-'*104}")
    _log(f"  {'arm':>22} {'WPRD':>8} {'weighted':>9} {'95% CI':>20} "
         f"{'head':>7} {'body':>7} {'tail':>7}")
    _log(f"{'-'*104}")
    for name, sc in scores.items():
        r = WPD.wprd(Gs, sc, args.cap)
        v = torch.tensor(r["_vals"], dtype=torch.float64)
        g = torch.Generator().manual_seed(1)
        bs_ = torch.stack([v[torch.randint(len(v), (len(v),), generator=g)].mean()
                           for _ in range(args.boot)])
        lo, hi = torch.quantile(bs_, torch.tensor([0.025, 0.975],
                                                  dtype=torch.float64)).tolist()
        cc = STRAT.cells(Gs, B, sc, args.cap, 0, drop_same_instance=False)
        bb = {}
        for key in ("head-head", "body-body", "tail-tail"):
            x1, x2 = key.split("-")
            sub = [c["auc"] for c in cc
                   if sorted([c["bucket_a"], c["bucket_b"]]) == sorted([x1, x2])]
            bb[key] = sum(sub) / len(sub) if sub else float("nan")
        res["arms"][name] = {"macro": r["wprd_macro"], "weighted": r["wprd_weighted"],
                             "ci95": [lo, hi], "n_cells": r["n_cells"], "by_bucket": bb}
        _log(f"  {name:>22} {r['wprd_macro']:8.4f} {r['wprd_weighted']:9.4f} "
             f"[{lo:.4f},{hi:.4f}] "
             f"{bb['head-head']:7.4f} {bb['body-body']:7.4f} {bb['tail-tail']:7.4f}")

    A = res["arms"]
    pr = A["P_prior"]["macro"]
    sh = A["N_shuffled"]["macro"]
    gates.insert(0, {"gate": "G3 prior exactly 0.5000",
                     "pass": bool(abs(pr - 0.5) < 1e-6), "detail": f"{pr:.6f}"})
    gates.insert(1, {"gate": "G4 shuffled at chance",
                     "pass": bool(0.49 <= sh <= 0.51), "detail": f"{sh:.4f}"})

    raw = A["R_raw_fixed"]["macro"]
    f1 = A["F_s1_fixed"]["macro"]
    d_four = raw - f1
    best_lo = max((A[f"F_s{s:g}_fixed"]["macro"] for s in scales if s != 1.0),
                  default=float("nan"))
    d_band = best_lo - f1
    primary = ("DESTRUCTIVE" if d_four >= 0.02 else
               "BENIGN" if d_four < 0.005 else "PARTIAL")
    allpass = all(g["pass"] for g in gates)
    res["verdict"] = {"delta_fourier": d_four, "delta_bandwidth": d_band,
                      "best_low_bandwidth": best_lo, "primary": primary,
                      "gates_all_pass": bool(allpass)}

    _log(f"\n  PRIMARY   delta_fourier  = R_raw_fixed - F_s1_fixed = {d_four:+.4f}  -> {primary}")
    _log(f"  SECONDARY delta_bandwidth = best_low_bw - F_s1_fixed = {d_band:+.4f}")
    _log("\n  gates: " + "  ".join(
        f"{g['gate'].split()[0]}={'PASS' if g['pass'] else 'FAIL'}" for g in gates))
    if not allpass:
        _log("  *** A GATE FAILED. By the registration, NO NUMBER HERE IS REPORTABLE. ***")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
