#!/usr/bin/env python
"""p73 -- amendment to p72. Part A: is p72's null gate mis-specified?
Part B: can the raw geometry channels be decoded back out of their own Fourier
encoding, with no labels involved at all?

Pre-registered in docs/GEOM_FOURIER_BANDWIDTH_AMENDMENT.md.

CPU only. Reads the p36 cache and the historical checkpoint's geom_B (read
only). Does not modify tools/geom_fourier_bandwidth.py and does not touch
runs/p72_VOID_null_gate_failed/.

Part A refits p72's own N_shuffled arm at eight shuffle seeds (7 included, so
p72's own draw is reproduced rather than replaced) and reports the null's
distribution instead of one realisation.

Part B is label-free. If `fourier(x, geom_B)` preserves x, an MLP of the same
class as geom_mlp must be able to recover x from it. If it cannot, the map is
destroying the information before any predicate head sees it -- a conclusion
that does not route through WPRD, the label shuffle, or gate G4.
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


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FB = _load("geom_fourier_bandwidth")   # reuse p72's own _geom8/_fourier/_std_bias
MECH = _load("cprime_mechanism")
WPD = _load("within_pair_discrimination")
OBJ = _load("objective_ablation_relfeat")
Mech = MECH.Mech
N_FOLDS = 5
COLS = ["dx", "dy", "rw", "rh", "ar1", "ar2", "a1", "a2"]


def _log(m: str = "") -> None:
    print(m, flush=True)


def _fit_cls(X: torch.Tensor, labels: torch.Tensor, fold: torch.Tensor, C: int,
             hidden: int, epochs: int, lr: float, l2: float,
             gen: torch.Generator) -> torch.Tensor:
    """p72's `fit`, verbatim in behaviour: cross-fitted softmax-CE MLP."""
    out = torch.zeros(X.shape[0], C)
    for f in range(N_FOLDS):
        te = fold == f
        tr = torch.nonzero(~te, as_tuple=True)[0]
        net = OBJ.mlp(X.shape[1], hidden, C, 0)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
        Xtr, ytr = X[tr], labels[tr]
        n, bs = Xtr.shape[0], 8192
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, bs):
                ix = perm[i:i + bs]
                opt.zero_grad()
                torch.nn.functional.cross_entropy(net(Xtr[ix]), ytr[ix]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            out[te] = net(X[te])
    return out


def _fit_reg(X: torch.Tensor, Y: torch.Tensor, fold: torch.Tensor, hidden: int,
             epochs: int, lr: float, l2: float,
             gen: torch.Generator) -> torch.Tensor:
    """Same estimator class, 8-output regression head, MSE loss."""
    out = torch.zeros_like(Y)
    for f in range(N_FOLDS):
        te = fold == f
        tr = torch.nonzero(~te, as_tuple=True)[0]
        net = OBJ.mlp(X.shape[1], hidden, Y.shape[1], 0)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=l2)
        Xtr, Ytr = X[tr], Y[tr]
        n, bs = Xtr.shape[0], 8192
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, bs):
                ix = perm[i:i + bs]
                opt.zero_grad()
                torch.nn.functional.mse_loss(net(Xtr[ix]), Ytr[ix]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            out[te] = net(X[te])
    return out


def _r2(pred: torch.Tensor, Y: torch.Tensor) -> List[float]:
    """Out-of-fold R^2 per column against the column's own mean."""
    ss_res = ((pred - Y) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-12)
    return (1.0 - ss_res / ss_tot).tolist()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="runs/p36_relfeat_cache/pair_logits_relfeat.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--ckpt", default="checkpoints/demo_best/pure_best_adapt_light_mR50.pt")
    ap.add_argument("--out", default="runs/p73_fourier_null_and_invertibility/est.json")
    ap.add_argument("--scales", default="1.0,0.25,0.1,0.05")
    ap.add_argument("--null-seeds", default="7,11,13,17,19,23,29,31")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--skip-part-a", action="store_true")
    args = ap.parse_args(argv)

    _log("=" * 108)
    _log("p73 FOURIER NULL CALIBRATION + INVERTIBILITY -- CPU only, NO GPU")
    _log("=" * 108)

    B = Mech(args.dump, args.prior, "raw50")
    Gs = WPD.Groups(B)
    fold = OBJ.folds_of(B, 0)
    y = B.gt_y
    C = B.n_classes

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    Bmat = ck["model"]["decoder.geom_B"].float()
    del ck
    _log(f"  geom_B {tuple(Bmat.shape)}  std {Bmat.std():.4f}  |max| {Bmat.abs().max():.3f}")

    G_fixed = FB._geom8(B, normalise=False)
    G_prod = FB._geom8(B, normalise=True)
    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    # --- H4: re-assert the p68 contract facts so Part B stands alone ---------
    def _n_const(Gm: torch.Tensor) -> int:
        return sum(1 for k in range(8)
                   if int(torch.unique(torch.round(Gm[:, k] * 1e6)).numel()) == 1)
    n_prod, n_fixed = _n_const(G_prod), _n_const(G_fixed)
    gates: List[Dict[str, Any]] = [
        {"gate": "H4 prod reproduces p68's 6-of-8 degeneracy, fixed has 0 constant",
         "pass": bool(n_prod == 6 and n_fixed == 0),
         "detail": f"prod {n_prod} constant / fixed {n_fixed} constant"}]
    _log(f"  H4 contract check: prod {n_prod} constant, fixed {n_fixed} constant  "
         f"{'PASS' if n_prod == 6 and n_fixed == 0 else 'FAIL'}")

    res: Dict[str, Any] = {
        "tool": "fourier_null_and_invertibility",
        "prereg": "docs/GEOM_FOURIER_BANDWIDTH_AMENDMENT.md",
        "p72_void_run": "runs/p72_VOID_null_gate_failed",
        "estimator": {"hidden": args.hidden, "epochs": args.epochs, "lr": args.lr,
                      "l2": args.l2, "folds": N_FOLDS, "salt": 0, "cap": args.cap},
        "geom_B": {"shape": list(Bmat.shape), "std": float(Bmat.std()),
                   "absmax": float(Bmat.abs().max())},
        "part_a": {}, "part_b": {}}

    X_raw_fixed = FB._std_bias(G_fixed)

    # ================= PART A: null calibration =============================
    if not args.skip_part_a:
        seeds = [int(s) for s in args.null_seeds.split(",") if s.strip()]
        _log(f"\n  PART A -- null WPRD at {len(seeds)} shuffle seeds "
             f"(p72's design: N_shuffled fitted on R_raw_fixed)")
        nulls: Dict[str, float] = {}
        for sd in seeds:
            gen = torch.Generator().manual_seed(0)
            ysh = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(sd))]
            sc = _fit_cls(X_raw_fixed, ysh, fold, C, args.hidden, args.epochs,
                          args.lr, args.l2, gen)
            w = WPD.wprd(Gs, sc, args.cap)["wprd_macro"]
            nulls[str(sd)] = w
            _log(f"    seed {sd:>3}   null WPRD {w:.4f}"
                 + ("   <- p72's own draw" if sd == 7 else ""))
        v = torch.tensor(list(nulls.values()), dtype=torch.float64)
        mean, sd_, lo, hi = float(v.mean()), float(v.std()), float(v.min()), float(v.max())
        p72_draw = nulls.get("7")
        within2sd = (p72_draw is not None and sd_ > 0
                     and abs(p72_draw - mean) <= 2.0 * sd_)
        verdict_a = ("NULL-UNSTABLE" if sd_ > 0.01 else
                     "NULL-BIASED" if mean < 0.49 else
                     "G4-MISSPECIFIED" if (0.49 <= mean <= 0.51 and within2sd) else
                     "INCONCLUSIVE")
        res["part_a"] = {"seeds": nulls, "mean": mean, "sd": sd_,
                         "min": lo, "max": hi,
                         "p72_seed7_draw": p72_draw,
                         "p72_draw_within_2sd": bool(within2sd),
                         "verdict": verdict_a}
        _log(f"\n    null mean {mean:.4f}  sd {sd_:.4f}  range [{lo:.4f}, {hi:.4f}]")
        _log(f"    p72's 0.4873 within 2 sd of the mean: {within2sd}")
        _log(f"    PART A VERDICT -> {verdict_a}")

    # ================= PART B: invertibility ================================
    _log(f"\n  PART B -- can the 8 raw channels be decoded back out of their "
         f"own Fourier encoding? (label-free)")
    Y = ((G_fixed - G_fixed.mean(0, keepdim=True))
         / G_fixed.std(0, keepdim=True).clamp_min(1e-6))

    arms: Dict[str, torch.Tensor] = {"I_raw": X_raw_fixed}
    for s in scales:
        arms[f"I_s{s:g}"] = FB._std_bias(FB._fourier(G_fixed, Bmat * s))
    arms["I_s1_prod"] = FB._std_bias(FB._fourier(G_prod, Bmat))
    perm_rows = torch.randperm(Y.shape[0], generator=torch.Generator().manual_seed(5))
    arms["I_shuffled_rows"] = arms["I_s1"][perm_rows]

    _log(f"    rows {Y.shape[0]:,}   targets 8 standardised channels")
    _log(f"\n    {'arm':>16} {'meanR2':>8}  " + " ".join(f"{c:>7}" for c in COLS))
    _log("    " + "-" * 96)
    for name, X in arms.items():
        gen = torch.Generator().manual_seed(0)
        pred = _fit_reg(X, Y, fold, args.hidden, args.epochs, args.lr, args.l2, gen)
        r2 = _r2(pred, Y)
        mean_r2 = float(sum(r2) / len(r2))
        res["part_b"][name] = {"mean_r2": mean_r2,
                               "r2": {c: r2[k] for k, c in enumerate(COLS)},
                               "dims": int(X.shape[1])}
        _log(f"    {name:>16} {mean_r2:8.4f}  " + " ".join(f"{x:7.3f}" for x in r2))

    mr = {k: v["mean_r2"] for k, v in res["part_b"].items()}
    gates.append({"gate": "H1 identity control I_raw mean R2 > 0.95",
                  "pass": bool(mr["I_raw"] > 0.95), "detail": f"{mr['I_raw']:.4f}"})
    gates.append({"gate": "H2 row-shuffled null mean R2 < 0.05",
                  "pass": bool(mr["I_shuffled_rows"] < 0.05),
                  "detail": f"{mr['I_shuffled_rows']:.4f}"})
    gates.append({"gate": "H3 every arm scores every GT row, all finite",
                  "pass": bool(all(X.shape[0] == B.n_gt and torch.isfinite(X).all()
                                   for X in arms.values())),
                  "detail": f"{len(arms)} arms x {B.n_gt} rows"})

    s1 = mr["I_s1"]
    verdict_b = ("DESTRUCTIVE-CONFIRMED" if (s1 < 0.30 and mr["I_raw"] > 0.95) else
                 "BENIGN-CONFIRMED" if s1 > 0.70 else "PARTIAL")
    allpass = all(g["pass"] for g in gates)
    res["gates"] = gates
    res["verdict"] = {
        "part_a": res["part_a"].get("verdict", "SKIPPED"),
        "part_b": verdict_b,
        "mean_r2_s1": s1,
        "mean_r2_raw": mr["I_raw"],
        "gates_all_pass": bool(allpass)}

    _log(f"\n    PART B VERDICT -> {verdict_b}   "
         f"(mean R2 at the checkpoint's own bandwidth = {s1:.4f})")
    _log("\n  gates: " + "  ".join(
        f"{g['gate'].split()[0]}={'PASS' if g['pass'] else 'FAIL'}" for g in gates))
    if not allpass:
        _log("  *** A GATE FAILED. By the registration, NO NUMBER HERE IS REPORTABLE. ***")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    _log(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
