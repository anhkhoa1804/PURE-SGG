#!/usr/bin/env python
"""C0/C1 preflight -- verify the control is bit-exact and the treatment reaches
the live forward path, BEFORE any training GPU is spent.

Pre-registered in docs/PAPER_C_C0_C1_PREREGISTRATION.md.

Runs the SAME bounded eval entry point p68 used, once per arm, and records what
the model actually receives:

  * the 8 geometry channels, in situ, after the units decision at
    relational_model.py:727-738  -- C0 must show p68's 6-of-8 degeneracy,
    C1 must show 0 constant channels;
  * the Fourier features geom_mlp is actually fed (captured by a forward hook on
    geom_mlp itself, so geom_fourier_scale is proved to reach the live path
    rather than assumed);
  * fusion_gate statistics;
  * the bounded eval's own R@50 / mR@50.

PURELY OBSERVATIONAL: no forward-pass numerics are changed. Nothing is trained
and no checkpoint is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openvocab_rel.models.relational_model import ProgressiveRelationalDecoder  # noqa: E402
from openvocab_rel import geometry as geometry_mod  # noqa: E402
import openvocab_rel.models.relational_model as rel_mod  # noqa: E402

COLS = ["dx", "dy", "rw", "rh", "ar1", "ar2", "a1", "a2"]

_GATE: List[Dict[str, Any]] = []
_GEOM: List[torch.Tensor] = []
_FOURIER: List[torch.Tensor] = []
_FLAGS: Dict[str, Any] = {}

_orig_fp = ProgressiveRelationalDecoder.forward_pairs
_orig_geom = geometry_mod.geom_feats_torch


def _patched_fp(self, visual_tokens, sub_feat, obj_feat, geom_feat_raw):
    # Attach the geom_mlp input hook once, on the live module.
    if not _FLAGS.get("_hooked") and hasattr(self, "geom_mlp"):
        def _hook(mod, inp):
            t = inp[0]
            if isinstance(t, torch.Tensor) and t.numel() > 0 and len(_FOURIER) < 40:
                _FOURIER.append(t.detach().float().cpu())
        self.geom_mlp.register_forward_pre_hook(_hook)
        _FLAGS["_hooked"] = True
        _FLAGS["decoder.geom_fourier_scale"] = float(
            getattr(self, "geom_fourier_scale", float("nan")))
        _FLAGS["decoder.img_res"] = int(getattr(self, "img_res", -1))
        gb = getattr(self, "geom_B", None)
        if gb is not None:
            _FLAGS["geom_B_std"] = float(gb.detach().float().std())
            _FLAGS["geom_B_requires_grad"] = bool(gb.requires_grad)
    out = _orig_fp(self, visual_tokens, sub_feat, obj_feat, geom_feat_raw)
    gate = getattr(self, "last_vector_gate", None)
    if isinstance(gate, torch.Tensor) and gate.numel() > 0:
        _GATE.append({"per_pair_mean": gate.mean(dim=-1).detach().cpu(),
                      "flat": gate.detach().float().cpu().flatten()})
    return out


def _patched_geom(b1, b2):
    feat = _orig_geom(b1, b2)
    _GEOM.append(feat.detach().float().cpu())
    return feat


def _q(x: torch.Tensor, q: float, cap: int = 4_000_000) -> Any:
    """Quantile on a fixed-seed subsample -- torch.quantile refuses huge inputs."""
    if x.numel() == 0:
        return None
    if x.numel() > cap:
        g = torch.Generator().manual_seed(0)
        x = x[torch.randint(x.numel(), (cap,), generator=g)]
    return float(x.quantile(q))


def _stats(x: torch.Tensor) -> Dict[str, Any]:
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "std": float(x.std()),
            "n_distinct": int(torch.unique(torch.round(x * 1e6)).numel())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["C0", "C1", "C1a", "C1b"])
    ap.add_argument("--eval_batches", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--fourier_scale", type=float, default=0.01)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    pixel = args.arm in ("C1", "C1a")
    scale = args.fourier_scale if args.arm in ("C1", "C1b") else 1.0
    out_dir = Path(args.out_dir or f"runs/preflight_{args.arm}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ProgressiveRelationalDecoder.forward_pairs = _patched_fp
    geometry_mod.geom_feats_torch = _patched_geom
    rel_mod.geom_feats_torch = _patched_geom

    from openvocab_rel import train as train_mod

    argv = [
        "--stage", "3", "--gpu_preset", "l4_24gb",
        "--eval_only", "true", "--epochs", "0", "--resume", "true",
        "--resume_from", "checkpoints/demo_best/pure_best_adapt_light_mR50.pt",
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
        "--freq_bias_path", "datasets_vg150_clean/frequency_prior_train.json",
        "--freq_bias_alpha", "3.75", "--freq_bias_smoothing", "1.0",
        "--eval_sgg_use_gt_pairs", "true",
        "--eval_sgg_grounding_dino_enabled", "false",
        # ---- the ONLY two flags that differ between arms ----
        "--geom_input_pixel_space", "true" if pixel else "false",
        "--geom_fourier_scale", str(scale),
        "--run_name", f"preflight_{args.arm}", "--out_dir", str(out_dir),
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

    G = torch.cat(_GEOM, 0) if _GEOM else torch.zeros(0, 8)
    F = torch.cat(_FOURIER, 0) if _FOURIER else torch.zeros(0, 1)
    gate_flat = torch.cat([r["flat"] for r in _GATE]) if _GATE else torch.zeros(0)
    gate_pair = torch.cat([r["per_pair_mean"] for r in _GATE]) if _GATE else torch.zeros(0)

    chan = {c: _stats(G[:, k]) for k, c in enumerate(COLS)} if G.numel() else {}
    n_const = sum(1 for c in COLS if chan.get(c, {}).get("n_distinct") == 1)

    res: Dict[str, Any] = {
        "tool": "c0_c1_preflight",
        "prereg": "docs/PAPER_C_C0_C1_PREREGISTRATION.md",
        "arm": args.arm,
        "requested": {"geom_input_pixel_space": pixel, "geom_fourier_scale": scale},
        "observed_in_live_model": _FLAGS,
        "n_pairs_observed": int(G.shape[0]),
        "geom_channels": chan,
        "n_constant_channels": n_const,
        "fourier_features_into_geom_mlp": {
            "rows": int(F.shape[0]), "dim": int(F.shape[1]),
            "std": float(F.std()) if F.numel() else None,
            "absmax": float(F.abs().max()) if F.numel() else None,
            "mean": float(F.mean()) if F.numel() else None,
            "frac_saturated_abs_gt_0.99": float((F.abs() > 0.99).float().mean())
            if F.numel() else None,
        },
        "fusion_gate": {
            "n_elements": int(gate_flat.numel()),
            "mean": float(gate_flat.mean()) if gate_flat.numel() else None,
            "std": float(gate_flat.std()) if gate_flat.numel() else None,
            "min": float(gate_flat.min()) if gate_flat.numel() else None,
            "max": float(gate_flat.max()) if gate_flat.numel() else None,
            # torch.quantile caps its input size, so quantiles are taken on a
            # fixed-seed subsample; mean/std/min/max above use every element.
            "q25": _q(gate_flat, 0.25), "median": _q(gate_flat, 0.5),
            "q75": _q(gate_flat, 0.75),
            "per_pair_mean_std": float(gate_pair.std()) if gate_pair.numel() else None,
            "frac_in_0.45_0.55": float(((gate_flat > 0.45) & (gate_flat < 0.55))
                                       .float().mean()) if gate_flat.numel() else None,
        },
    }

    # ---- gates -----------------------------------------------------------
    g = []
    if args.arm == "C0":
        g.append({"gate": "P1 C0 reproduces p68's 6-of-8 degeneracy",
                  "pass": bool(n_const == 6), "detail": f"{n_const} constant"})
        g.append({"gate": "P2 C0 fusion_gate mean within 0.001 of p68's 0.50155",
                  "pass": bool(res["fusion_gate"]["mean"] is not None
                               and abs(res["fusion_gate"]["mean"] - 0.50155) < 0.001),
                  "detail": f"{res['fusion_gate']['mean']}"})
        g.append({"gate": "P3 C0 did not inherit C1 flags",
                  "pass": bool(_FLAGS.get("decoder.geom_fourier_scale") == 1.0),
                  "detail": f"scale={_FLAGS.get('decoder.geom_fourier_scale')}"})
    else:
        if pixel:
            g.append({"gate": "P4 C1 has 0 constant geometry channels",
                      "pass": bool(n_const == 0), "detail": f"{n_const} constant"})
            g.append({"gate": "P5 dx/dy retain sign-symmetric centre-offset semantics",
                      "pass": bool(chan and chan["dx"]["min"] < 0 < chan["dx"]["max"]
                                   and chan["dy"]["min"] < 0 < chan["dy"]["max"]),
                      "detail": f"dx[{chan.get('dx',{}).get('min')},"
                                f"{chan.get('dx',{}).get('max')}]"})
        g.append({"gate": f"P6 geom_fourier_scale={scale} reached the live decoder",
                  "pass": bool(_FLAGS.get("decoder.geom_fourier_scale") == scale),
                  "detail": f"observed {_FLAGS.get('decoder.geom_fourier_scale')}"})
    g.append({"gate": "P7 geom_B is the frozen historical matrix (std ~9.91, no grad)",
              "pass": bool(_FLAGS.get("geom_B_requires_grad") is False
                           and abs(_FLAGS.get("geom_B_std", 0) - 9.91) < 0.2),
              "detail": f"std={_FLAGS.get('geom_B_std')} "
                        f"requires_grad={_FLAGS.get('geom_B_requires_grad')}"})
    g.append({"gate": "P8 all captured tensors finite",
              "pass": bool(torch.isfinite(G).all() and torch.isfinite(F).all()
                           and torch.isfinite(gate_flat).all()),
              "detail": f"{G.shape[0]} pairs / {F.shape[0]} fourier rows"})
    res["gates"] = g
    res["gates_all_pass"] = bool(all(x["pass"] for x in g))

    (out_dir / "preflight.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"PREFLIGHT {args.arm}   pixel_space={pixel}  fourier_scale={scale}")
    print("=" * 100)
    print(f"  live decoder.geom_fourier_scale = {_FLAGS.get('decoder.geom_fourier_scale')}"
          f"   img_res = {_FLAGS.get('decoder.img_res')}")
    print(f"  geometry channels observed on {G.shape[0]:,} real pairs "
          f"-- {n_const} of 8 constant")
    for c in COLS:
        s = chan.get(c, {})
        print(f"    {c:>4}  std {s.get('std', 0):12.6f}  distinct {s.get('n_distinct', 0):>7}"
              f"  [{s.get('min', 0):.4f}, {s.get('max', 0):.4f}]")
    fo = res["fourier_features_into_geom_mlp"]
    print(f"  fourier into geom_mlp: dim {fo['dim']}  std {fo['std']:.4f}  "
          f"|max| {fo['absmax']:.4f}  frac|.|>0.99 {fo['frac_saturated_abs_gt_0.99']:.4f}")
    fg = res["fusion_gate"]
    print(f"  fusion_gate: mean {fg['mean']:.5f}  std {fg['std']:.5f}  "
          f"per-pair std {fg['per_pair_mean_std']:.6f}  "
          f"range [{fg['min']:.4f}, {fg['max']:.4f}]  in[.45,.55] {fg['frac_in_0.45_0.55']:.4f}")
    print("  gates:")
    for x in g:
        print(f"    {'PASS' if x['pass'] else 'FAIL'}  {x['gate']}  |  {x['detail']}")
    print(f"\n  wrote {out_dir / 'preflight.json'}")
    return 0 if res["gates_all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
