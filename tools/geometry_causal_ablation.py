#!/usr/bin/env python
"""p70 -- inference-time causal ablation of PURE's two geometry pathways.

Pre-registered in docs/GEOMETRY_CAUSAL_ABLATION_PREREGISTRATION.md.

WHY THIS IS CHEAP. With the historical checkpoint's own eval configuration
(`ensemble_alpha = 0.0`, `text_conditioned_projection_enabled = false`), the
deployed model term is exactly

    model_term = layer_norm_51( normalize(rel_feat) @ normalize(pred_emb).T )

verified against the p36 cache to 1.1e-4 (pure fp16 storage error). So every
downstream metric this project computes from a pair-logit dump is a
deterministic function of `rel_feat`. One GPU pass that emits `rel_feat` under
K geometry ablations therefore yields K complete evaluation arms.

WHY THERE IS NO REIMPLEMENTATION RISK. Every arm calls the *unmodified*
`ProgressiveRelationalDecoder.forward_pairs`. The two geometry pathways are
separated purely by *what is fed in*:

    arm                  forward_pairs geom_raw   edge-layer geom injection
    A_full               real                     real   (untouched)
    B_no_fusion_geom     neutralised              real   (overridden)
    C_no_edge_geom       real                     neutralised (overridden)
    D_no_geom            neutralised              neutralised (untouched)
    E_dx_only            dy neutralised           follows input
    F_dy_only            dx neutralised           follows input

"Neutralised" means dx=dy=0 in the 8-vector. p68 measured that the other six
channels are ALREADY bit-exactly constant in production, so zeroing dx,dy makes
the whole geometry vector constant across pairs: geometry then carries exactly
zero information while keeping its full magnitude and its unit-norm
contribution to the fusion. This is an information ablation, not a magnitude
ablation -- gate=0.5 still mixes in a geom_norm of norm 1.

Arm A is gated against the eval's own dumped rel_feat for bit-exact equality.
Nothing historical is modified; output goes to runs/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openvocab_rel.models.relational_model import (  # noqa: E402
    ProgressiveEdgeConditionedLayer,
    ProgressiveRelationalDecoder,
)
import openvocab_rel.evals as evals_mod  # noqa: E402

ARMS = [
    "A_full",
    "B_no_fusion_geom",
    "C_no_edge_geom",
    "D_no_geom",
    "E_dx_only",
    "F_dy_only",
]

_RECORD = False
_EDGE_OVERRIDE: Optional[torch.Tensor] = None
_REC: Dict[str, List[torch.Tensor]] = {a: [] for a in ARMS}
_DIAG: List[Dict[str, Any]] = []

_orig_fp = ProgressiveRelationalDecoder.forward_pairs
_orig_edge = ProgressiveEdgeConditionedLayer.forward


def _patched_edge(self, sub_feat, obj_feat, rel_feat, visual_tokens, geom_feat):
    g = geom_feat if _EDGE_OVERRIDE is None else _EDGE_OVERRIDE.to(
        device=geom_feat.device, dtype=geom_feat.dtype)
    return _orig_edge(self, sub_feat, obj_feat, rel_feat, visual_tokens, g)


def _geom_proj(dec, g):
    """Recompute geom_feat_proj exactly as forward_pairs does (lines 441-448)."""
    gc = g[..., :8]
    gc = torch.nan_to_num(gc, nan=0.0, posinf=1.0, neginf=-1.0)
    gc = torch.clamp(gc, -10.0, 10.0)
    proj = (2.0 * math.pi * gc) @ dec.geom_B
    return dec.geom_mlp(torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1))


def _patched_forward_pairs(self, visual_tokens, sub_feat, obj_feat, geom_feat_raw):
    global _EDGE_OVERRIDE
    if not _RECORD:
        return _orig_fp(self, visual_tokens, sub_feat, obj_feat, geom_feat_raw)

    g_real = geom_feat_raw
    g_zero = geom_feat_raw.clone(); g_zero[:, 0] = 0.0; g_zero[:, 1] = 0.0
    g_dx = geom_feat_raw.clone(); g_dx[:, 1] = 0.0          # keep dx only
    g_dy = geom_feat_raw.clone(); g_dy[:, 0] = 0.0          # keep dy only

    with torch.no_grad():
        p_real = _geom_proj(self, g_real)
        p_zero = _geom_proj(self, g_zero)

    plan = [
        ("A_full", g_real, None),
        ("B_no_fusion_geom", g_zero, p_real),
        ("C_no_edge_geom", g_real, p_zero),
        ("D_no_geom", g_zero, None),
        ("E_dx_only", g_dx, None),
        ("F_dy_only", g_dy, None),
    ]
    out_a = None
    for name, g, ovr in plan:
        _EDGE_OVERRIDE = ovr
        try:
            rel_i, gate_i = _orig_fp(self, visual_tokens, sub_feat, obj_feat, g)
        finally:
            _EDGE_OVERRIDE = None
        _REC[name].append(rel_i.detach().half().cpu())
        if name == "A_full":
            out_a = (rel_i, gate_i)
            gate = getattr(self, "last_vector_gate", None)
            with torch.no_grad():
                sn = torch.nn.functional.normalize(sub_feat, dim=-1)
                on = torch.nn.functional.normalize(obj_feat, dim=-1)
                sem = self._semantic_pair_feature(sn, on)
                gnorm = torch.nn.functional.normalize(p_real, dim=-1, eps=1e-6)
                rec: Dict[str, Any] = {
                    "n_pairs": int(rel_i.shape[0]),
                    "sem_norm_mean": float(sem.float().norm(dim=-1).mean()),
                    "geom_norm_mean": float(gnorm.float().norm(dim=-1).mean()),
                    "cos_sem_geom_mean": float(
                        (torch.nn.functional.normalize(sem.float(), dim=-1)
                         * gnorm.float()).sum(-1).mean()),
                    "geom_proj_prenorm_mean": float(p_real.float().norm(dim=-1).mean()),
                }
                if isinstance(gate, torch.Tensor) and gate.numel() > 0:
                    gf = gate.float()
                    fused = gf * sem.float() + (1.0 - gf) * gnorm.float()
                    rec["fused_norm_mean"] = float(fused.norm(dim=-1).mean())
                    rec["gate_flat"] = gf.flatten()[
                        torch.randperm(gf.numel())[: min(4096, gf.numel())]
                    ].cpu().numpy().astype(np.float32).tolist()
                    rec["gate_pair_mean"] = gf.mean(dim=-1).cpu().numpy().astype(
                        np.float32).tolist()
                _DIAG.append(rec)
    return out_a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_batches", type=int, default=0, help="0 = full split")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--out_dir", type=str, default="runs/p70_geometry_causal_ablation")
    ap.add_argument("--run_name", type=str, default="p70_geometry_causal_ablation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restrict the eval battery to eval_sgg_standard: it is the only evaluator
    # that emits the pair-logit dump, and it is the only one whose forward_pairs
    # calls are row-aligned with that dump. Every other evaluator is stubbed so
    # no unrelated forward_pairs call can pollute the recording. This changes
    # nothing inside eval_sgg_standard.
    _stub_names = [
        "eval_query_grounding", "eval_query_grounding_split",
        "eval_global_predicate_classification", "eval_object_leakage_diagnostic",
        "eval_geometry_hard_negative_stress", "eval_swap_consistency",
        "eval_prompt_robustness", "eval_prune_reliability",
        "eval_latency_throughput", "eval_prune_tradeoff",
        "eval_query_grounding_vs_k",
    ]
    for nm in _stub_names:
        if hasattr(evals_mod, nm):
            setattr(evals_mod, nm, (lambda *a, **k: {"stubbed": True}))

    _orig_sgg = evals_mod.eval_sgg_standard

    def _wrapped_sgg(*a, **k):
        global _RECORD
        _RECORD = True
        try:
            return _orig_sgg(*a, **k)
        finally:
            _RECORD = False

    evals_mod.eval_sgg_standard = _wrapped_sgg
    ProgressiveRelationalDecoder.forward_pairs = _patched_forward_pairs
    ProgressiveEdgeConditionedLayer.forward = _patched_edge

    from openvocab_rel import train as train_mod

    dump_path = out_dir / "pair_logits_armA.pt"
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
        "--eval_sgg_dump_pair_logits_path", str(dump_path),
        "--eval_sgg_dump_rel_feat", "true",
        "--run_name", args.run_name, "--out_dir", str(out_dir),
        "--save_metrics_json", str(out_dir / "metrics.jsonl"),
    ]
    t0 = time.time()
    try:
        train_mod.main(argv)
    except SystemExit as exc:
        if int(exc.code or 0) != 0:
            raise
    finally:
        ProgressiveRelationalDecoder.forward_pairs = _orig_fp
        ProgressiveEdgeConditionedLayer.forward = _orig_edge
        evals_mod.eval_sgg_standard = _orig_sgg
    wall = time.time() - t0

    # ---- alignment gate: arm A must reproduce the eval's own dumped rel_feat
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    dumped = dump.get("rel_feat", [])
    recA = _REC["A_full"]
    n_dump_rows = int(sum(int(x.shape[0]) for x in dumped))
    n_rec_rows = int(sum(int(x.shape[0]) for x in recA))
    aligned = (len(dumped) == len(recA)) and (n_dump_rows == n_rec_rows)
    maxdiff = None
    if aligned:
        d_cat = torch.cat([x.float() for x in dumped], 0)
        a_cat = torch.cat([x.float() for x in recA], 0)
        maxdiff = float((d_cat - a_cat).abs().max())
        aligned = bool(maxdiff == 0.0)

    arms_path = out_dir / "arm_rel_feats.pt"
    torch.save({a: _REC[a] for a in ARMS}, arms_path)

    gate_flat = np.concatenate(
        [np.asarray(r["gate_flat"], dtype=np.float64) for r in _DIAG if "gate_flat" in r]
    ) if any("gate_flat" in r for r in _DIAG) else np.array([])
    gate_pair = np.concatenate(
        [np.asarray(r["gate_pair_mean"], dtype=np.float64) for r in _DIAG
         if "gate_pair_mean" in r]
    ) if any("gate_pair_mean" in r for r in _DIAG) else np.array([])

    def q(x, p):
        return float(np.quantile(x, p)) if x.size else None

    diag = {
        "n_calls": len(_DIAG),
        "n_pairs_total": int(sum(r["n_pairs"] for r in _DIAG)),
        "wall_seconds": wall,
        "alignment_gate": {
            "n_dump_images": len(dumped), "n_rec_images": len(recA),
            "n_dump_rows": n_dump_rows, "n_rec_rows": n_rec_rows,
            "max_abs_diff": maxdiff, "pass": bool(aligned),
        },
        "gate_elementwise": {
            "n": int(gate_flat.size),
            "mean": float(gate_flat.mean()) if gate_flat.size else None,
            "std": float(gate_flat.std()) if gate_flat.size else None,
            "min": float(gate_flat.min()) if gate_flat.size else None,
            "max": float(gate_flat.max()) if gate_flat.size else None,
            "q25": q(gate_flat, 0.25), "median": q(gate_flat, 0.5),
            "q75": q(gate_flat, 0.75),
            "frac_lt_0.1": float((gate_flat < 0.1).mean()) if gate_flat.size else None,
            "frac_lt_0.25": float((gate_flat < 0.25).mean()) if gate_flat.size else None,
            "frac_0.45_0.55": float(((gate_flat > 0.45) & (gate_flat < 0.55)).mean())
            if gate_flat.size else None,
            "frac_gt_0.75": float((gate_flat > 0.75).mean()) if gate_flat.size else None,
            "frac_gt_0.9": float((gate_flat > 0.9).mean()) if gate_flat.size else None,
        },
        "gate_per_pair_mean": {
            "n": int(gate_pair.size),
            "mean": float(gate_pair.mean()) if gate_pair.size else None,
            "std": float(gate_pair.std()) if gate_pair.size else None,
            "min": float(gate_pair.min()) if gate_pair.size else None,
            "max": float(gate_pair.max()) if gate_pair.size else None,
        },
        "norms": {
            k: float(np.mean([r[k] for r in _DIAG if k in r]))
            for k in ["sem_norm_mean", "geom_norm_mean", "fused_norm_mean",
                      "cos_sem_geom_mean", "geom_proj_prenorm_mean"]
        },
        "arms": ARMS,
        "arm_rel_feats": str(arms_path),
        "dump": str(dump_path),
    }
    with open(out_dir / "ablation_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
