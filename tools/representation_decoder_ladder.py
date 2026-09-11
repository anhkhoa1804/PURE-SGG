#!/usr/bin/env python
"""Paper C -- representation decoder ladder (H1 vs H2 diagnostic).

Pre-registered in docs/PAPER_C_REPRESENTATION_DECODER_LADDER_PREREGISTRATION.md.
Every architecture, budget, seed, partition and threshold below is quoted
from that document and is NOT recomputed or tuned here.

The question: after Readout v2 closed as a registered NULL, is the
relational information genuinely absent from `rel_feat` (H1), or present
but poorly decoded by the deployed cosine readout (H2)?

This replicates the p37 ladder (docs/READOUT_VS_REPRESENTATION_RESULT.md)
on the C1 geometry-REPAIRED representation, which p37 never saw -- p37 ran
on the pre-repair checkpoint (its R1_text = 0.5542; R0 here = 0.574988).

CPU only. Both dumps already cache `rel_feat`, so no forward pass and no
GPU is required. Reads dumps read-only; the explicit-output CLI writes a
result, console log, and provenance record under one new run directory.

Reuses, rather than reimplements, the validated machinery:
  - tools/within_pair_discrimination.py  (Groups, wprd)  -- the estimator
  - tools/wprd_geometry_control.py       (19-feature probe, train fitting)
  - tools/candidate_scorer_probe.py      (fold_of_image) -- the partition
  - tools/cprime_mechanism.py            (Mech)          -- the dump view
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_TOOLS = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, str(_TOOLS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MECH = _load("cprime_mechanism")
WPD = _load("within_pair_discrimination")
GEO = _load("wprd_geometry_control")
CSP = _load("candidate_scorer_probe")
VOCAB = _load("prepare_vg150_subset")

# ---- fixed by the preregistration; do not change without a new prereg ----
N_FOLDS = 5
FOLD_SALT = 0
SEED = 0
RIDGE_L2 = 1e-4
MLP_LR = 2e-3
MLP_WD = 1e-4
MLP_BATCH = 4096
MLP_EPOCHS = 25
GEOM_LBFGS_ITERS = 200
CAP = 64
BOOT = 2000
BOOT_SEED = 11
MATERIAL = 0.02          # p37's registered threshold, reused not reinvented
NULL_BAND = (0.47, 0.53)
TOTAL_DECODER_WIDTH = 51
BACKGROUND_INDEX = 50
FOREGROUND_INDICES = tuple(range(50))
PROTOCOL_AMENDMENT = "docs/PAPER_C_REPRESENTATION_DECODER_LADDER_AMENDMENT_2026-09-11.md"
HISTORICAL_RESULT = Path(__file__).resolve().parent.parent / "runs/paper_c_representation_decoder_ladder.json"
IMMUTABLE_OUTPUT_ROOTS = tuple(
    Path(__file__).resolve().parent.parent / p for p in (
        "runs/eval_C1",
        "runs/eval_readout_v2_R2_pilot_v3",
        "runs/eval_readout_v2_R2_matched_20260909T154739Z",
        "runs/eval_readout_v2_R2c_20260911T023722Z",
    )
)


def _log(m: str = "") -> None:
    print(m, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def _nvidia_smi() -> Dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False, "output": None, "returncode": None}
    try:
        p = subprocess.run([exe], check=False, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return {"available": True, "output": p.stdout, "returncode": int(p.returncode)}
    except OSError as exc:
        return {"available": False, "output": f"unavailable: {exc}", "returncode": None}


def decoder_contract(pred_vocab: List[str], background_indices: List[int]) -> Dict[str, Any]:
    """Validate and expose the registered 51-total/50-foreground contract."""
    vocab = [str(x).strip().lower() for x in pred_vocab]
    bg = sorted(int(i) for i in background_indices)
    if len(vocab) != TOTAL_DECODER_WIDTH:
        raise ValueError(f"registered decoder width is {TOTAL_DECODER_WIDTH}, got {len(vocab)}")
    if bg != [BACKGROUND_INDEX]:
        raise ValueError(f"registered background index is [{BACKGROUND_INDEX}], got {bg}")
    if vocab[BACKGROUND_INDEX] != "relation":
        raise ValueError(
            f"registered background column {BACKGROUND_INDEX} must be synthetic 'relation', "
            f"got {vocab[BACKGROUND_INDEX]!r}"
        )
    fg = [i for i in range(len(vocab)) if i not in bg]
    if fg != list(FOREGROUND_INDICES):
        raise ValueError(f"foreground columns must be 0..49, got {fg}")
    expected_foreground = sorted(str(x).strip().lower()
                                for x in VOCAB.STANDARD_VG150_PREDICATES)
    if vocab[:BACKGROUND_INDEX] != expected_foreground:
        raise ValueError("foreground predicate vocabulary is not canonical VG150 order")
    if len(set(vocab[:BACKGROUND_INDEX])) != len(FOREGROUND_INDICES):
        raise ValueError("foreground predicate vocabulary contains duplicate names")
    return {
        "total_decoder_width": TOTAL_DECODER_WIDTH,
        "background_index": BACKGROUND_INDEX,
        "background_name": vocab[BACKGROUND_INDEX],
        "foreground_indices": list(FOREGROUND_INDICES),
        "foreground_names": vocab[:BACKGROUND_INDEX],
    }


def decoder_class_mapping(col_to_class: torch.Tensor, foreground_indices: List[int]) -> torch.Tensor:
    """Map raw50 class ids to their total-vocabulary decoder columns."""
    if len(foreground_indices) != len(col_to_class):
        raise ValueError("foreground column map and raw50 class map have different lengths")
    n_classes = int(col_to_class.numel())
    out = torch.full((n_classes,), -1, dtype=torch.long)
    for foreground_position, decoder_column in enumerate(foreground_indices):
        class_id = int(col_to_class[foreground_position])
        if not 0 <= class_id < n_classes or int(out[class_id]) != -1:
            raise ValueError("raw50 class-to-decoder mapping is not a bijection")
        out[class_id] = int(decoder_column)
    if bool((out < 0).any()):
        raise ValueError("raw50 class-to-decoder mapping is incomplete")
    return out


def foreground_scores(scores: torch.Tensor) -> torch.Tensor:
    """Return the registered 50-column WPRD score view."""
    if scores.ndim != 2:
        raise ValueError(f"decoder scores must be rank-2, got shape {tuple(scores.shape)}")
    if scores.shape[1] == len(FOREGROUND_INDICES):
        return scores
    if scores.shape[1] == TOTAL_DECODER_WIDTH:
        return scores[:, list(FOREGROUND_INDICES)]
    raise ValueError(
        f"scores must have {TOTAL_DECODER_WIDTH} total or {len(FOREGROUND_INDICES)} "
        f"foreground columns, got {scores.shape[1]}"
    )


def _ensure_safe_output(out_path: Path, log_path: Optional[Path], provenance_path: Path) -> None:
    out = out_path.resolve()
    if out == HISTORICAL_RESULT.resolve():
        raise ValueError(f"refusing to overwrite immutable historical result: {out}")
    for root in IMMUTABLE_OUTPUT_ROOTS:
        try:
            out.relative_to(root.resolve())
        except ValueError:
            continue
        raise ValueError(f"refusing to write inside immutable historical run: {root}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing ladder output: {out}")
    if provenance_path.resolve().exists():
        raise FileExistsError(
            f"refusing to overwrite existing ladder artifact: {provenance_path.resolve()}"
        )


def _initial_provenance(
    repo: Path, dump_r0: str, dump_r2: str, prior: str, train_jsonl: str,
    out_path: Path, log_path: Optional[Path], mlp_epochs: int,
    command: Optional[List[str]],
) -> Dict[str, Any]:
    smi = _nvidia_smi()
    return {
        "provenance_schema": 1,
        "status": "started",
        "started_utc": _utc_now(),
        "repository": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "git_head": _git(repo, "rev-parse", "HEAD"),
        "git_status_porcelain": _git(repo, "status", "--porcelain"),
        "preregistration": "docs/PAPER_C_REPRESENTATION_DECODER_LADDER_PREREGISTRATION.md",
        "amendment": PROTOCOL_AMENDMENT,
        "command": list(command) if command is not None else None,
        "output_path": str(out_path),
        "console_log_path": str(log_path) if log_path is not None else None,
        "input_artifact_paths": {
            "dump_r0": str(Path(dump_r0).resolve()),
            "dump_r2": str(Path(dump_r2).resolve()),
            "prior": str(Path(prior).resolve()),
            "train_jsonl": str(Path(train_jsonl).resolve()),
        },
        "configuration": {
            "seed": SEED, "n_folds": N_FOLDS, "fold_salt": FOLD_SALT,
            "ridge_l2": RIDGE_L2, "mlp_lr": MLP_LR, "mlp_weight_decay": MLP_WD,
            "mlp_batch": MLP_BATCH, "mlp_epochs": int(mlp_epochs),
            "geometry_lbfgs_iters": GEOM_LBFGS_ITERS, "wprd_cap": CAP,
            "bootstrap_resamples": BOOT, "bootstrap_seed": BOOT_SEED,
            "material_threshold": MATERIAL,
            "total_decoder_width": TOTAL_DECODER_WIDTH,
            "background_index": BACKGROUND_INDEX,
            "foreground_indices": list(FOREGROUND_INDICES),
        },
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cpu_environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python_executable": sys.executable,
            "cpu_count": os.cpu_count(), "uname": dict(platform.uname()._asdict()),
        },
        "cuda": {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "nvidia_smi": smi,
        },
    }


# ------------------------------------------------------------------ fitting
def folds_of(B, salt: int = FOLD_SALT) -> torch.Tensor:
    """Image-level folds, broadcast onto GT rows. Identical partition rule to
    p22/p25/p26/p29/p32/p37 -- rows from one image never straddle a fold."""
    per = [CSP.fold_of_image(str(s), N_FOLDS, salt) for s in B.meta["image_id"]]
    flat: List[int] = []
    for i in range(B.n_images):
        flat.extend([per[i]] * len(B.meta["pairs"][i]))
    return torch.tensor(flat)[B.gt_row]


def xfit_ridge(X: torch.Tensor, y: torch.Tensor, fold: torch.Tensor, C: int,
               l2: float = RIDGE_L2) -> torch.Tensor:
    """Closed-form one-hot ridge, cross-fitted. Deterministic: no seed, no
    epochs. This is p37's `xfit_linear`, unmodified."""
    out = torch.zeros(X.shape[0], C)
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        A = X[tr]
        AtA = A.T @ A + l2 * A.shape[0] * torch.eye(A.shape[1])
        Y = torch.zeros(A.shape[0], C)
        Y[torch.arange(A.shape[0]), y[tr]] = 1.0
        W = torch.linalg.solve(AtA, A.T @ Y)
        out[te] = X[te] @ W
    return out


def make_mlp(d_in: int, C: int) -> torch.nn.Module:
    """The EXACT architecture fixed by the preregistration:
    Linear(768->768) -> GELU -> Linear(768->51). Note this differs from
    p37's MLP (768->512->512->51, ReLU) -- the two arms are not comparable
    and must never be reported as the same probe."""
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, 768),
        torch.nn.GELU(),
        torch.nn.Linear(768, C),
    )


def xfit_mlp(X: torch.Tensor, y: torch.Tensor, fold: torch.Tensor, C: int,
             epochs: int = MLP_EPOCHS, seed: int = SEED) -> Tuple[torch.Tensor, int]:
    out = torch.zeros(X.shape[0], C)
    n_params = 0
    for f in range(N_FOLDS):
        te = fold == f
        tr = ~te
        torch.manual_seed(seed)
        net = make_mlp(X.shape[1], C)
        n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
        Xtr, ytr = X[tr], y[tr]
        n = Xtr.shape[0]
        g = torch.Generator().manual_seed(seed + f)
        for _ep in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, MLP_BATCH):
                ix = perm[i:i + MLP_BATCH]
                opt.zero_grad()
                loss = F.cross_entropy(net(Xtr[ix]), ytr[ix])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            out[te] = net(X[te])
    return out, n_params


def fit_geometry_train_split(B, train_jsonl: str, Xr: torch.Tensor,
                             y_to_decoder_column: torch.Tensor) -> torch.Tensor:
    """p37's R8 protocol: fit the 19-feature probe on the TRAIN split, apply
    to validation. Removes the probe's cross-fitting advantage."""
    Xt_raw, yt = GEO.train_split_geometry(train_jsonl, list(B.classes))
    Xt, tstats = GEO._standardise(Xt_raw)
    Xv, _ = GEO._standardise(Xr, tstats)
    yt = y_to_decoder_column[yt]
    W = torch.zeros(Xt.shape[1], TOTAL_DECODER_WIDTH, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=GEOM_LBFGS_ITERS, history_size=10,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xt @ W, yt) + RIDGE_L2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (Xv @ W).detach()


# ------------------------------------------------------------------ scoring
def score_arm(Gs, scores: torch.Tensor, cap: int = CAP) -> Dict[str, Any]:
    r = WPD.wprd(Gs, foreground_scores(scores), cap)
    return {"wprd_macro": r["wprd_macro"], "wprd_weighted": r["wprd_weighted"],
            "n_cells": r["n_cells"], "frac_cells_above_half": r["frac_cells_above_half"],
            "_vals": r["_vals"]}


def paired_delta(vals_a: List[float], vals_b: List[float], n_boot: int = BOOT,
                  seed: int = BOOT_SEED) -> Dict[str, Any]:
    """Paired cell bootstrap, identical convention to tools/c0_c1_compare.py."""
    a = torch.tensor(vals_a, dtype=torch.float64)
    b = torch.tensor(vals_b, dtype=torch.float64)
    d = b - a
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(d), (n_boot, len(d)), generator=g)
    bs = d[idx].mean(dim=1)
    lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"delta": float(d.mean()), "ci95": [lo, hi],
            "p_delta_negative": float((bs < 0).double().mean()),
            "p_delta_positive": float((bs > 0).double().mean()),
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0)}


def cell_distribution(vals_a: List[float], vals_b: List[float],
                       keys: Optional[List[str]] = None, top: int = 15) -> Dict[str, Any]:
    """Is a gain broad or carried by a handful of cells? The preregistration
    binds this to be reported alongside every delta."""
    a = torch.tensor(vals_a, dtype=torch.float64)
    b = torch.tensor(vals_b, dtype=torch.float64)
    d = b - a
    n = int(d.numel())
    out = {
        "n_cells": n,
        "n_exactly_zero": int((d == 0).sum()),
        "n_positive": int((d > 0).sum()),
        "n_negative": int((d < 0).sum()),
        "frac_changed": float((d != 0).double().mean()),
        "gross_positive": float(d[d > 0].sum()) if bool((d > 0).any()) else 0.0,
        "gross_negative": float(d[d < 0].sum()) if bool((d < 0).any()) else 0.0,
        "mean": float(d.mean()), "median": float(d.median()), "std": float(d.std()),
        "max": float(d.max()), "min": float(d.min()),
    }
    order = torch.argsort(d.abs(), descending=True)[:top]
    out["top_movers"] = [
        {"key": (keys[int(i)] if keys is not None else str(int(i))),
         "a": float(a[int(i)]), "b": float(b[int(i)]), "delta": float(d[int(i)])}
        for i in order.tolist()]
    # concentration: what share of the net gain do the top 1% of cells carry?
    k = max(1, n // 100)
    topk = torch.argsort(d.abs(), descending=True)[:k]
    net = float(d.sum())
    out["top1pct_share_of_net"] = (float(d[topk].sum()) / net) if net != 0 else None
    return out


def cell_keys_for(B, Gs) -> List[str]:
    """Per-cell identity, in the exact order wprd() emits values."""
    keys: List[str] = []
    for p in range(Gs.G):
        cls = sorted(Gs.classes_of[p])
        if len(cls) < 2:
            continue
        for ai in range(len(cls)):
            for bi in range(ai + 1, len(cls)):
                keys.append(f"{Gs.key_of[p]}::{B.classes[cls[ai]]}|{B.classes[cls[bi]]}")
    return keys


# ------------------------------------------------------------------ main
def run_ladder(dump_r0: str, dump_r2: str, prior: str, train_jsonl: str,
               out_path: str, mlp_epochs: int = MLP_EPOCHS,
               log_path: Optional[str] = None,
               provenance_path: Optional[str] = None,
               command: Optional[List[str]] = None) -> Dict[str, Any]:
    t0 = time.time()
    out = Path(out_path)
    log = Path(log_path) if log_path is not None else None
    prov_path = (Path(provenance_path) if provenance_path is not None
                 else out.parent / "provenance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() == HISTORICAL_RESULT.resolve():
        raise ValueError(f"refusing to overwrite immutable historical result: {out.resolve()}")
    if log is not None and log.resolve().parent != out.resolve().parent:
        raise ValueError("console log must live in the registered ladder output directory")
    if prov_path.resolve().parent != out.resolve().parent:
        raise ValueError("provenance must live in the registered ladder output directory")
    _ensure_safe_output(out, log, prov_path)
    repo = Path(__file__).resolve().parent.parent
    provenance = _initial_provenance(
        repo, dump_r0, dump_r2, prior, train_jsonl, out, log, mlp_epochs, command)
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")

    res: Dict[str, Any] = {
        "tool": "representation_decoder_ladder",
        "prereg": "docs/PAPER_C_REPRESENTATION_DECODER_LADDER_PREREGISTRATION.md",
        "amendment": PROTOCOL_AMENDMENT,
        "dump_r0": dump_r0, "dump_r2": dump_r2, "prior": prior,
        "output_path": str(out),
        "console_log_path": str(log) if log is not None else None,
        "provenance_path": str(prov_path),
        "seed": SEED, "n_folds": N_FOLDS, "fold_salt": FOLD_SALT, "cap": CAP,
        "gates": [], "arms": {}, "contrasts": {},
    }

    _log("=" * 100)
    _log("REPRESENTATION DECODER LADDER -- CPU only, dumps read-only, NO GPU")
    _log("=" * 100)

    B = MECH.Mech(dump_r0, prior, "raw50")
    Gs = WPD.Groups(B)
    keys = cell_keys_for(B, Gs)
    _log(f"  images={B.n_images}  GT rows={B.n_gt:,}  classes={B.n_classes}  cells={len(keys):,}")

    def gate(name: str, passed: bool, detail: str) -> None:
        res["gates"].append({"gate": name, "pass": bool(passed), "detail": detail})
        _log(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    # ---- G-A: prior constant within group -> WPRD is prior-free
    dev = Gs.prior_is_constant(B.prior)
    gate("G-A prior constant within (s,o) group", dev < 1e-3, f"max dev = {dev:.3e}")

    # ---- G-B: rel_feat present and finite
    raw0 = torch.load(dump_r0, map_location="cpu", weights_only=False)
    contract = decoder_contract(
        raw0["pred_vocab"], raw0.get("background_predicate_indices", []))
    if B.n_classes != len(contract["foreground_indices"]):
        raise ValueError(
            f"raw50 evaluator has {B.n_classes} classes, expected "
            f"{len(contract['foreground_indices'])}"
        )
    class_to_decoder_column = decoder_class_mapping(
        B.col_to_class, contract["foreground_indices"])
    res["decoder_contract"] = {
        **contract,
        "class_id_to_decoder_column": class_to_decoder_column.tolist(),
        "wprd_scoring_columns": list(FOREGROUND_INDICES),
        "background_training_target": {
            "positive_gt_target": False,
            "ridge": "all_zero_one_hot_target_column",
            "cross_entropy": "non_target_competing_output",
            "wprd": "excluded",
        },
    }
    assert "rel_feat" in raw0, f"{dump_r0} has no rel_feat"
    RF = torch.cat([x.float() for x in raw0["rel_feat"]], 0)[B.gt_row]
    finite = bool(torch.isfinite(RF).all())
    gate("G-B rel_feat present & finite", finite and RF.shape[0] == B.n_gt,
         f"shape={tuple(RF.shape)} finite={finite}")

    # ---- G-C: rel_feat IS the tensor the deployed head read (A4 vs A1)
    bg = set(int(i) for i in raw0.get("background_predicate_indices", []))
    fg = [i for i in range(len(raw0["pred_vocab"])) if i not in bg]
    PE = raw0["pred_emb"].float()
    A4 = F.normalize(RF, dim=-1) @ F.normalize(PE, dim=-1).T
    stored_text = torch.cat([x.float() for x in raw0["text_logits"]], 0)[B.gt_row]
    md = float((A4[:, fg] - stored_text[:, fg]).abs().max())
    gate("G-C rel_feat reconstructs stored text_logits", md < 1e-2,
         f"max|recomputed - stored| = {md:.3e} (fp16 tol 1e-2)")

    # ---- G-D: rel_feat identical between R0 and R2 (backbone was frozen)
    raw2 = torch.load(dump_r2, map_location="cpu", weights_only=False)
    RF2 = torch.cat([x.float() for x in raw2["rel_feat"]], 0)[B.gt_row]
    rf_max_diff = float((RF - RF2).abs().max())
    gate("G-D rel_feat identical R0 vs R2 (frozen backbone)", rf_max_diff == 0.0,
         f"max|RF_R0 - RF_R2| = {rf_max_diff:.3e}")
    del raw2, RF2

    y = class_to_decoder_column[B.gt_y]
    fold = folds_of(B)
    fold_sizes = [int((fold == f).sum()) for f in range(N_FOLDS)]
    res["fold_sizes"] = fold_sizes
    _log(f"  folds: {fold_sizes}")

    # standardized rel_feat (+ bias column for the closed-form arms)
    RFs = (RF - RF.mean(0, keepdim=True)) / RF.std(0, keepdim=True).clamp_min(1e-6)
    RFs_b = torch.cat([RFs, torch.ones(RFs.shape[0], 1)], 1)

    # geometry features (raw 19), standardized for fusion
    Xr = GEO.geometry_features_raw(B)
    Xg_b, _ = GEO._standardise(Xr)                      # (n, 20) incl. bias
    Xg_s = Xg_b[:, :-1]                                  # (n, 19) standardized, no bias

    arms: Dict[str, torch.Tensor] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    _log("\n  fitting arms ...")
    arms["A1_frozen_baseline"] = B.fixed_ensemble(0.0)[B.gt_row]
    meta["A1_frozen_baseline"] = {"params": 0, "fitted": False,
                                   "output_width": len(FOREGROUND_INDICES),
                                   "wprd_width": len(FOREGROUND_INDICES),
                                   "form": "stored text_logits channel, fixed_ensemble(0.0)"}

    arms["A4_cosine_recomputed"] = A4
    meta["A4_cosine_recomputed"] = {"params": 0, "fitted": False,
                                     "output_width": TOTAL_DECODER_WIDTH,
                                     "wprd_width": len(FOREGROUND_INDICES),
                                     "form": "normalize(rel_feat) @ normalize(pred_emb).T (identity gate, not a capacity probe)"}

    t = time.time()
    arms["A2_linear"] = xfit_ridge(RFs_b, y, fold, TOTAL_DECODER_WIDTH)
    meta["A2_linear"] = {"params": int(RFs_b.shape[1] * TOTAL_DECODER_WIDTH), "fitted": True,
                          "output_width": TOTAL_DECODER_WIDTH,
                          "wprd_width": len(FOREGROUND_INDICES),
                          "form": "Linear(768->51) closed-form ridge on standardized rel_feat + bias",
                          "l2": RIDGE_L2, "fit_seconds": round(time.time() - t, 1)}
    _log(f"    A2_linear done ({meta['A2_linear']['fit_seconds']}s)")

    t = time.time()
    a3, a3_params = xfit_mlp(RFs, y, fold, TOTAL_DECODER_WIDTH, epochs=mlp_epochs)
    arms["A3_mlp"] = a3
    meta["A3_mlp"] = {"params": int(a3_params), "fitted": True,
                       "output_width": TOTAL_DECODER_WIDTH,
                       "wprd_width": len(FOREGROUND_INDICES),
                       "form": "Linear(768->768) -> GELU -> Linear(768->51)",
                       "epochs": mlp_epochs, "lr": MLP_LR, "weight_decay": MLP_WD,
                       "batch": MLP_BATCH, "seed": SEED,
                       "fit_seconds": round(time.time() - t, 1)}
    _log(f"    A3_mlp done ({meta['A3_mlp']['fit_seconds']}s, {a3_params:,} params)")

    t = time.time()
    arms["A5a_geometry_xfit"] = GEO.cross_fit_logits(Xg_b, y, fold, TOTAL_DECODER_WIDTH,
                                                      GEOM_LBFGS_ITERS, RIDGE_L2)
    meta["A5a_geometry_xfit"] = {"params": int(Xg_b.shape[1] * TOTAL_DECODER_WIDTH), "fitted": True,
                                  "output_width": TOTAL_DECODER_WIDTH,
                                  "wprd_width": len(FOREGROUND_INDICES),
                                  "form": "19 box-geometry features + bias, cross-fitted LBFGS",
                                  "fit_seconds": round(time.time() - t, 1)}
    _log(f"    A5a_geometry_xfit done ({meta['A5a_geometry_xfit']['fit_seconds']}s)")

    t = time.time()
    arms["A5b_geometry_trainfit"] = fit_geometry_train_split(
        B, train_jsonl, Xr, class_to_decoder_column)
    meta["A5b_geometry_trainfit"] = {"params": int(Xg_b.shape[1] * TOTAL_DECODER_WIDTH), "fitted": True,
                                      "output_width": TOTAL_DECODER_WIDTH,
                                      "wprd_width": len(FOREGROUND_INDICES),
                                      "form": "same probe, fitted on train.jsonl (p37 R8 protocol)",
                                      "fit_seconds": round(time.time() - t, 1)}
    _log(f"    A5b_geometry_trainfit done ({meta['A5b_geometry_trainfit']['fit_seconds']}s)")

    t = time.time()
    FUSE = torch.cat([RFs, Xg_s, torch.ones(RFs.shape[0], 1)], 1)
    arms["A6_fusion"] = xfit_ridge(FUSE, y, fold, TOTAL_DECODER_WIDTH)
    meta["A6_fusion"] = {"params": int(FUSE.shape[1] * TOTAL_DECODER_WIDTH), "fitted": True,
                          "output_width": TOTAL_DECODER_WIDTH,
                          "wprd_width": len(FOREGROUND_INDICES),
                          "form": "Linear(768+19+1 -> 51) ridge on concat[rel_feat, geometry]",
                          "l2": RIDGE_L2, "fit_seconds": round(time.time() - t, 1)}
    _log(f"    A6_fusion done ({meta['A6_fusion']['fit_seconds']}s)")

    g = torch.Generator().manual_seed(SEED)
    y_shuf = y[torch.randperm(len(y), generator=g)]
    t = time.time()
    arms["N1_shuffled_label_null"] = xfit_ridge(RFs_b, y_shuf, fold, TOTAL_DECODER_WIDTH)
    meta["N1_shuffled_label_null"] = {"params": int(RFs_b.shape[1] * TOTAL_DECODER_WIDTH), "fitted": True,
                                       "output_width": TOTAL_DECODER_WIDTH,
                                       "wprd_width": len(FOREGROUND_INDICES),
                                       "form": "A2 form fitted on permuted labels",
                                       "fit_seconds": round(time.time() - t, 1)}
    arms["N2_prior_control"] = B.prior[B.gt_row]
    meta["N2_prior_control"] = {"params": 0, "fitted": False,
                                 "output_width": len(FOREGROUND_INDICES),
                                 "wprd_width": len(FOREGROUND_INDICES),
                                 "form": "prior scored directly"}

    # ---- score every arm on the identical cells
    _log("\n  scoring (WPRD, cap=64, identical cells) ...")
    scored: Dict[str, Dict[str, Any]] = {}
    for name, s in arms.items():
        scored[name] = score_arm(Gs, s)
        _log(f"    {name:>26}  macro={scored[name]['wprd_macro']:.6f}  "
             f"weighted={scored[name]['wprd_weighted']:.6f}  cells={scored[name]['n_cells']:,}")

    n_cells_set = {v["n_cells"] for v in scored.values()}
    gate("G-G all arms scored on identical cell count", len(n_cells_set) == 1,
         f"cell counts = {sorted(n_cells_set)}")
    gate("G-E prior control reads exactly 0.5",
         abs(scored["N2_prior_control"]["wprd_macro"] - 0.5) < 1e-6,
         f"{scored['N2_prior_control']['wprd_macro']:.8f}")
    n1 = scored["N1_shuffled_label_null"]["wprd_macro"]
    gate("G-F shuffled-label null within [0.47, 0.53]",
         NULL_BAND[0] <= n1 <= NULL_BAND[1], f"{n1:.6f}")

    # ---- contrasts
    _log("\n  paired contrasts (same cells, bootstrap over cells) ...")
    for name in arms:
        if name == "A1_frozen_baseline":
            continue
        c = paired_delta(scored["A1_frozen_baseline"]["_vals"], scored[name]["_vals"])
        c["distribution"] = cell_distribution(scored["A1_frozen_baseline"]["_vals"],
                                               scored[name]["_vals"], keys)
        res["contrasts"][f"{name} - A1_frozen_baseline"] = c
        _log(f"    {name:>26} - A1  {c['delta']:+.6f}  CI [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}]"
             f"  excl0={c['ci_excludes_zero']}")

    for name in ("A2_linear", "A3_mlp", "A6_fusion", "A1_frozen_baseline"):
        c = paired_delta(scored["A5b_geometry_trainfit"]["_vals"], scored[name]["_vals"])
        res["contrasts"][f"{name} - A5b_geometry_trainfit"] = c
        _log(f"    {name:>26} - A5b {c['delta']:+.6f}  CI [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}]"
             f"  excl0={c['ci_excludes_zero']}")

    for name, sc in scored.items():
        res["arms"][name] = {**{k: v for k, v in sc.items() if not k.startswith("_")},
                              **meta.get(name, {})}

    # ---- registered classification (thresholds quoted, not invented)
    P_star_name = max(("A2_linear", "A3_mlp"), key=lambda n: scored[n]["wprd_macro"])
    P_star = scored[P_star_name]["wprd_macro"]
    G_val = scored["A5b_geometry_trainfit"]["wprd_macro"]
    A1_val = scored["A1_frozen_baseline"]["wprd_macro"]
    fusion = scored["A6_fusion"]["wprd_macro"]

    vs_geom = ("BEYOND_GEOMETRY" if P_star >= G_val + MATERIAL
               else "BELOW_GEOMETRY" if P_star <= G_val - MATERIAL
               else "GEOMETRY_EQUIVALENT")
    res["classification"] = {
        "P_star_arm": P_star_name, "P_star": P_star,
        "A1_frozen_baseline": A1_val, "G_train_fitted_geometry": G_val,
        "A6_fusion": fusion,
        "P_star_minus_A1": P_star - A1_val,
        "P_star_minus_G": P_star - G_val,
        "fusion_minus_P_star": fusion - P_star,
        "fusion_minus_G": fusion - G_val,
        "material_threshold": MATERIAL,
        "vs_geometry": vs_geom,
        "outcome_A_decoder_ceiling_near_A1": bool(abs(P_star - A1_val) < MATERIAL),
        "outcome_B_decoder_materially_exceeds_A1": bool(P_star >= A1_val + MATERIAL),
        "outcome_C_fusion_materially_exceeds_both": bool(
            fusion >= P_star + MATERIAL and fusion >= G_val + MATERIAL),
        "outcome_D_geometry_strongest": bool(
            G_val > max(P_star, A1_val, fusion)),
    }
    res["all_gates_pass"] = all(g["pass"] for g in res["gates"])
    res["runtime_seconds"] = round(time.time() - t0, 1)

    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    provenance.update({
        "status": "completed", "completed_utc": _utc_now(),
        "runtime_seconds": res["runtime_seconds"], "result_written": str(out),
    })
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"\n  all gates pass: {res['all_gates_pass']}")
    _log(f"  P* = {P_star_name} = {P_star:.6f}   A1 = {A1_val:.6f}   G = {G_val:.6f}")
    _log(f"  vs geometry: {vs_geom}")
    _log(f"\n[written] {out_path}  ({res['runtime_seconds']}s)")
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-r0", default="runs/eval_C1/pair_logits.pt")
    ap.add_argument("--dump-r2", default="runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt")
    ap.add_argument("--prior", default="datasets_vg150_clean/frequency_prior_train.json")
    ap.add_argument("--train-jsonl", default="datasets_vg150_clean/train.jsonl")
    ap.add_argument("--out", required=True,
                    help="new registered result path; historical ladder path is rejected")
    ap.add_argument("--log", required=True,
                    help="new registered console log path in the same output directory")
    ap.add_argument("--provenance", default=None,
                    help="optional provenance path; defaults to <out-dir>/provenance.json")
    ap.add_argument("--mlp-epochs", type=int, default=MLP_EPOCHS,
                    help="preregistered at 25; exposed only so tests can run a tiny budget")
    args = ap.parse_args(argv)
    out = Path(args.out)
    log = Path(args.log)
    if log.resolve().parent != out.resolve().parent:
        ap.error("--log must be in the same new output directory as --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        ap.error(f"refusing to overwrite existing console log: {log}")
    if out.resolve() == HISTORICAL_RESULT.resolve():
        ap.error(f"refusing to overwrite immutable historical result: {out}")
    provenance = Path(args.provenance) if args.provenance is not None else out.parent / "provenance.json"
    if provenance.resolve().exists():
        ap.error(f"refusing to overwrite existing provenance: {provenance}")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, value):
            for stream in self.streams:
                stream.write(value)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    with log.open("x", encoding="utf-8") as log_fh:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, log_fh)
        try:
            run_ladder(args.dump_r0, args.dump_r2, args.prior, args.train_jsonl,
                       args.out, mlp_epochs=args.mlp_epochs, log_path=args.log,
                       provenance_path=args.provenance,
                       command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        finally:
            sys.stdout = old_stdout
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
