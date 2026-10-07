#!/usr/bin/env python3
"""CPU-only, identity-joined nested R2 correction from frozen artifacts.

No model is trained here. The only fitted quantities are scalar temperatures
and nonnegative residual coefficients on CAL-FIT. FIT-trained pair/geometry
weights and the frozen rel_feat readout are loaded from existing artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs/paper_c_fullscale_residual_audit_20261003T084403Z"
PILOT = ROOT / "runs/paper_c_n4fit_pilot_training_20261003T182118Z"
OUT = ROOT / "research_analysis"
TRAIN = ROOT / "datasets_vg150_clean/train.jsonl"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (str(row["image_id"]), int(row["subject_index"]),
            int(row["object_index"]), str(row["predicate"]).strip().lower())


def assert_same_population(parent_keys: list[Any], extension_keys: list[Any]) -> None:
    """Reject rowwise comparisons unless exact identity sets and multiplicity agree."""
    if len(parent_keys) != len(set(parent_keys)) or len(extension_keys) != len(set(extension_keys)):
        raise ValueError("duplicate row identity in nested comparison")
    if set(parent_keys) != set(extension_keys):
        raise ValueError("parent/extension row populations differ")


def assert_image_disjoint(*populations: set[str]) -> None:
    for index, left in enumerate(populations):
        for right in populations[index + 1:]:
            if left & right:
                raise ValueError("image populations overlap")


def derive_cal_partition(cal_image_ids: list[str], seed: int,
                         fit_count: int) -> tuple[list[str], list[str]]:
    chosen = set(np.random.default_rng(seed).choice(len(cal_image_ids), fit_count,
                                                    replace=False).tolist())
    fit = [image for index, image in enumerate(cal_image_ids) if index in chosen]
    check = [image for index, image in enumerate(cal_image_ids) if index not in chosen]
    return fit, check


def deterministic_vocabulary(fit_labels: list[str]) -> list[str]:
    """Registered lowercase, deduplicated, lexicographic vocabulary convention."""
    return sorted({str(label).strip().lower() for label in fit_labels if str(label).strip()})


def nested_logits(parent_logits: np.ndarray, residual_logp: np.ndarray,
                  alpha: float) -> np.ndarray:
    if parent_logits.shape != residual_logp.shape:
        raise ValueError("parent and residual logits must have identical row/class shapes")
    if alpha < 0:
        raise ValueError("nested residual coefficient must be nonnegative")
    return parent_logits + alpha * residual_logp


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.25, max_iter=100,
                                  line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(z / log_t.exp().clamp(1e-3, 1e3), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(1e-3, 1e3))


def fit_nested_alpha(parent_logits: np.ndarray, rel_logp: np.ndarray,
                     targets: np.ndarray) -> tuple[float, float, float]:
    """Fit alpha >= 0 and include alpha=0 as an exact candidate."""
    z = torch.as_tensor(parent_logits, dtype=torch.float64)
    r = torch.as_tensor(rel_logp, dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.long)

    def objective(alpha: float) -> float:
        return float(F.cross_entropy(z + float(alpha) * r, y))

    raw = torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float64,
                       requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.25, max_iter=100,
                                  line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(z + F.softplus(raw) * r, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    candidate = float(F.softplus(raw.detach()))
    candidates = [(0.0, objective(0.0)), (candidate, objective(candidate))]
    alpha, loss = min(candidates, key=lambda item: item[1])
    return alpha, loss, objective(0.0)


def metric(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.long)
    pred = z.argmax(dim=1).numpy()
    truth = y.numpy()
    classes = sorted(int(c) for c in np.unique(truth) if int(c) < 50)
    recalls = {str(c): float(np.mean(pred[truth == c] == c)) for c in classes}
    precisions = {str(c): float(np.sum((pred == c) & (truth == c)) /
                                max(1, np.sum(pred == c))) for c in classes}
    return {
        "rows": int(len(truth)),
        "log_loss": float(F.cross_entropy(z, y)),
        "accuracy": float(np.mean(pred == truth)),
        "macro_recall_present_foreground": float(np.mean(list(recalls.values()))),
        "classes_present": len(classes),
        "per_predicate_recall": recalls,
        "per_predicate_precision": precisions,
    }


def bootstrap_ci(loss_delta: np.ndarray, image_ids: np.ndarray,
                 seed: int = 20261003, reps: int = 2000) -> dict[str, Any]:
    images = np.asarray(sorted(set(image_ids.tolist())))
    code = {image: i for i, image in enumerate(images)}
    groups = np.asarray([code[x] for x in image_ids])
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        counts = np.bincount(rng.integers(0, len(images), len(images)),
                             minlength=len(images))
        weights = counts[groups]
        estimates[index] = np.dot(weights, loss_delta) / weights.sum()
    return {"unit": "image_id", "repetitions": reps, "seed": seed,
            "paired": True,
            "aggregation": "resample images with replacement; retain all rows and compute pooled row-weighted difference",
            "estimate": float(loss_delta.mean()),
            "ci95_percentile": [float(np.quantile(estimates, .025)),
                                float(np.quantile(estimates, .975))],
            "replicate_values": estimates.tolist()}


def source_rows(keep_images: set[str]) -> tuple[dict[tuple[str, int, int, str], dict[str, Any]], list[str]]:
    records: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    image_order: list[str] = []
    with TRAIN.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            iid = str(record["image_id"])
            image_order.append(iid)
            if iid not in keep_images:
                continue
            objects = record.get("objects") or []
            boxes = record.get("obj_boxes") or []
            names = []
            for obj in objects:
                labels = obj.get("names") or obj.get("name") or obj.get("label") or "object"
                names.append(str(labels[0] if isinstance(labels, list) and labels else labels).strip().lower())
            for rel in record.get("relationships") or []:
                s, o = int(rel.get("subject_id", -1)), int(rel.get("object_id", -1))
                predicate = str(rel.get("predicate", "")).strip().lower()
                if s < 0 or o < 0 or s >= len(objects) or o >= len(objects) or s == o or not boxes:
                    continue
                row_key = (iid, s, o, predicate)
                require(row_key not in records, f"duplicate source relation key: {row_key}")
                records[row_key] = {"pair": (names[s], names[o]), "record": record,
                                    "s": s, "o": o, "predicate": predicate}
    require(len(image_order) == len(set(image_order)), "duplicate source image identity")
    return records, image_order


def load_predictions(path: Path) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows_data"]
    out = {key(row): row for row in rows}
    require(len(out) == len(rows), f"duplicate saved prediction keys in {path}")
    return out


def run() -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "",
            "CPU-only invariant: CUDA_VISIBLE_DEVICES must be empty")
    # Reuse the exact registered geometry implementation, without running its
    # fit/runner path or initializing CUDA.
    runner_path = AUDIT / "fullscale_residual_runner.py"
    spec = importlib.util.spec_from_file_location("frozen_fullscale_runner", runner_path)
    require(spec is not None and spec.loader is not None, "cannot load frozen geometry source")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    frozen = json.loads((AUDIT / "train_calibration_manifest.json").read_text())
    pilot_split = json.loads((PILOT / "population_manifests/calfit_calcheck_manifest.json").read_text())
    pilot_pop = json.loads((PILOT / "population_manifests/pilot_population_manifest.json").read_text())
    params = json.loads((PILOT / "calibration_parameters.json").read_text())
    predictions_fit = load_predictions(PILOT / "calfit_predictions.json")
    predictions_check = load_predictions(PILOT / "calcheck_predictions.json")
    full_predictions = np.load(AUDIT / "fullscale_residual_row_predictions.npz", allow_pickle=False)
    saved_models = torch.load(AUDIT / "fullscale_residual_fit_models.pt", map_location="cpu", weights_only=False)
    fit_ids = set(map(str, frozen["fit_image_ids"]))
    cal_ids = set(map(str, frozen["calibration_image_ids"]))
    e2_ids = set(map(str, frozen["excluded_e2_image_ids"]))
    assert_image_disjoint(fit_ids, cal_ids, set(map(str, frozen["validation_image_ids"])))
    pfit = set(map(str, pilot_pop["FIT"]["image_ids"]))
    pcal = set(map(str, pilot_pop["CAL"]["image_ids"]))
    require(not (pfit | pcal) & e2_ids, "pilot population overlaps frozen E2 exclusions")
    cal_order = list(map(str, pilot_pop["CAL"]["image_ids"]))
    calfit_ids = set(map(str, pilot_split["CAL-FIT_image_ids"]))
    calcheck_ids = set(map(str, pilot_split["CAL-CHECK_image_ids"]))
    require(pfit <= fit_ids and pcal <= cal_ids, "pilot split differs from frozen full split")
    require(calfit_ids | calcheck_ids == pcal and not calfit_ids & calcheck_ids,
            "CAL-FIT/CAL-CHECK partition mismatch")
    expected_calfit, expected_calcheck = derive_cal_partition(
        cal_order, int(pilot_split["seed"]), len(calfit_ids))
    require(expected_calfit == list(map(str, pilot_split["CAL-FIT_image_ids"])) and
            expected_calcheck == list(map(str, pilot_split["CAL-CHECK_image_ids"])),
            "CAL-FIT/CAL-CHECK deterministic split does not reproduce")
    assert_image_disjoint(pfit, pcal)
    assert_image_disjoint(calfit_ids, calcheck_ids)
    require(calfit_ids | calcheck_ids == pcal,
            "CAL-FIT/CAL-CHECK do not exhaustively partition pilot CAL")

    source, image_order = source_rows(pcal)

    all_saved = {**predictions_fit, **predictions_check}
    expected_keys = {k for k in source if k[0] in pcal}
    assert_same_population(list(all_saved), list(expected_keys))
    require(len(predictions_fit) == int(pilot_split["CAL-FIT_relation_rows"]), "CAL-FIT row count mismatch")
    require(len(predictions_check) == int(pilot_split["CAL-CHECK_relation_rows"]), "CAL-CHECK row count mismatch")
    require({k[0] for k in predictions_fit} == calfit_ids, "CAL-FIT image identities mismatch")
    require({k[0] for k in predictions_check} == calcheck_ids, "CAL-CHECK image identities mismatch")
    pred_vocab = [str(x).strip().lower() for x in frozen["predicate_vocab"]]
    pidx = {p: i for i, p in enumerate(pred_vocab)}

    pair_keys = [tuple(x) for x in saved_models["pair_keys"]]
    pair_index = {pair: index for index, pair in enumerate(pair_keys)}
    require(len(pair_index) == len(pair_keys), "duplicate saved pair-prior keys")
    pair_counts = np.asarray(saved_models["pair_counts"], dtype=np.int64)
    pair_lp_map = {pair: np.log((count + 1.0) / (count.sum() + 51.0))
                   for pair, count in zip(pair_keys, pair_counts)}
    global_count = np.asarray(saved_models["FIT_global_predicate_counts"], dtype=np.int64)
    global_lp = np.log((global_count + 1.0) / (global_count.sum() + 51.0))
    geom_mean = saved_models["GO_geometry_mean"].numpy()
    geom_scale = saved_models["GO_geometry_scale"].numpy()
    geom_weights = saved_models["GO_geometry_weights"].numpy()
    go_temperature_prior = float(saved_models["calibration"]["GO_temperature"])
    pair_temperature_prior = float(saved_models["calibration"]["pair_prior_temperature"])

    def assemble(keys: list[tuple[str, int, int, str]], predictions: dict, mode: str):
        targets, images, pair_logits, go_logits, rel_logits, supported = [], [], [], [], [], []
        for k in keys:
            row = predictions[k]
            src = source[k]
            y = pidx[k[3]]
            require(int(row["target"]) == y, f"target/vocabulary mismatch at {k}")
            targets.append(y)
            images.append(k[0])
            pair = src["pair"]
            counts = pair_counts[pair_index[pair]] if pair in pair_index else None
            lp = pair_lp_map.get(pair, global_lp)
            pair_logits.append(lp)
            supported.append(counts is not None)
            geom = runner.geometry19(src["record"].get("obj_boxes") or [], [src["s"]], [src["o"]])[0]
            standardized = (geom - geom_mean) / geom_scale
            go_logits.append(lp + np.concatenate([standardized, [1.0]]) @ geom_weights)
            if mode == "fit":
                rel_logits.append(np.asarray(row["relfeat_logits"], dtype=np.float64))
            else:
                # The CAL-CHECK sidecar stores the exact frozen pilot combined
                # score. Recover its log-probability offset rowwise, preserving
                # its class-order/identity and avoiding any new representation.
                alpha_old = float(params["alpha_nonnegative_softplus"])
                require(alpha_old > 0, "saved pilot alpha is zero; cannot recover CHECK offset")
                delta = (np.asarray(row["combined_raw_logits"], dtype=np.float64) -
                         np.asarray(row["N4_raw_logits"], dtype=np.float64)) / alpha_old
                rel_logits.append(delta * float(params["relfeat_temperature"]))
        return (np.asarray(targets, dtype=np.int64), np.asarray(images),
                np.asarray(pair_logits), np.asarray(go_logits),
                np.asarray(rel_logits), np.asarray(supported, dtype=bool))

    # Stable identity order follows each saved prediction file's row order,
    # which has already been exactly matched to the source key set.
    fit_keys = list(predictions_fit)
    check_keys = list(predictions_check)
    yf, imf, of, gf, rf, supf = assemble(fit_keys, predictions_fit, "fit")
    yc, imc, oc, gc, rc, supc = assemble(check_keys, predictions_check, "check")
    require(np.array_equal(yf, np.asarray([int(predictions_fit[k]["target"]) for k in fit_keys])), "FIT target order mismatch")
    require(np.array_equal(yc, np.asarray([int(predictions_check[k]["target"]) for k in check_keys])), "CHECK target order mismatch")
    rel_t = fit_temperature(rf, yf)
    require(abs(rel_t - float(params["relfeat_temperature"])) < 1e-6,
            "recomputed CAL-FIT rel_feat temperature disagrees with frozen pilot calibration")
    o_t = fit_temperature(of, yf)
    go_t = fit_temperature(gf, yf)
    o_parent_fit, o_parent_check = of / o_t, oc / o_t
    go_parent_fit, go_parent_check = gf / go_t, gc / go_t
    rel_logp_fit = torch.log_softmax(torch.as_tensor(rf / rel_t, dtype=torch.float64), dim=1).numpy()
    rel_logp_check = torch.log_softmax(torch.as_tensor(rc / rel_t, dtype=torch.float64), dim=1).numpy()
    old_alpha = float(params["alpha_nonnegative_softplus"])
    recovered_offsets = np.stack([
        (np.asarray(predictions_check[k]["combined_raw_logits"], dtype=np.float64) -
         np.asarray(predictions_check[k]["N4_raw_logits"], dtype=np.float64)) / old_alpha
        for k in check_keys
    ])
    recovery_error = float(np.max(np.abs(recovered_offsets - rel_logp_check)))
    require(recovery_error < 1e-8,
            f"recovered CAL-CHECK rel_feat log-probabilities disagree: {recovery_error}")
    o_alpha, o_fit_ext, o_fit_parent = fit_nested_alpha(o_parent_fit, rel_logp_fit, yf)
    go_alpha, go_fit_ext, go_fit_parent = fit_nested_alpha(go_parent_fit, rel_logp_fit, yf)

    def evaluate(parent, parent_raw, alpha, name):
        nested = nested_logits(parent, rel_logp_check, alpha)
        nested_raw = nested_logits(parent_raw, rel_logp_check, alpha)
        parent_m = metric(parent, yc)
        nested_m = metric(nested, yc)
        parent_raw_m = metric(parent_raw, yc)
        nested_raw_m = metric(nested_raw, yc)
        lp = F.cross_entropy(torch.as_tensor(parent, dtype=torch.float64), torch.as_tensor(yc), reduction="none").numpy()
        ln = F.cross_entropy(torch.as_tensor(nested, dtype=torch.float64), torch.as_tensor(yc), reduction="none").numpy()
        raw_lp = F.cross_entropy(torch.as_tensor(parent_raw, dtype=torch.float64), torch.as_tensor(yc), reduction="none").numpy()
        raw_ln = F.cross_entropy(torch.as_tensor(nested_raw, dtype=torch.float64), torch.as_tensor(yc), reduction="none").numpy()
        ci = bootstrap_ci(ln - lp, imc)
        delta = {
            "log_loss": nested_m["log_loss"] - parent_m["log_loss"],
            "accuracy": nested_m["accuracy"] - parent_m["accuracy"],
            "macro_recall_present_foreground": nested_m["macro_recall_present_foreground"] - parent_m["macro_recall_present_foreground"],
        }
        per = []
        for cls in sorted(set(map(int, yc.tolist()))):
            if cls >= 50:
                continue
            ix = yc == cls
            per.append({"predicate_index": cls, "predicate": pred_vocab[cls], "support": int(ix.sum()),
                        "baseline_recall": parent_m["per_predicate_recall"][str(cls)],
                        "extended_recall": nested_m["per_predicate_recall"][str(cls)],
                        "delta_recall": nested_m["per_predicate_recall"][str(cls)] - parent_m["per_predicate_recall"][str(cls)],
                        "baseline_precision": parent_m["per_predicate_precision"][str(cls)],
                        "extended_precision": nested_m["per_predicate_precision"][str(cls)],
                        "delta_precision": nested_m["per_predicate_precision"][str(cls)] - parent_m["per_predicate_precision"][str(cls)]})
        return {"model": name, "images": len(set(imc.tolist())),
                "alpha": float(alpha), "parent": parent_m,
                "extended": nested_m, "parent_raw": parent_raw_m,
                "extended_raw": nested_raw_m,
                "delta": delta, "raw_delta": {
                    "log_loss": nested_raw_m["log_loss"] - parent_raw_m["log_loss"],
                    "accuracy": nested_raw_m["accuracy"] - parent_raw_m["accuracy"],
                    "macro_recall_present_foreground": nested_raw_m["macro_recall_present_foreground"] - parent_raw_m["macro_recall_present_foreground"]},
                "bootstrap": ci, "raw_bootstrap": bootstrap_ci(raw_ln - raw_lp, imc),
                "per_predicate": per,
                "nesting_fit_check": {"parent_CE_CALFIT": o_fit_parent if name == "O" else go_fit_parent,
                                      "selected_extended_CE_CALFIT": o_fit_ext if name == "O" else go_fit_ext,
                                      "alpha_zero_exact_parent": True,
                                      "selected_fit_objective_not_worse_than_parent": (o_fit_ext <= o_fit_parent + 1e-12) if name == "O" else (go_fit_ext <= go_fit_parent + 1e-12)}}

    result_o = evaluate(o_parent_check, oc, o_alpha, "O")
    result_go = evaluate(go_parent_check, gc, go_alpha, "G+O")
    # The all-validation saved predictions serve only as a reproducibility
    # cross-check of the already-published parent outputs; no validation fit.
    val_y = full_predictions["targets"].astype(np.int64)
    val_go = metric(full_predictions["GO_calibrated_logits"], val_y)
    val_o = metric(full_predictions["pair_prior_calibrated_logits"], val_y)
    require(abs(val_go["log_loss"] - 1.3242384157227447) < 1e-6, "saved validation G+O metric no longer reproduces")
    require(abs(val_o["log_loss"] - 1.6323624327446515) < 1e-6, "saved validation calibrated O metric no longer reproduces")

    return {
        "schema": "paper-c-r2-properly-nested-correction-v1",
        "status": "COMPLETED_CPU_ONLY_IDENTITY_JOINED",
        "conclusion": "SUPPORTED_FOR_BOUNDED_CALCHECK_PILOT_ONLY",
        "device": "CPU; no CUDA APIs used",
        "population": {"FIT_images": len(fit_ids), "FIT_rows": int(frozen["fit_rows"]),
                       "pilot_FIT_images": len(pfit), "CAL_images": len(pcal),
                       "CAL_rows": int(pilot_pop["CAL"]["rows"]),
                       "CAL-FIT_images": len(calfit_ids), "CAL-FIT_rows": len(yf),
                       "CAL-CHECK_images": len(calcheck_ids), "CAL-CHECK_rows": len(yc),
                       "CAL-FIT_seed": int(pilot_split["seed"]),
                       "same_rows_for_all_four_arms": True,
                       "image_disjoint_CALFIT_CALCHECK": True,
                       "validation_used_for_fitting": False,
                       "E2_used": False,
                       "key": "(image_id, subject_index, object_index, normalized predicate)"},
        "vocabulary_and_prior": {"predicate_classes": 51, "object_label_rule": "first object label, strip+lowercase",
                                  "prior_fit": "saved fullscale FIT-only ordered pair counts; Laplace alpha=1; unsupported pair uses FIT-global Laplace prior",
                                  "fit_supported_CALFIT_rows": int(supf.sum()), "unsupported_CALFIT_rows": int((~supf).sum()),
                                  "fit_supported_CALCHECK_rows": int(supc.sum()), "unsupported_CALCHECK_rows": int((~supc).sum()),
                                  "unique_fit_pairs": len(pair_keys),
                                  "CAL_CHECK_relfeat_logp_recovery_max_abs": recovery_error,
                                  "pair_prior_calibration_temperature": float(o_t),
                                  "GO_calibration_temperature": float(go_t),
                                  "relfeat_temperature": float(rel_t)},
        "G_plus_O": {"geometry": "exact saved fullscale registered geometry19 implementation; standardized with FIT mean/scale; intercept appended",
                     "weights": "saved FIT-only 20x51 matrix (1,020 parameters); no refit",
                     "GO_raw_formula": "log P_FIT(y|ordered_pair) + [(geometry19 - FIT_mean)/FIT_scale, 1] @ W_FIT",
                     "O": result_o, "G+O": result_go},
        "validation_parent_reproduction_only": {"rows": len(val_y), "images": len(set(full_predictions["image_id"].tolist())),
                                                 "O_calibrated_LL": val_o["log_loss"], "G+O_calibrated_LL": val_go["log_loss"],
                                                 "validation_used_for_fit": False},
        "historical_population_reconciliation": {
            "calibrated_O_prior_vs_wrong_current_O_only": {
                "historical": {"log_loss": 1.6323624327446515, "images": 1182, "rows": 14991,
                               "population": "fixed current validation", "calibration": "temperature fit on full CAL; T=0.6527293293"},
                "previous_R2": {"log_loss": 1.8078775991028484, "images": len(calcheck_ids), "rows": len(yc),
                                "population": "bounded CAL-CHECK", "calibration": "no temperature; raw FIT pair log-probabilities"},
                "same_pair_counts_vocabulary_support_and_backoff": True,
                "same_population": False,
                "same_labels": True,
                "same_pair_support_rule": True,
                "same_supported_row_set": False,
                "same_backoff": True,
                "same_calibration": False,
                "same_split": False,
                "same_row_count": False,
                "same_image_count": False,
                "same_fitting_regime": False,
                "same_population_labels_calibration_split_or_fitting_regime": False,
                "comparable": False,
                "explanation": "The prior table, 51 outputs, FIT counts, Laplace smoothing, and FIT-global unsupported-pair backoff match. The rows and image population differ, and the old validation score is temperature-calibrated while the previous R2 O score is raw on CAL-CHECK. The 0.175515 loss difference is therefore not a model regression comparison."
            },
            "historical_O_plus_rel_vs_previous_R2_O_plus_rel": {
                "historical": {"log_loss": 1.2663293004670928, "images": 1182, "rows": 14991,
                               "population": "fixed current validation", "source": "nuisance_ladder_v2 results; O+rel offset and temperature fit on full CAL"},
                "previous_R2": {"log_loss": 1.1857673743258468, "images": len(calcheck_ids), "rows": len(yc),
                                "population": "bounded CAL-CHECK", "source": "bounded pilot; raw O parent and untemperatured rel log-probabilities"},
                "same_population_labels_calibration_split_or_fitting_regime": False,
                "same_population": False,
                "same_labels": True,
                "same_pair_support_rule": True,
                "same_supported_row_set": False,
                "same_backoff": True,
                "same_calibration": False,
                "same_split": False,
                "same_row_count": False,
                "same_image_count": False,
                "same_fitting_regime": False,
                "comparable": False,
                "explanation": "The evaluation population and fit/calibration regime differ. The former is full-validation scoring after full-CAL fitting; the latter is CAL-CHECK after CAL-FIT fitting. These values must not be differenced or pooled."
            },
            "retracted_headline": {"delta_log_loss": -0.6221102247770016,
                                   "status": "RETRACTED_AS_INTENDED_R2_HEADLINE",
                                   "reason": "It compared raw O and a residual fit without matched parent calibration; corrected here using CAL-FIT-calibrated fixed O and G+O parents with an explicit alpha=0 nesting point."}
        },
        "provenance": {str(path.relative_to(ROOT)): sha256(path) for path in [
            AUDIT / "train_calibration_manifest.json", AUDIT / "fullscale_residual_registration.json",
            AUDIT / "fullscale_residual_fit_models.pt", AUDIT / "fullscale_residual_row_predictions.npz",
            PILOT / "population_manifests/pilot_population_manifest.json",
            PILOT / "population_manifests/calfit_calcheck_manifest.json",
            PILOT / "calfit_predictions.json", PILOT / "calcheck_predictions.json",
            PILOT / "calibration_parameters.json", TRAIN,
            AUDIT / "fullscale_residual_runner.py", ROOT / "tools/r2_nested_correction.py"]},
        "limitations": ["evaluation is bounded CAL-CHECK pilot only; not current validation or fullscale N4",
                        "CAL-CHECK rel_feat logits are recovered algebraically from the frozen N4-FIT pilot sidecar; row identity and reconstruction are checked",
                        "GO is the already-fitted registered FIT-only linear geometry correction, not U/S",
                        "scalar log-probability offset tests predictive complementarity, not semantic specificity",
                        "per-predicate results are descriptive on this pilot population"]}


def main() -> dict[str, Any]:
    OUT.mkdir(exist_ok=True)
    result = run()
    (OUT / "R2_nested_nuisance_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows = []
    for model in ("O", "G+O"):
        item = result["G_plus_O"][model]
        for pred in item["per_predicate"]:
            rows.append({"contrast": model, **pred})
    with (OUT / "R2_per_predicate_effects.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["contrast"], -row["support"], row["predicate"])))
    reconciliation = result["historical_population_reconciliation"]
    (OUT / "R2_population_reconciliation.json").write_text(
        json.dumps(reconciliation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "R2_population_reconciliation.md").write_text(
        "# R2 Population Reconciliation\n\n"
        "The rows below are distinct analyses, not interchangeable estimates.\n\n"
        "| Quantity | Old result | Current previous-R2 result | Same population? | Same labels? | Same object-pair support? | Same backoff? | Same calibration? | Same split? | Same row count? | Same image count? | Same fitting regime? | Comparable? |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| O pair prior LL | 1.632362; validation, 14,991 rows / 1,182 images | 1.807878; CAL-CHECK, 3,142 rows / 250 images | No | Yes, same frozen 51 classes | Same lookup/rule; supported rows differ with population | Yes, FIT-global Laplace-1 | No: full-CAL T vs raw | No | No | No | No: count table same, calibration regime differs | No |\n"
        "| O + rel_feat LL | 1.266329; validation, 14,991 / 1,182 | 1.185767; CAL-CHECK, 3,142 / 250 | No | Yes, same 51 classes | Same O support rule; supported rows differ | Yes | No: full-CAL calibration vs CAL-FIT fitted offset and uncalibrated O | No | No | No | No: calibration/fitting regime differs | No |\n\n"
        "The previous O-only result is the same underlying FIT count lookup and backoff as the canonical pair prior, but it is an *uncalibrated score on a different held-out population*. It is worse numerically than the calibrated validation result without implying a changed or degraded prior. The earlier O+rel values also differ in evaluation population and calibration regime. Neither pair of numbers may be directly subtracted.\n\n"
        "The corrected R2 instead uses the same CAL-CHECK rows for O, G+O, O+rel_feat, and G+O+rel_feat; the parent logits are temperature-calibrated on CAL-FIT and remain fixed. Full structured fields and the historical retraction are in [R2_population_reconciliation.json](R2_population_reconciliation.json).\n",
        encoding="utf-8")
    # Refresh the active analysis hash inventory while retaining the previous
    # R2 hashes as superseded provenance rather than erasing the correction
    # history.
    hashes_path = OUT / "hashes.json"
    hash_inventory = json.loads(hashes_path.read_text(encoding="utf-8")) if hashes_path.exists() else {
        "schema": "portfolio-gate-artifact-hashes-v1", "files": {}}
    files = hash_inventory.setdefault("files", {})
    superseded = hash_inventory.setdefault("superseded_R2_hashes", {})
    correction_targets = [
        OUT / "R2_nested_nuisance_analysis.json", OUT / "R2_nested_nuisance_analysis.md",
        OUT / "R2_population_reconciliation.json", OUT / "R2_population_reconciliation.md",
        OUT / "R2_per_predicate_effects.csv", OUT / "R2_CORRECTION_SUMMARY.md",
        OUT / "FINAL_REPORT.md", OUT / "README.md", OUT / "portfolio_gate_decision.md",
        OUT / "R3_WPRD_construct_validity.md", ROOT / "tools/r2_nested_correction.py",
        ROOT / "tools/portfolio_gate_analysis.py", ROOT / "tests/test_r2_nested_correction.py",
    ]
    for path in correction_targets:
        relative = str(path.relative_to(ROOT))
        if relative in files and relative.startswith("research_analysis/R2_") and relative not in superseded:
            superseded[relative] = files[relative]
        files[relative] = sha256(path)
    hashes_path.write_text(json.dumps(hash_inventory, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({name: {"parent_LL": result["G_plus_O"][name]["parent"]["log_loss"],
                             "extended_LL": result["G_plus_O"][name]["extended"]["log_loss"],
                             "delta_LL": result["G_plus_O"][name]["delta"]["log_loss"],
                             "CI95": result["G_plus_O"][name]["bootstrap"]["ci95_percentile"],
                             "alpha": result["G_plus_O"][name]["alpha"]}
                      for name in ("O", "G+O")}, indent=2))
    return result


if __name__ == "__main__":
    main()
