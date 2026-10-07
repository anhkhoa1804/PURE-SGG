#!/usr/bin/env python3
"""Reproduce legacy R3 summaries and delegate current R2 to its correction.

This tool does not train a neural model or modify historical artifacts. Its
R2 entrypoint delegates to ``tools/r2_nested_correction.py``; the old
independent/raw pair-prior comparison is retained under an explicitly
superseded function name for forensic reference. R3 reads saved WPRD tables
and computes descriptive leave-one-arm-out summaries.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_analysis"
AUDIT = ROOT / "runs/paper_c_fullscale_residual_audit_20261003T084403Z"
PILOT = ROOT / "runs/paper_c_n4fit_pilot_training_20261003T182118Z"
CANONICAL = ROOT / "datasets_vg150_clean/train.jsonl"
REP_MANIFEST = ROOT / "runs/canonical_train_representation_full_20261001T020630Z/representation/canonical_train_relfeat_manifest.json"
WPRD_PATHS = {
    "validation_all_labels": ROOT / "runs/p49_metric_grounding/corr.json",
    "validation_vg150_only": ROOT / "runs/p53_metric_grounding_vg150only/corr.json",
    "test_all_labels": ROOT / "runs/p61_test_metric_grounding/corr.json",
}
WPRD_POPULATIONS = {
    "validation_all_labels": {"split": "validation", "images": 10401, "relation_rows": 132556,
                              "wprd_cells": 20016, "eligibility": "unrestricted raw-name subject/object groups"},
    "validation_vg150_only": {"split": "validation", "images": 10401, "relation_rows": 132556,
                              "eligible_object_rows": 37121, "decidable_rows": 31125,
                              "wprd_cells": None, "eligibility": "subject and object both in VG150 object vocabulary"},
    "test_all_labels": {"split": "test", "images": 10403, "relation_rows": 132334,
                        "wprd_cells": None, "eligibility": "unrestricted raw-name subject/object groups; exact cell count not saved"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized(value: Any) -> str:
    return str(value).strip().lower()


def ordered_object_labels(record: dict[str, Any], subject: int, obj: int) -> tuple[str, str]:
    objects = record.get("objects") or []
    def label(index: int) -> str:
        names = objects[index].get("names") or []
        return str(names[0]).strip().lower() if names else ""
    return label(subject), label(obj)


def target_rows(record: dict[str, Any], predicate_index: dict[str, int]):
    objects = record.get("objects") or []
    boxes = record.get("obj_boxes") or []
    for relation in record.get("relationships") or []:
        s, o = int(relation.get("subject_id", -1)), int(relation.get("object_id", -1))
        pred = normalized(relation.get("predicate", ""))
        if 0 <= s < len(objects) and 0 <= o < len(objects) and s != o and boxes and pred in predicate_index:
            yield s, o, pred, predicate_index[pred]


def fit_pair_prior() -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray, dict[str, Any]]:
    split = read_json(AUDIT / "train_calibration_manifest.json")
    fit_ids = set(map(str, split["fit_image_ids"]))
    cal_ids = set(map(str, split["calibration_image_ids"]))
    val_ids = set(map(str, split["validation_image_ids"]))
    e2_ids = set(map(str, split["excluded_e2_image_ids"]))
    if fit_ids & cal_ids or fit_ids & val_ids or cal_ids & val_ids:
        raise ValueError("registered FIT/CAL/validation image manifests overlap")
    if (fit_ids | cal_ids) & e2_ids:
        raise ValueError("registered nuisance FIT/CAL overlap E2 exclusions")
    predicate_vocab = [normalized(x) for x in split["predicate_vocab"]]
    if len(predicate_vocab) != 51 or len(set(predicate_vocab)) != 51:
        raise ValueError("expected frozen 51-class output vocabulary")
    pidx = {p: i for i, p in enumerate(predicate_vocab)}
    pair_counts: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(51, dtype=np.int64))
    global_counts = np.zeros(51, dtype=np.int64)
    seen_fit: set[str] = set()
    fit_rows = 0
    with CANONICAL.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            iid = str(record["image_id"])
            if iid not in fit_ids:
                continue
            seen_fit.add(iid)
            for s, o, _pred, y in target_rows(record, pidx):
                pair = ordered_object_labels(record, s, o)
                pair_counts[pair][y] += 1
                global_counts[y] += 1
                fit_rows += 1
    if seen_fit != fit_ids:
        raise ValueError(f"canonical source did not cover registered FIT images: missing={len(fit_ids-seen_fit)}")
    if fit_rows != int(split["fit_rows"]):
        raise ValueError(f"FIT row count differs from registered source: {fit_rows} != {split['fit_rows']}")
    # Registered Laplace alpha=1 smoothing and FIT-global backoff.
    global_prob = (global_counts + 1.0) / (global_counts.sum() + 51.0)
    pair_prob = {key: (counts + 1.0) / (counts.sum() + 51.0) for key, counts in pair_counts.items()}
    meta = {
        "fit_images": len(fit_ids), "fit_rows": fit_rows,
        "calibration_images": len(cal_ids), "registered_calibration_rows": split["calibration_rows"],
        "validation_images": len(val_ids), "validation_rows_not_read": split.get("validation_rows"),
        "e2_excluded_images": len(e2_ids), "output_classes": 51,
        "observed_ordered_pairs": len(pair_prob),
        "pair_table_probability_entries": len(pair_prob) * 51,
        "pair_table_free_probability_parameters": len(pair_prob) * 50,
        "smoothing": "Laplace alpha=1 over the exact frozen 51-class vocabulary",
        "unsupported_pair_backoff": "FIT-global predicate prior, also Laplace alpha=1",
        "fit_manifest_sha256": sha256(AUDIT / "train_calibration_manifest.json"),
        "canonical_train_sha256": sha256(CANONICAL),
        "predicate_manifest_sha256": sha256(REP_MANIFEST),
        "FIT_image_ids_sha256": split["fit_image_sequence_sha256"],
        "CAL_image_ids_sha256": split["calibration_image_sequence_sha256"],
        "validation_image_ids_sha256": split["validation_image_ids_sha256"],
        "analysis_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    return pair_prob, global_prob, meta


def load_pilot_rows(split: str, pair_prob: dict[tuple[str, str], np.ndarray], global_prob: np.ndarray):
    prediction_file = "calfit_predictions.json" if split == "CAL-FIT" else "calcheck_predictions.json"
    pred = read_json(PILOT / prediction_file)
    raw_path = PILOT / "cal_selected_raw.jsonl"
    cal_manifest = read_json(PILOT / "population_manifests/calfit_calcheck_manifest.json")
    included_ids = set(map(str, cal_manifest[f"{split}_image_ids"]))
    raw_by_key: dict[tuple[str, int, int, str], tuple[tuple[str, str], int]] = {}
    p_manifest = read_json(AUDIT / "train_calibration_manifest.json")
    pidx = {normalized(p): i for i, p in enumerate(p_manifest["predicate_vocab"])}
    raw_seen_ids: set[str] = set()
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            iid = str(record["image_id"])
            if iid not in included_ids:
                continue
            raw_seen_ids.add(iid)
            for s, o, pred_name, _y in target_rows(record, pidx):
                key = (iid, s, o, pred_name)
                if key in raw_by_key:
                    raise ValueError(f"duplicate source relation identity {key}")
                raw_by_key[key] = (ordered_object_labels(record, s, o), _y)
    actual: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in pred["rows_data"]:
        key = (str(row["image_id"]), int(row["subject_index"]), int(row["object_index"]), normalized(row["predicate"]))
        if key in actual:
            raise ValueError(f"duplicate saved prediction identity {key}")
        actual[key] = row
    if set(actual) != set(raw_by_key):
        raise ValueError(f"raw/prediction relation keys disagree: raw_only={len(set(raw_by_key)-set(actual))}, prediction_only={len(set(actual)-set(raw_by_key))}")
    sorted_keys = list(actual)
    rows = [actual[k] for k in sorted_keys]
    y = np.asarray([int(r["target"]) for r in rows], dtype=np.int64)
    if all("relfeat_logits" in r for r in rows):
        rel_logits = np.asarray([r["relfeat_logits"] for r in rows], dtype=np.float64)
        recovery_max_abs = 0.0
    else:
        # The frozen CAL-CHECK sidecar retains raw combined and N4 logits but
        # omits the standalone rel_feat logits. Recover the already-frozen
        # temperature-normalized rel_feat log-probabilities algebraically
        # from z_combined = z_N4 + alpha * log_softmax(z_rel / T_rel).
        # alpha and T_rel were fit on CAL-FIT only. Multiplication by T_rel
        # gives an offset CE-equivalent to raw-logit log_softmax (up to a
        # row-wise constant, which cannot alter softmax or cross-entropy).
        cal_params = read_json(PILOT / "calibration_parameters.json")
        prior_alpha = float(cal_params["alpha_nonnegative_softplus"])
        prior_temperature = float(cal_params["relfeat_temperature"])
        if prior_alpha <= 0:
            raise ValueError("cannot recover CAL-CHECK rel_feat logits from zero saved pilot alpha")
        n4 = np.asarray([r["N4_raw_logits"] for r in rows], dtype=np.float64)
        combined = np.asarray([r["combined_raw_logits"] for r in rows], dtype=np.float64)
        rel_logits = ((combined - n4) / prior_alpha) * prior_temperature
        recovered_combined = n4 + prior_alpha * torch.log_softmax(
            torch.as_tensor(rel_logits / prior_temperature, dtype=torch.float64), dim=1
        ).numpy()
        recovery_max_abs = float(np.max(np.abs(recovered_combined - combined)))
        if recovery_max_abs > 1e-8:
            raise ValueError(f"saved CAL-CHECK rel_feat-logit recovery did not reproduce combined logits: {recovery_max_abs}")
    if split == "CAL-FIT":
        n4_logits = np.asarray([r["N4_raw_logits"] for r in rows], dtype=np.float64)
        if not np.isfinite(n4_logits).all():
            raise ValueError("unexpected N4 logits in CAL-FIT; this should not be used as an O baseline")
    if not np.isfinite(rel_logits).all():
        raise ValueError("nonfinite frozen rel_feat logits")
    for key in sorted_keys:
        if int(actual[key]["target"]) != raw_by_key[key][1]:
            raise ValueError(f"saved target differs from source predicate mapping at {key}")
    probs = np.stack([pair_prob.get(raw_by_key[k][0], global_prob) for k in sorted_keys])
    image_ids = np.asarray([k[0] for k in sorted_keys], dtype=str)
    supported = np.asarray([raw_by_key[k][0] in pair_prob for k in sorted_keys], dtype=bool)
    return y, rel_logits, np.log(probs), image_ids, supported, len(raw_seen_ids), recovery_max_abs


def fit_positive_alpha(base: np.ndarray, residual_logp: np.ndarray, y: np.ndarray) -> float:
    base_t = torch.as_tensor(base, dtype=torch.float64)
    offset_t = torch.as_tensor(residual_logp, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.long)
    raw = torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.25, max_iter=100, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-12)
    def closure():
        optimizer.zero_grad()
        alpha = F.softplus(raw)
        loss = F.cross_entropy(base_t + alpha * offset_t, y_t)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(F.softplus(raw.detach()))


def metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    z = torch.as_tensor(logits, dtype=torch.float64)
    target = torch.as_tensor(y, dtype=torch.long)
    prediction = z.argmax(1)
    present = [int(c) for c in torch.unique(target).tolist() if int(c) < 50]
    recall = [float((prediction[target == c] == c).double().mean()) for c in present]
    return {
        "rows": len(y), "log_loss": float(F.cross_entropy(z, target)),
        "accuracy": float((prediction == target).double().mean()),
        "macro_recall_present_predicate_classes": float(np.mean(recall)),
        "predicate_classes_present": len(present),
    }


def image_cluster_bootstrap(row_delta: np.ndarray, image_ids: np.ndarray, seed: int = 20261003, reps: int = 2000) -> dict[str, Any]:
    unique = np.asarray(sorted(set(image_ids.tolist())))
    lookup = {image: i for i, image in enumerate(unique)}
    groups = np.asarray([lookup[x] for x in image_ids])
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        count = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        weights = count[groups]
        vals[b] = np.dot(weights, row_delta) / weights.sum()
    return {"unit": "image_id", "repetitions": reps, "seed": seed, "paired": True,
            "aggregation": "resample held-out images with replacement; retain all rows and compute pooled row-mean loss difference",
            "ci95_percentile": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))],
            "estimate": float(row_delta.mean()), "replicate_values": vals.tolist()}


def run_r2_superseded_invalid_comparison() -> dict[str, Any]:
    """Retained for audit only; not a valid R2 comparison and never a headline."""
    pair_prob, global_prob, prior_meta = fit_pair_prior()
    full_split = read_json(AUDIT / "train_calibration_manifest.json")
    full_fit_ids = set(map(str, full_split["fit_image_ids"]))
    full_cal_ids = set(map(str, full_split["calibration_image_ids"]))
    pilot_population = read_json(PILOT / "population_manifests/pilot_population_manifest.json")
    pilot_fit_ids = set(map(str, pilot_population["FIT"]["image_ids"]))
    pilot_cal_ids = set(map(str, pilot_population["CAL"]["image_ids"]))
    cal_partition = read_json(PILOT / "population_manifests/calfit_calcheck_manifest.json")
    calfit_ids = set(map(str, cal_partition["CAL-FIT_image_ids"]))
    calcheck_ids = set(map(str, cal_partition["CAL-CHECK_image_ids"]))
    if not pilot_fit_ids <= full_fit_ids or not pilot_cal_ids <= full_cal_ids:
        raise ValueError("bounded N4-FIT pilot images are not subsets of the registered full FIT/CAL split")
    if calfit_ids | calcheck_ids != pilot_cal_ids or calfit_ids & calcheck_ids:
        raise ValueError("CAL-FIT/CAL-CHECK do not exactly partition the frozen CAL pilot images")
    y_fit, rel_fit, log_o_fit, img_fit, support_fit, nimg_fit, _recovery_fit = load_pilot_rows("CAL-FIT", pair_prob, global_prob)
    y_check, rel_check, log_o_check, img_check, support_check, nimg_check, recovery_check = load_pilot_rows("CAL-CHECK", pair_prob, global_prob)
    if set(img_fit) != calfit_ids or set(img_check) != calcheck_ids:
        raise ValueError("saved prediction image identities differ from frozen CAL-FIT/CAL-CHECK manifests")
    if len(y_fit) != int(cal_partition["CAL-FIT_relation_rows"]) or len(y_check) != int(cal_partition["CAL-CHECK_relation_rows"]):
        raise ValueError("saved prediction row counts differ from frozen CAL-FIT/CAL-CHECK manifest")
    log_rel_fit = torch.log_softmax(torch.as_tensor(rel_fit, dtype=torch.float64), dim=1).numpy()
    log_rel_check = torch.log_softmax(torch.as_tensor(rel_check, dtype=torch.float64), dim=1).numpy()
    alpha = fit_positive_alpha(log_o_fit, log_rel_fit, y_fit)
    base = metrics(log_o_check, y_check)
    nested_logits = log_o_check + alpha * log_rel_check
    nested = metrics(nested_logits, y_check)
    losses_base = F.cross_entropy(torch.as_tensor(log_o_check), torch.as_tensor(y_check), reduction="none").numpy()
    losses_nested = F.cross_entropy(torch.as_tensor(nested_logits), torch.as_tensor(y_check), reduction="none").numpy()
    bootstrap = image_cluster_bootstrap(losses_nested - losses_base, img_check)
    return {
        "schema": "paper-c-r2-nested-object-pair-residual-v1",
        "status": "COMPLETED_CPU_IDENTITY_CHECKED",
        "estimand": "held-out predictive contribution of a frozen rel_feat log-probability offset conditional on a FIT-only ordered-pair prior with its logits fixed",
        "formula": "z_O = log P_FIT(y | ordered subject/object class); z_nested = z_O + alpha * log_softmax(frozen rel_feat logits)",
        "split": {"FIT_images": prior_meta["fit_images"], "FIT_rows": prior_meta["fit_rows"],
                  "CAL-FIT_images": nimg_fit, "CAL-FIT_rows": len(y_fit),
                  "CAL-CHECK_images": nimg_check, "CAL-CHECK_rows": len(y_check),
                  "CAL-FIT_CAL-CHECK_split_seed": cal_partition["seed"],
                  "CAL-FIT_CAL-CHECK_selection_rule": cal_partition["selection"],
                  "pilot_FIT_is_subset_of_registered_FIT": True,
                  "pilot_CAL_is_subset_of_registered_CAL": True,
                  "CAL-FIT_CAL-CHECK_exact_partition": True,
                  "CAL-FIT_CAL-CHECK_image_disjoint": not (set(img_fit) & set(img_check)),
                  "validation_outcomes_used": False, "E2_used": False},
        "prior": prior_meta,
        "residual": {"fitting_population": "CAL-FIT only", "parameter_count": 1,
                     "constraint": "alpha >= 0 via softplus", "optimizer": "same registered N4-FIT LBFGS scalar-alpha routine: lr=.25, max_iter=100, strong_wolfe, tolerance_grad=1e-10, tolerance_change=1e-12",
                     "alpha": alpha, "rel_feat_predictor_fitted_or_changed": False},
        "support": {"CAL-FIT_supported_pair_rows": int(support_fit.sum()), "CAL-FIT_unsupported_rows": int((~support_fit).sum()),
                    "CAL-CHECK_supported_pair_rows": int(support_check.sum()), "CAL-CHECK_unsupported_rows": int((~support_check).sum()),
                    "support_definition": "ordered normalized first subject/object object labels observed in FIT prior table"},
        "saved_CALCHECK_relfeat_recovery": {"method": "invert saved raw combined logits using CAL-FIT-only alpha and relfeat temperature; reconstruct logits and compare", "max_abs_reconstruction_error": recovery_check},
        "CAL_CHECK": {"O_only": base, "O_plus_relfeat": nested,
                      "delta_log_loss": nested["log_loss"] - base["log_loss"],
                      "delta_accuracy": nested["accuracy"] - base["accuracy"],
                      "delta_macro_recall": nested["macro_recall_present_predicate_classes"] - base["macro_recall_present_predicate_classes"]},
        "bootstrap": bootstrap,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": "CPU"},
        "provenance": {str(p.relative_to(ROOT)): sha256(p) for p in [AUDIT / "train_calibration_manifest.json", CANONICAL, REP_MANIFEST,
             PILOT / "population_manifests/calfit_calcheck_manifest.json", PILOT / "calfit_predictions.json",
             PILOT / "calcheck_predictions.json", PILOT / "fit_selected_raw.jsonl", PILOT / "cal_selected_raw.jsonl",
             PILOT / "calibration_parameters.json", PILOT / "n4fit_training_config.json", PILOT / "pilot_results.json",
             AUDIT / "fullscale_residual_fit_models.pt"]},
        "limitations": ["bounded CAL-CHECK pilot, not full-population N4", "pair prior uses the larger registered full FIT population, while rel_feat is the frozen predictor used in the saved pilot", "CAL-CHECK rel_feat logits are algebraically recovered from the saved raw combined and N4 logits using calibration parameters fitted only on CAL-FIT; this is exact up to row-wise additive constants/float serialization", "one scalar residual coefficient does not test semantic specificity", "no final validation outcome was read"],
    }


def rank(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return out


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def run_r3() -> dict[str, Any]:
    results = {}
    for name, path in WPRD_PATHS.items():
        data = read_json(path)
        rows = data["rows"]
        if len(rows) != 12:
            raise ValueError(f"WPRD arm count changed in {path}")
        stats = {}
        for key, field in [("R@50", "R"), ("mR@50", "mR"), ("Pareto gap", "pareto")]:
            full_x, full_y = [float(r[field]) for r in rows], [float(r["wprd"]) for r in rows]
            loo = [spearman(full_x[:i] + full_x[i+1:], full_y[:i] + full_y[i+1:]) for i in range(12)]
            stats[key] = {"rho": data["spearman"][key]["rho"], "permutation_p": data["spearman"][key]["p_perm"],
                          "leave_one_arm_out_rho_min": min(loo), "leave_one_arm_out_rho_max": max(loo),
                          "leave_one_arm_out_rho": loo}
        # Three treatment-family means are descriptive only, not independent replications.
        groups = [rows[:2], rows[2:7], rows[7:12]]
        group_means = [{field: float(np.mean([r[field] for r in group])) for field in ["R", "mR", "pareto", "wprd"]} for group in groups]
        family_rho = {"R@50": spearman([g["R"] for g in group_means], [g["wprd"] for g in group_means]),
                      "mR@50": spearman([g["mR"] for g in group_means], [g["wprd"] for g in group_means]),
                      "Pareto gap": spearman([g["pareto"] for g in group_means], [g["wprd"] for g in group_means])}
        results[name] = {"source": str(path.relative_to(ROOT)), "sha256": sha256(path), "n_arms": 12,
                         "documented_population": WPRD_POPULATIONS[name],
                         "stored_correlations": stats, "descriptive_three_family_means": group_means,
                         "three_family_rank_correlations_not_inferential": family_rho}
    return {
        "schema": "paper-c-r3-wprd-construct-validity-audit-v1",
        "status": "COMPLETED_DESCRIPTIVE_DEPENDENCE_AUDIT",
        "results": results,
        "design_facts": {"unit": "12 deterministic scoring arms derived from one checkpoint/cache, evaluated on shared image-relation cache", "independent_model_draws": 1,
                         "repeated_treatment_families": {"prior_and_random_null": 2, "PURE_text_to_classifier_sweep": 5, "geometry_including_fusion": 5},
                         "within_arm_R_mR_WPRD_filtering": "same 12 saved arm rows in each corr.json; no missing metric fields",
                         "permutation_method": "historical script permutes arm ranks over 2,000 draws, seed 0; plus-one p-value; does not account for arm-family dependence",
                         "pair_prior_WPRD": {name: next(r["wprd"] for r in read_json(path)["rows"] if r["arm"] == "pair prior only") for name, path in WPRD_PATHS.items()},
                         "random_null_WPRD": {name: next(r["wprd"] for r in read_json(path)["rows"] if r["arm"] == "random null") for name, path in WPRD_PATHS.items()}},
        "conclusion": "WPRD has descriptive responsiveness to score changes and is exactly 0.5 for the pair-prior arm in these artifacts, but correlation p-values treat highly dependent arms as exchangeable observations. These data do not validate WPRD as a general construct distinct from SGG quality.",
        "missing": ["independent model/checkpoint families", "cluster-aware inference over independently trained models", "pre-registered construct validity criterion", "a controlled planted relation signal and matched shortcut/null system", "complete cell-level WPRD bootstrap artifacts for the correlation population"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Do not allow this historical entrypoint to overwrite the corrected
    # R2 result with its former uncalibrated O-only comparison.
    from tools.r2_nested_correction import main as corrected_r2_main
    corrected_r2_main()
    r3 = run_r3()
    (OUT / "R3_WPRD_construct_validity.json").write_text(json.dumps(r3, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    hash_targets = [path for path in OUT.rglob("*") if path.is_file() and path.name != "hashes.json"]
    hash_targets.extend([ROOT / "tools/portfolio_gate_analysis.py", ROOT / "tests/test_portfolio_gate_analysis.py"])
    hashed = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(hash_targets)}
    (OUT / "hashes.json").write_text(json.dumps({"schema": "portfolio-gate-artifact-hashes-v1", "files": hashed}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    r2 = json.loads((OUT / "R2_nested_nuisance_analysis.json").read_text(encoding="utf-8"))
    print(json.dumps({"R2_G_plus_O_delta_log_loss": r2["G_plus_O"]["G+O"]["delta"]["log_loss"],
                      "R2_O_delta_log_loss": r2["G_plus_O"]["O"]["delta"]["log_loss"],
                      "R3": {k: v["stored_correlations"] for k, v in r3["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
