"""Build an additive Paper C manuscript bundle from frozen saved evidence.

This tool reads JSON/NPZ evidence, independently checks saved predictions, and
renders tables/figures and fixed analysis text. It has no model fitting,
inference, dataset sampling, CUDA, or network path. Existing outputs are refused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASELINE = "7d6eaf2098ad601d4d573d374f99e0feccc03252"
AUDIT = "runs/paper_c_final_evidence_audit_20261003T185925Z"
SOURCES = {
    "ladder": "runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/result.json",
    "residual": "runs/paper_c_nuisance_ladder_v2_20261003T092837Z/nuisance_ladder_v2_results.json",
    "full": "runs/paper_c_fullscale_residual_audit_20261003T084403Z/fullscale_residual_audit.json",
    "full_predictions": "runs/paper_c_fullscale_residual_audit_20261003T084403Z/fullscale_residual_row_predictions.npz",
    "n4": "runs/paper_c_n4fit_pilot_training_20261003T182118Z/pilot_results.json",
    "n4_config": "runs/paper_c_n4fit_pilot_training_20261003T182118Z/n4fit_training_config.json",
    "n4_predictions": "runs/paper_c_n4fit_pilot_training_20261003T182118Z/calcheck_predictions.npz",
    "n4_bootstrap": "runs/paper_c_n4fit_pilot_training_20261003T182118Z/pilot_bootstrap.json",
    "n4_decision": "runs/paper_c_n4fit_pilot_training_20261003T182118Z/fullscale_decision.md",
    "bridge": "runs/paper_c_same_validation_checkpoint_rescore_20261002T082100Z/same_validation_checkpoint_rescoring.json",
    "seed": "runs/l4_breakthrough_20260930/seed_stability/fit_retry1/result.json",
    "seed_similarity": "runs/l4_breakthrough_20260930/seed_stability/representation_metrics.json",
    "smoke": "runs/paper_c_matched_supervision_smoke_20261002T111542Z/smoke_result_forensics.json",
    "smoke_audit": "runs/paper_c_matched_supervision_smoke_20261002T111542Z/c1_smoke_final_audit.json",
    "readout": "runs/eval_readout_v2_R2c_20260911T023722Z/result.json",
    "vocab": "runs/paper_c_nuisance_vocab_contract_20261003T114634Z/vocabulary_provenance.json",
    "geometry": "docs/GEOMETRY_INPUT_DEGENERACY_RESULT.md",
    "fourier": "docs/FOURIER_NULL_AND_INVERTIBILITY_RESULT.md",
    "predicate_disjoint": "docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md",
    "e2_manifest": "runs/e2_development_20260912/FROZEN_CANDIDATE_MANIFEST.json",
    "e2_schema": "runs/e2_development_20260912/annotation_schema.json",
    "e2_design": "docs/PAPER_C_E2_POPULATION_REDESIGN_AMENDMENT_2026-09-12.md",
    "e2_human": "docs/E2_HUMAN_STATISTICAL_FIDELITY_AUDIT_20260915.md",
    "model_source": "runs/paper_c_canonical_train_relfeat_20260930T062248Z/code/openvocab_rel/models/relational_model.py",
    "train_source": "runs/paper_c_canonical_train_relfeat_20260930T062248Z/code/openvocab_rel/train.py",
    "audit_metrics": AUDIT + "/posthoc_metrics.json",
    "audit_index": AUDIT + "/evidence_index.json",
    "audit_status": AUDIT + "/final_status.json",
    "c1_result": "runs/eval_C1/result.json",
    "c1_preregistration": "docs/PAPER_C_C0_C1_PREREGISTRATION.md",
    "c1_result_report": "docs/PAPER_C_C1_RESULT.md",
    "c1_seed2_report": "docs/PAPER_C_C1_SEED2_RESULT.md",
    "conditional": "runs/paper_c_decision_pilot_20261001T123842Z/conditional_pilot_20261001T171120Z/conditional_pilot_results.json",
    "n4_training_source": "runs/paper_c_n4fit_pilot_training_20261003T182118Z/n4fit_pilot.py",
}
IMMUTABLE = {
    "checkpoints/C1_seed1234.pt": "79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854",
    "datasets_vg150_clean/train.jsonl": "306fc0db96ca716bbc78c2cf32d49fdc695ab0074004a99c7399c850a1c605a4",
    "runs/canonical_train_representation_full_20261001T020630Z/representation/canonical_train_relfeat_manifest.json": "f70b819eb22a40b431f32280fd6b04d74caeb821fc37764fdb288d6abdb9c8d4",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_close(actual: float, expected: float, name: str, tol: float = 1e-7) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tol:
        raise ValueError(f"Evidence mismatch for {name}: {actual} versus {expected}")


def saved_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets)
    if x.ndim != 2 or x.shape[1] != 51 or y.shape != (len(x),) or len(x) == 0:
        raise ValueError("Expected nonempty aligned [rows,51] logits and targets")
    if not np.isfinite(x).all() or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= 51)):
        raise ValueError("Nonfinite logits or invalid predicate targets")
    shifted = x - x.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]
    prediction = x.argmax(axis=1)
    present = [c for c in np.unique(y) if c < 50]
    return {
        "rows": len(y), "log_loss": float(losses.mean()),
        "accuracy": float((prediction == y).mean()),
        "macro_recall": float(np.mean([(prediction[y == c] == c).mean() for c in present])),
        "classes_present": len(present), "finite": True,
    }


def check_identity(z: dict, images: int, rows: int) -> dict:
    if len(z["targets"]) != rows or len(set(z["image_id"].tolist())) != images:
        raise ValueError("Saved prediction population differs from frozen population")
    keys = list(zip(z["image_id"].tolist(), z["subject_index"].tolist(),
                    z["object_index"].tolist(), z["predicate"].tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate labeled pair identity")
    return {"images": images, "rows": rows, "unique_labeled_pair_keys": len(set(keys)), "duplicates": 0}


def load(repo: Path, key: str) -> dict:
    return json.loads((repo / SOURCES[key]).read_text())


def verify_evidence(repo: Path) -> dict:
    for key, rel in SOURCES.items():
        if not (repo / rel).is_file():
            raise FileNotFoundError(f"Missing source {key}: {rel}")
    if load(repo, "audit_status")["final_status"] != "PAPER_C_FINAL_EVIDENCE_AUDIT_COMPLETE":
        raise ValueError("Prior evidence audit is not complete")
    ladder = load(repo, "ladder")
    expected = {"A1_frozen_baseline": .5749881522134409, "A2_linear": .5801279465080909,
                "A3_mlp": .5856688775586617, "A4_cosine_recomputed": .5758335904938229,
                "A5a_geometry_xfit": .5881443828571307, "A5b_geometry_trainfit": .5962347576211017,
                "A6_fusion": .585503429510664, "N1_shuffled_label_null": .5054514476497997,
                "N2_prior_control": .5}
    for name, value in expected.items():
        require_close(ladder["arms"][name]["wprd_macro"], value, name, 1e-12)
    require_close(load(repo, "c1_result")["wprd_macro"], expected["A1_frozen_baseline"], "C1 registered WPRD", 1e-12)
    conditional = load(repo, "conditional")
    require_close(conditional["N4"]["log_loss"], 3.094027228103338, "historical conditional N4", 1e-12)
    require_close(conditional["N5"]["log_loss"], 2.742337422316174, "historical conditional N5", 1e-12)
    residual = load(repo, "residual")
    expected_effects = [(-.36603313227755896, -.3966375880120899, -.33616490324423837),
                        (-.005128832905433961, -.026359921659650332, .01586280669203568)]
    if len(residual["paired_image_cluster_bootstrap"]) != len(expected_effects):
        raise ValueError("Stored residual contrast count changed")
    for effect, expected_effect in zip(residual["paired_image_cluster_bootstrap"], expected_effects):
        for field, value in zip(("estimate_nats_per_row", "ci_low_percentile", "ci_high_percentile"), expected_effect):
            require_close(effect[field], value, "residual bootstrap " + field, 1e-12)
        if effect["seed"] != 20261003 or effect["bootstrap_replicates"] != 2000:
            raise ValueError("Registered residual bootstrap protocol changed")
    pilot_bootstrap = load(repo, "n4_bootstrap")
    if pilot_bootstrap["seed"] != 20261003 or pilot_bootstrap["replicates"] != 2000:
        raise ValueError("Registered pilot bootstrap protocol changed")
    draws = np.asarray(pilot_bootstrap["replicate_deltas"], dtype=np.float64)
    if draws.shape != (2000,) or not np.isfinite(draws).all():
        raise ValueError("Invalid saved pilot bootstrap draws")
    for actual_ci, stored_ci in zip(np.percentile(draws, [2.5, 97.5]), pilot_bootstrap["ci95_percentile"]):
        require_close(actual_ci, stored_ci, "saved pilot bootstrap percentile", 1e-12)
    old = load(repo, "audit_metrics")
    verified = {}
    with np.load(repo / SOURCES["full_predictions"], allow_pickle=False) as z:
        verified["validation"] = check_identity(z, 1182, 14991)
        verified["validation"]["pair_supported_rows"] = int(z["pair_supported"].sum())
        if verified["validation"]["pair_supported_rows"] != 12330:
            raise ValueError("Pair support count changed")
        verified["validation"]["metrics"] = {}
        for model, field in {"O": "pair_prior_calibrated_logits", "G+O": "GO_calibrated_logits",
                             "rel_feat": "relfeat_M2_calibrated_logits", "G+O+rel_feat": "residual_offset_calibrated_logits"}.items():
            m = saved_metrics(z[field], z["targets"])
            for metric in ("log_loss", "accuracy", "macro_recall"):
                require_close(m[metric], old["validation"]["metrics"][model][metric], model + metric, 1e-10)
            verified["validation"]["metrics"][model] = m
    n4 = load(repo, "n4")
    with np.load(repo / SOURCES["n4_predictions"], allow_pickle=False) as z:
        verified["N4FIT_CAL_CHECK"] = check_identity(z, 250, 3142)
        verified["N4FIT_CAL_CHECK"]["metrics"] = {}
        for model, field in {"N4_calibrated": "N4_calibrated_logits", "N4_rel_calibrated": "combined_calibrated_logits",
                             "N4_raw": "N4_raw_logits", "N4_rel_raw": "combined_raw_logits"}.items():
            m = saved_metrics(z[field], z["targets"])
            for metric in ("log_loss", "accuracy", "macro_recall"):
                require_close(m[metric], n4["CAL_CHECK"][model][metric], model + metric, 1e-10)
            verified["N4FIT_CAL_CHECK"]["metrics"][model] = m
    actual = {rel: sha256(repo / rel) for rel in IMMUTABLE}
    if actual != IMMUTABLE:
        raise ValueError("Immutable artifact hash mismatch")
    snapshot = repo / "runs/paper_c_canonical_train_relfeat_20260930T062248Z/code"
    commit = subprocess.check_output(["git", "-C", str(snapshot), "rev-parse", "HEAD"], text=True).strip()
    if commit != "ec4cca6ff01ab46929b7b428220fe0df2ba00f2a":
        raise ValueError("Canonical source commit changed")
    verified["immutable_hashes"] = actual
    verified["canonical_source_commit"] = commit
    verified["stored_bootstrap_checks"] = {"residual_contrasts": 2, "pilot_saved_draws": 2000,
                                            "new_resampling": False, "pilot_percentiles_match": True}
    verified["method"] = "Independent float64 log-sum-exp from saved 51-output logits; no fitting, inference, resampling or new split"
    return verified


SECTIONS = {
"abstract": """# Abstract

Predicate prediction from a scene-graph representation does not identify which visual or statistical cues support that prediction. We reconstruct a controlled research program on a frozen, predicate-supervised C1 representation and distinguish decodability from evidence for relation-specific content. On 20,016 historical within-pair discrimination cells, a GELU readout reached WPRD 0.585669 compared with 0.574988 for the frozen text readout, while geometry-only readouts reached 0.588144 (cross-fit) and 0.596235 (train-fit). On a separate fixed validation population of 1,182 images and 14,991 labeled rows, combining frozen rel_feat with an ordered-object-pair prior reduced calibrated log-loss by 0.366033 nats per row; after adding geometry the increment was 0.005129, with an image-cluster interval including zero. A separately held-out, leakage-safe N4-FIT feasibility pilot incorporating object identity, geometry, CLIP crops and global context improved log-loss by 0.002446, with a pilot-only interval below zero, but did not meet its registered resource criterion for fullscale reconstruction. These contrasts differ in population and offset construction and are not a strictly nested causal ladder. Architecture and provenance audits identify substantial nuisance pathways and prevent several tempting cross-experiment comparisons. The evidence establishes predictive decodability and utility relative to specified baselines, while full-population residual beyond appearance/context and semantic relational discrimination remain unresolved. We recommend closing the present predictive hypothesis and making human-verified controlled evaluation a separately registered project before further model development. [E1–E8]
""",
"introduction": """# Introduction

Scene-graph predicate labels combine spatial, object, appearance and context regularities. A representation trained on those labels may support accurate prediction while primarily integrating those regularities. Paper C initially asked whether repairing the inherited geometry pathway improved C1 over C0 on within-pair relation discrimination. Its later operational question became whether a frozen predicate-supervised representation retained usable predictive structure after progressively stronger nuisance estimators were considered. These are related but distinct questions. A decoder improvement does not itself identify a semantic mechanism.

The program matters because the proposed constructive method—a residual relation expert trained against a nuisance offset—requires a residual worth targeting. Without that premise, another training campaign could optimize a weak or misidentified effect. The evidence therefore tests alternative explanations rather than treating any positive readout as proof of relational content: ordered object-pair frequency, geometry, decoder capacity, sampling composition, readout implementation, seed transfer, and appearance/context. The last alternative was tested only in a bounded pilot. [E1–E10]

This manuscript is an evidence reconstruction, not a claim of a new state-of-the-art method. Its contribution is the measured contraction of a particular predictive contrast, the disclosure of implementation and lineage constraints, and the distinction between a resource decision and a scientific null. No global novelty claim is certified by this repository-only review. Five findings deserve the main text: registered C0/C1 performance and its threshold miss; readout decodability with competitive geometry; the O and G+O held-out contrasts; the narrowly interpretable N4-FIT pilot; and the architecture/evaluation limits that prevent a semantic conclusion. Sampling, seed transfer, readout-v2, Fourier recovery and intervention smoke belong in the appendix.

The preferred explanation is that C1 integrates several predictive cues. The strongest alternative to that explanation is that useful image-conditioned relation structure exists but is poorly identified by the current nuisance estimators and labels. Current evidence cannot distinguish these fully. The most valuable next question concerns human-verified discrimination when object labels are held fixed and measured nuisance predictors are explicit competitors. [E7–E13]
""",
"methods": """# Methods

## Evidence and provenance

The baseline is closeout commit `7d6eaf2098ad601d4d573d374f99e0feccc03252`. The canonical source is `runs/paper_c_canonical_train_relfeat_20260930T062248Z/code` at `ec4cca6ff01ab46929b7b428220fe0df2ba00f2a`; checkpoint and train JSONL hashes are independently checked in `independent_verification.json`. The canonical representation has 83,249 images, 1,046,427 labeled rows, 17 shards, width 768 and float16 features. All original runs remain immutable. This synthesis recomputes probabilities from saved logits on CPU and renders evidence tables; it fits no predictive model. The evidence map records each hypothesis, population, treatment, comparator, metric, uncertainty, interpretation and limitation. [E0]

## Historical readout experiment

The ladder uses the accepted C1 dump: 132,556 relation rows, 10,401 images and 20,016 WPRD cells across 50 foreground predicates. Learned arms use five image-based folds; the prior control is exactly 0.5 and the shuffled null is near 0.5. A3 is 768→768→51 with GELU, seed 0, 25 epochs, learning rate 0.002, weight decay 0.0001 and batch 4096. A2 is standardized ridge; A5a uses 19 box features and cross-fit LBFGS. A5b is a train-fitted geometry estimator and is not identical to A5a's fitting population. WPRD contrasts resample cells as registered; overlapping images across cells limit their interpretation as independent-image uncertainty. The A3−A1 stored delta is 0.0106807253, CI [0.0044921035, 0.0167044730]. [E1]

## Current predictive residual experiment

An image-disjoint canonical-train split contains FIT 75,066 images / 943,815 rows and CAL 8,183 / 102,612. Current evaluation is 1,182 images / 14,991 rows with frozen E2 exclusions. O is FIT-only ordered subject/object label counts with Laplace(1) over 51 outputs and FIT-global backoff for unsupported pairs. Exactly 12,330 validation rows have FIT-supported pairs, and 2,661 back off. G+O is a FIT-trained linear correction using the registered 19-D geometry on top of the pair-prior offset. The rel_feat readout is registered M2 trained on the same full FIT with FIT-only standardization. Temperatures and combination coefficients are fitted on CAL only. [E2]

The full O/G+O combination is `z_rel + alpha * log_softmax(z_nuisance / T_nuisance)`, followed by one scalar combined temperature. Thus the saved comparison is a calibrated combined predictor against a nuisance predictor; alpha does not directly quantify the amount of relational information. Primary loss is mean negative log probability per row, in nats. Accuracy uses argmax; macro recall averages observed foreground classes. Image-cluster bootstrap resamples images with all row multiplicities and computes the row-weighted paired delta, 2,000 repetitions, seed 20261003. This preserves the row-average estimand, rather than silently switching to equal image weighting. These exploratory intervals condition on trained weights, calibration and the selected split; they do not measure training-seed uncertainty. [E2]

## N4-FIT feasibility experiment

Historical N4 used a 2,907-label vocabulary from an earlier 1,200-image train population; 122 of those images lie in current CAL. It cannot serve as a clean current CAL comparator. N4-FIT instead freezes 5,000 FIT images / 63,286 rows, a FIT-only 6,909-label vocabulary plus unknown, and an input width of 16,900. Inputs are geometry-8, subject and object one-hots, 768-D subject/object/union CLIP crop features and a 768-D global image vector. The new decoder is 16,900→768→51 GELU with 13,019,187 parameters. All historical visual components are preserved. [E3,E4]

N4-FIT uses AdamW, seed 0, lr .002, weight decay .0001, batch 4096, 25 epochs, unweighted mean multiclass CE, no scheduler/early stopping, and the fixed final epoch. Geometry and CLIP channel mean/sample-std are FIT-only; one-hots are unstandardized. CAL-FIT contains 750 images / 9,078 rows; CAL-CHECK has 250 / 3,142. N4 temperature, rel_feat temperature, nonnegative alpha and combined temperature are fitted only on CAL-FIT. The predictor is `z_N4 + alpha * log_softmax(z_rel / T_rel)` with an additional combined temperature. The rel_feat predictor comes from the earlier full FIT model and is not refitted, so its fitting scale differs from the pilot nuisance model. This is a second reason not to present the pilot as the next matched rung of the full O/G+O experiment. [E4]

Pilot bootstrap uses 2,000 image resamples, seed 20261003, percentile 95% interval. CAL-CHECK labels do not fit weights/calibration. Final validation is untouched by this pilot. The resource rule requires a stable negative increment materially greater than the earlier G+O reference magnitude; it is not a final statistical test of the full nuisance question. [E4]

## Diagnostic experiments

The M2 bridge freshly scores four frozen checkpoints on exact current validation identities using each run's saved scaler; it establishes common evaluation, not randomized sampling effects. FULL/A/B seed1234 smoke shares its training protocol. A zeros only geometry input to fusion_gate; B deranges incoming pair-state rows within the same image at both edge-layer interfaces, retaining target object/geometry/visual inputs and leaving singleton images unchanged. Its engineering GO cannot be promoted into confirmatory effect estimation. Cross-seed feature similarity and decoder transfer use a separate validation-held-out protocol; they cannot establish robustness of the residual model. [E5–E8]
""",
"results": """# Results

## Predictive decoding and geometry

Recorded WPRD values are A1 .574988, A2 .580128, A3 .585669, A4 .575834, A5a .588144, A5b .596235, A6 .585503, N1 .505451, N2 .500000. A3 exceeds A1 by .0106807; geometry cross-fit exceeds A3 by .0024755 and geometry train-fit by .0105659. These latter differences are arithmetic comparisons of recorded means; no new paired CI is assigned to them. A5b changes fitting population but still evaluates held-out historical validation; it must not be mislabeled an in-sample validation score. The registered C0→C1 change was +.008261, CI [.003831,.012697], below the registered +.010 success target. This supports decodability and competitive geometry, with a qualified architecture improvement, rather than the stronger original target. [E1,E13]

## Held-out residual prediction

On fixed validation, calibrated O LL is 1.6323624327; O+rel_feat is 1.2663293005; delta −.3660331323, image CI [−.3966375880,−.3361649032]. O accuracy/macro recall are .6458541792/.2000605986; combined .6404509372/.1666924060. A reduction in probability loss does not require an increase in argmax accuracy or equal-class recall. Geometry, appearance and scene context remain alternatives to the O residual. [E2]

G+O calibrated LL is 1.3242384157 and the rel_feat combination 1.3191095828; delta −.0051288329, CI [−.0263599217,+.0158628067]. Standalone matched-scale rel_feat is worse at 1.4205308987. G+O accuracy/macro recall are .6561270095/.1218934710; combination .6200386899/.1540253465. The raw comparison worsens, 1.3279949152→1.4190692328, while calibrated loss improves slightly. Thus this particular G+O combination is calibration-sensitive and does not improve argmax accuracy. Earlier statements that both raw loss and accuracy improved for this full contrast are contradicted by the final saved results; the statement holds only for the small N4 pilot. [E2]

The magnitude of the O-relative increment contracts by roughly 98.6% in the G+O comparison. This descriptive calculation is about these estimators and calibrated combinations; it is not a decomposition of causal or mutual information. The G+O interval permits both a modest benefit and a modest loss. No equivalence claim is warranted. [E2]

## Richer nuisance pilot

On CAL-CHECK, N4-FIT calibrated LL is 1.5984459578; its rel_feat combination is 1.5959997348. Delta is −.0024462230, exploratory CI [−.0026268349,−.0022835433]. Alpha is .0057614225; N4/combined temperatures are 2.5614361/2.5656223. Raw LL changes 2.4195268202→2.4193531598 (−.0001736604); accuracy .5566518141→.5582431572 (+.0015913431); macro recall .1386469772→.1395603580. The bootstrap interval excludes zero conditional on this pilot, but does not include seed, model-selection, or fullscale-population uncertainty. A small alpha is not a measure of information, because logit scale, calibration, and collinearity affect it. [E4]

The absolute pilot change is approximately 47.7% of the full G+O reference magnitude. That comparison triggered the recorded resource decision, `PILOT_DOES_NOT_SUPPORT_FULLSCALE`; it is not a statistically matched contraction estimate because the populations, offset orientation, and nuisance/rel_feat fitting scales differ. Full 83,249-image N4-FIT and constructive residual C1 training were not executed. The pilot cannot prove an absence of residual beyond U/S. [E4]

## Sampling, seed and readout diagnostics

The common-validation M2 results are LL 1.662453/1.684642/1.692843/1.685949 for 2.5k/5k/10k/15k-prefix, versus 1.630454 balanced15k. Balanced−prefix delta −.0554957, image CI [−.0824371,−.0307560], while accuracy and macro recall decrease by .0032019 and .0120571. Training rows differ (188,736 prefix versus 206,669 balanced), and sampling was not randomized across replicated fits; no clean scaling or generalization claim follows. [E5]

Cross-seed cosine .940568 and sampled linear CKA .960323 indicate feature similarity. Separate cross-seed decoder transfer loses .019870 and .018640 accuracy, with intervals below zero in both directions. The representation similarity therefore does not guarantee readout invariance. The matched residual and intervention studies have no second-seed replication. Corrected R2c WPRD .573407 is below R0 .574988; no paired R2c CI was found. Historical R2's near-zero gain is fidelity-invalid as a trained-prototype result. [E6,E8]

FULL/A/B smoke M2 validation LL are 1.830500/1.825392/1.875606; native-head LL are 2.697031/2.715586/2.809679. Registered spatial M2 LL are 1.614668/1.635027/1.681200. Those observations are consistent with sensitivity to the intended operations but cannot identify relation semantics, independent nuisance control, or causal effects. No matched seed5678 smoke or B2 was run. [E7]

## Error and population limits

Saved per-predicate results are rendered without new grouping or fitting. “on”, “in”, “has” and “of” dominate total loss largely through prevalence. Rare predicates can have only 4–30 rows, so tail deltas are unstable. No robust tail advantage or interaction-specific residual is established. The whitelist-member predicate union used in some smoke descriptive tables is not the frozen pairwise interactional endpoint; the current residual validation has zero eligible rows for that endpoint. [E0,E7,E10]
""",
"discussion": """# Discussion

The strongest conclusion is that rel_feat supports closed-set predictive decoding and complements an ordered-pair prior under the tested scoring protocol. Geometry is a strong competing predictor, and the full G+O increment is small and uncertain. The N4-FIT pilot indicates a small conditional benefit after its richer nuisance estimator, but does not identify semantic content or resolve the full nuisance question. The scientific lesson concerns identification: measured decoder utility depends on comparator strength, calibration, population, and architecture lineage. [E1–E4]

The best current hypothesis is that C1 is an integrator of object, visual, geometry and scene cues. Its source allows all these pathways into rel_feat. The best competing hypothesis is that useful relational discrimination survives those cues, but the current semantic evaluation and nuisance estimators do not identify it adequately. Predictive performance alone cannot decide between them. A neural nuisance expert is also fallible: failure to condition away a measured effect may reflect underfitting, vocabulary coverage, misspecified geometry, or calibration. Conversely, a strong nuisance fit does not mathematically remove semantic content. [E2,E3,E11]

Scientific significance and practical value should be separated. The pilot interval below zero indicates a conditional predictive difference on its population. Its .002446-nat change has no demonstrated deployment or semantic utility and did not satisfy the frozen spending criterion. There is no universal effect-size threshold established here. Calling the effect negligible in an absolute sense, or reporting an equivalence result, would require a new registered criterion. The paper can honestly report the effect, uncertainty and resource decision without either inflation or erasure. [E4]

The highest-value next question is evaluation: does a frozen visual model track human-verified relation truth in exact ordered-object-pair contrast blocks beyond explicit geometry and appearance/context competitors? Existing E2 development candidates offer a model-blind starting instrument, not gold labels. A human development phase can reveal whether the task is solvable, whether predicates co-occur rather than contrast, and whether nuisance-matched support exists. Architecture redesign before this evidence would change several things while leaving the semantic target unidentified. [E10–E12]

Paper C should close as a bounded predictive and reproducibility investigation. A future semantic study should be a separately registered Paper D. Its success would strengthen a narrower claim of controlled image-conditioned relation discrimination; it would not automatically establish causality, general semantic understanding, or unseen-predicate transfer. No new experiment is executed in this branch because the highest-value question lacks human gold and a powered final evaluation population. Existing CPU artifacts are used for the manuscript and independent consistency checks. [E10]
""",
"limitations": """# Limitations

1. **Semantic ground truth is missing.** Dataset predicate annotations and E2 candidate metadata are not adjudicated human truth. The current validation has zero eligible frozen interactional-family rows. Individual interactional-named predicates can occur without creating the paired semantic endpoint. [E10]
2. **The ladder is conceptual rather than strictly nested.** O/G+O use a full FIT/CAL validation protocol; N4-FIT uses a smaller FIT and CAL-CHECK, a new vocabulary/input width, a different combination orientation and a full-FIT frozen rel_feat predictor. Exact cross-level semantic effect attribution is unavailable. [E2–E4]
3. **Calibration affects the primary result.** G+O+rel_feat improves final loss slightly while raw loss and accuracy worsen. Alpha and temperature are not information measurements. [E2]
4. **Intervals are conditional.** Image-cluster bootstrap accounts for dependence among rows within sampled images; it does not include seed uncertainty, training instability, population selection or all research degrees of freedom. WPRD cell bootstrap is a different dependence unit and should not be relabeled image-cluster inference. Multiple exploratory branches preclude retroactive confirmatory claims. [E1,E2,E4]
5. **Full appearance/context conditioning is untested.** N4-FIT fullscale was not run. Its pilot CI cannot be extrapolated to a fullscale fit or treated as a null/equivalence result. [E4]
6. **Representation lineage is mixed.** Historical seed-transfer and conditional N4/N5 comparisons differ in fitting populations and vocabulary. Feature cosine/CKA do not establish predictive seed robustness. Old N4 is not admissible for current CAL. [E3,E6]
7. **Component interpretation is narrow.** No clean relation-only branch exists. A leaves geometry elsewhere; B disrupts pair-state alignment while preserving upstream contextual mixing. Smoke scores do not isolate a causal semantic mechanism. [E7,E11]
8. **Fourier evidence is probe-specific.** Poor recovery by a finite MLP does not prove mathematical information loss or noninvertibility. Historical strong wording is retained as provenance and qualified here. The units-contract defect itself is established by source arithmetic. [E9]
9. **Repository-only review cannot certify global novelty.** A clone lacks ignored model/data payloads. Reproduction needs the indexed external artifacts, and installed PyTorch 2.9.1 is below requirements' declared 2.10 minimum. [E0]
10. **No open-vocabulary evidence.** All 50 real predicates occur in original train/validation/test. C1's supervised heads/text CE see them; a pretrained text encoder does not make the experiment unseen-predicate generalization. [E12]
""",
"conclusion": """# Conclusion

Paper C establishes protocol-specific decodability and predictive utility beyond an ordered-object-pair prior. Adding geometry greatly reduces the measured increment; the remaining full G+O contrast is uncertain. A richer leakage-safe nuisance pilot yields a small benefit that does not justify its registered fullscale resource expense. These findings support a restrained account of nuisance-sensitive predictive representations. They leave semantic relation discrimination and full-population residual beyond appearance/context unresolved. The constructive residual-training hypothesis is untested rather than disproved. The current direction should remain closed; a separately registered, human-verified controlled evaluation has greater expected scientific value than another model or extraction run with the present labels.
""",
}

CENTRAL = """# Central scientific questions

| Question | Evidence-based answer |
| --- | --- |
| Q1 Decodable? | Yes, under the historical fixed vocabulary/readout protocols; A3 WPRD .585669. [E1] |
| Q2 Beyond pair prior? | Yes for the specified O comparator and calibrated held-out scoring, delta −.366033. Does not identify the source. [E2] |
| Q3 Geometry explanation? | Geometry is competitive, and the measured increment contracts to −.005129 with CI crossing zero. This is estimator-dependent attenuation, not a causal fraction. [E1,E2] |
| Q4 Richer nuisance residual? | A small N4-FIT pilot residual is observed; full-population residual is NOT ESTABLISHED. [E4] |
| Q5 Scientific size? | Pilot is distinguishable from zero under its conditional bootstrap, but its scientific mechanism and cross-population reproducibility are unknown. [E4] |
| Q6 Practical size? | No practical outcome benefit demonstrated; fails the registered compute-allocation rule. No universal negligible threshold is inferred. [E4] |
| Q7 Seed robustness? | Residual seed robustness NOT ESTABLISHED. Similarity and transfer studies are a different experiment; transfer declines. [E6] |
| Q8 Noun-pair strata? | Saved support strata are available, but no replicated full-nuisance residual by noun-pair group. 12,330 supported rows; unmatched pairs back off. [E2] |
| Q9 Predicate frequency? | Per-predicate outputs show heterogeneity and sparse tails, not a stable tail advantage. No outcome-driven bins are introduced. [E0] |
| Q10 Spatial concentration? | Geometry competition and smoke spatial results are suggestive; no confirmatory residual-family comparison. [E1,E7] |
| Q11 Interactional evidence? | The frozen paired endpoint has zero eligible current rows; BLOCKED_BY_MISSING_EVIDENCE. Whitelist membership is a different diagnostic. [E10] |
| Q12 Appearance explanation? | Remains viable. Pilot nuisance includes crop appearance but cannot rule it out in full population. [E4] |
| Q13 Scene explanation? | Remains viable. Global/context inputs enter rel_feat; pilot bundles appearance and context rather than isolating each. [E4,E11] |
| Q14 Nuisance integrator? | Plausible best current hypothesis; source supports every pathway. Not proven to be exclusively nuisance. [E11] |
| Q15 Priors cause decoding? | Priors contribute; O is insufficient for the tested calibrated comparison. Geometry/appearance/context combinations can explain the rest. [E2] |
| Q16 Decoder overfit? | Held-out prediction and nulls address simple leakage/overfit. One split/seed and comparator misspecification remain limitations. [E1,E2] |
| Q17 Sampling-sensitive? | Yes descriptively; common-validation M2 scores differ. No randomized sampling-effect conclusion. [E5] |
| Q18 Calibration-sensitive? | Yes, especially G+O combination: raw LL/accuracy worsen while calibrated LL slightly improves. [E2] |
| Q19 Metric-specific? | Yes, WPRD, NLL, accuracy and macro recall weight different phenomena and populations. [E1,E2,E5] |
| Q20 Distinguish relation content? | Human-verified contrasting relations under exact object labels and explicit geometry/appearance/context controls; then independent held-out transfer. Current labels alone cannot establish this. [E10] |
"""

REVIEW = """# Reviewer premortem

**Summary.** The submission audits predicate-supervised scene-graph representations using readouts, nuisance estimators, source/provenance checks and a bounded richer-nuisance pilot. The strongest result is predictive utility against object-pair priors, followed by pronounced contraction with geometry. The work does not yet identify semantic relation content.

**Strengths.** Transparent identity joins and immutable lineage; appropriate image clustering for current residuals; disclosure of invalid comparisons and stopped experiments; a held-out CAL-CHECK pilot; independently recomputable probabilities. The negative or bounded conclusions are potentially useful if the contribution is framed as an empirical audit.

**Weaknesses.** Heterogeneous populations/offset families; no full appearance/context comparison; one residual split/seed; no semantic endpoint; finite-probe geometry claims sometimes overstated historically; no demonstrated new method or global novelty. These are fatal for a semantic-method paper but not automatically fatal for a carefully scoped audit paper.

| Reviewer concern | Severity under a predictive-audit submission | Answer / needed evidence |
| --- | --- | --- |
| 1 Just pair prior? | MINOR | O combination improves held-out NLL substantially; residual source remains unidentified. Existing artifacts answer this restricted objection. [E2] |
| 2 Just geometry? | MAJOR | G+O accounts for most measured increment; CI includes zero. Current evidence supports geometry as a competing explanation. [E2] |
| 3 Just CLIP appearance? | MAJOR | Full U/S is absent; pilot insufficient for final exclusion. Requires full comparator or a new controlled evaluation. [E4] |
| 4 Just global context? | MAJOR | Context enters rel_feat and is bundled with crop features in pilot. Remains alive. [E4,E11] |
| 5 Why relational? | FATAL if semantic claim retained | Name denotes model output/function, not identified semantic content. Title and claims must use predictive scope. [E11] |
| 6 Full nuisance missing? | MAJOR | Historically blocked coverage, then resource-stopped below the frozen trigger. Disclose; do not extrapolate. [E3,E4] |
| 7 Pilot not fullscale? | MINOR for resource decision; MAJOR for residual claim | Resource choice is coherent; it does not test a full-population null. [E4] |
| 8 Enough power? | MAJOR | 250 independent image clusters; bootstrap narrowness is conditional, not power certification. No seed or population uncertainty. [E4] |
| 9 C1 not retrained? | NOT A REAL ISSUE for audit | Constructive hypothesis is outside completed evidence; retraining lacked justification, not refutation. [E4] |
| 10 Interactional evaluation absent? | FATAL for interaction-specific claim | Zero eligible paired endpoint; missing human gold. Requires a new evaluation. [E10] |
| 11 Open-vocabulary implication? | FATAL if retained | Every real predicate is supervised; no unseen test. Remove implication. [E12] |
| 12 Subset dependence? | MAJOR | M2 bridge demonstrates sample sensitivity, N4 pilot is smaller and frozen rel_feat has larger training scale. [E4,E5] |
| 13 Practical meaning? | MAJOR | Report nats/row and resource rule; no deployment or semantic benefit established. [E4] |
| 14 Dependence unit? | MINOR for residuals; MAJOR for WPRD interpretation | Current residuals cluster images. Historical WPRD intervals cluster cells, which can share images. No silent relabeling. [E1,E2] |
| 15 Shortcut encoding? | MAJOR | Entirely consistent with source and results. Explicitly retain as alternative. [E11] |
| 16 Falsification? | MAJOR but fixable in design | Frozen controlled semantic discrimination beyond nuisance with independent gold, or stable full-nuisance residual, would weaken shortcut-only prediction. [E10] |

**Technical concerns:** exact output vocabulary/background handling; source-specific geometry-8 versus geometry-19; scaler lineage; different offset orientations; historical N4 CAL contamination. Existing artifacts resolve identity/protocol details, not model adequacy. **Statistical concerns:** exploratory branching, clustered rows, confidence conditional on fitted models, lack of second seed, no equivalence margin. **Data concerns:** missing semantic labels, ambiguous/coexistent predicates, rare classes and exact-nuisance-match sparsity. **Novelty concerns:** no literature-wide claim justified; empirical audit must show reusable methodological insight. **Reproducibility concerns:** external payload availability and runtime-sensitive exactness, addressed by hashes and code snapshots but not a Git-only clone.

**Verdict:** Reject a submission claiming relational semantics or a validated residual method. Consider a scoped predictive/provenance audit after clear population/offset tables and claim revisions; venue suitability and novelty remain unassessed. The highest-value remedy is human-verified controlled evaluation. An expensive fullscale nuisance replay primarily refines a predictive question and is lower priority given the pilot and absent semantic endpoint. No extra model training is recommended solely to make the submission stronger.
"""

ARCH = """# Architecture, losses and predicate-disjoint feasibility

The immutable `relational_model.py` defines `ProgressiveEdgeConditionedLayer` near line 35, node grounding at 345, pair decoding at 431, `_forward_impl` at 660 and text-predicate scoring at 859. Grounding projects whole-image visual tokens and box queries, uses global bias and node attention/pooling. `_forward_impl` applies a two-layer `context_mixer` across object nodes around line 695 before selecting subject/object rows. Thus node appearance and other-object context already mix upstream of pair operations. Pair decoding combines subject/object features, Fourier geometry and fusion_gate, includes SPOA state when configured and `rel_seed`, performs two edge updates conditioned on incoming relation state, subject/object, visual tokens and geometry, then bilinear/output normalization. The final 768-D rel_feat has no clean relation-only branch. Each artifact's saved config determines which optional pathways execute. [E11]

The earliest explicit pair conditioning occurs in pair selection/subject-object fusion and geometry conditioning; SPOA's predicate-conditioned visual readout strengthens it. Edge layers receive pair-specific state and geometry as well as contextualized node features and image tokens. Object appearance, global scene structure, measured geometry and lexical supervision can all flow into the output. Train-source predicate CE consumes learned relation outputs; text predicate CE uses relation projections against text embeddings. Grounding, SPOA/counterfactual, relationness, KL/rank and configured auxiliary losses can also route gradients upstream; enabled terms/weights must be read from checkpoint/config, not assumed active merely because code exists. [E11]

A changes geometry conditioning at one fusion gate, leaving geometry elsewhere. B changes pair-state alignment at both edge interfaces, while preserving target unary/geometry/visual tensors and earlier context mixing. Objective formulas can stay fixed while gradients change. These diagnostic treatments do not identify exclusive semantic content. A distinct branch would need explicit separated inputs, declared information barriers, shared matched supervision and controls; even then separation by architecture would not establish semantic identification without controlled labels and evaluation. A stronger architecture claim is therefore not the first missing experiment. [E7,E11]

The existing predicate-disjoint audit establishes that all 50 real predicates occur in each original train/validation/test split; the 51st model placeholder `relation` is not a real predicate. Metadata lists three orphan labels (`around`, `growing on`, `says`) and cannot define the trained vocabulary. `wearing`/`wears` and other aliases require joint assignment. Current C1 has seen every evaluation predicate via predicate and text-predicate supervision, so rescoring it on a renamed held-out set is not unseen-predicate evaluation. [E12]

A future unseen-predicate protocol is technically designable: freeze alias-aware predicate groups before training; remove all held-out predicate supervision and text-conditioned relation losses; audit frequency priors, samplers, prompts, prototypes, metadata and checkpoint initialization; keep object splits and image boundaries explicit; separate development predicates from final held-out predicates; score frozen text embeddings without fitting unseen prototypes. Removing annotated relationships does not make visually present unseen concepts absent, so task-unseen must be distinguished from globally unseen. Distribution matching cannot be guaranteed when common or rare predicates are withheld; the historical audit shows severe imbalance. This requires a fresh model and protocol and does not win the current next-direction ranking. No split or implementation is created here.
"""

E2 = """# Evaluation-first protocol design (not executed)

**Selected target:** controlled image-conditioned relation discrimination on human-verified exact ordered-object-label contrast blocks, against explicit nuisance competitors. This is a proposed new Paper D, not a retrofit of Paper C's primary endpoint. [E10]

The frozen E2 development packet contains 45 contrast blocks / 90 images and a random 40-block / 80-image cohort, with six relation-pair strata: holding/looking-at, holding/using, holding/riding, holding/playing, carrying/riding, riding/using. Selection was model-blind. Metadata labels only propose candidates. The schema is `annotation_pending` and requires three independent annotators; truth is recorded separately for each relation, with both-valid/neither/uncertain and direction/visibility handling. No human success or model score can be inferred before those files exist. Existing 170-image exclusion, later 18-image current-validation exclusion and candidate-development membership are different contracts and must not be merged by name. [E10]

**Instrument sufficiency:** the schema can measure whether one of two descriptions is unambiguously image-supported and whether the highlighted subject/object direction is correct. It cannot by itself identify full relational semantics or eliminate appearance/global cues. Exact object labels, geometry covariates, independent nuisance predictions, crop/global context controls, an independent frozen visual positive control, no image reuse and a development/final firewall remain required. Tight geometry matching was historically too sparse; the existing amendment treats geometry as a comparator rather than claiming its elimination. [E10]

**Annotators:** three are required under the frozen E2 contract. The logical need is independent disagreement assessment, not a magic number. One human plus an independent adjudicator may check feasibility under a new amendment, but cannot reproduce the three-annotator primary agreement gate or estimate its agreement uncertainty. Single-person self-adjudication is not independent. Recommended next human development retains the existing three independent roles. No annotation is invented or inferred from a model.

**AI preannotation:** do not expose model labels, confidence, rationales or suggested answer to annotators. AI assistance can prepare presentation or flag missing assets only under a declared process and blinded review; visible answers can anchor judgments and create circular gold. Randomize description order and display image/subject/object highlights consistently. Record uncertainty and multi-valid truth rather than forcing a binary answer.

**Development and sample size:** reuse only the existing development packet to measure yield, solvability, ambiguity, per-stratum support and paired discordance. It is not the final test. Final sample size is BLOCKED_BY_MISSING_HUMAN_DATA, not guessed as 200 or 500. For a paired binary block difference, an initial planning approximation is N ≈ (z[1−a/2]+z[1−b])²·(p01+p10−delta²)/delta², followed by simulation under development-only discordance and image/block constraints. Freeze the smallest scientifically meaningful delta and error rates before final model scores. Independent image choices give chance .25 for both-correct if binary; forced bijective assignment gives chance .5. The interface/tie/uncertain rule must determine which, not a convenient retrospective choice. The current block task allows separate choices, so report the exact decision contract.

**Frozen final design proposal:** exact ordered labels, distinct images, no image reuse, whitelist strata and quotas selected model-blind; primary inclusion requires visible roles, unambiguous separate truth, one-valid-only contrast, direction agreement and independent adjudication. Maintain a held-out final image manifest excluded from every fitting/calibration operation. Freeze geometry/object/appearance-context competitors on non-final populations and an independent visual positive control before focal-model scoring. Primary endpoint is both-correct block accuracy; primary contrast is focal visual model minus the strongest registered nuisance competitor, paired by block. Uncertainty resamples independent blocks (or connected image groups if reuse accidentally occurs); use the existing E2 proposed 5,000 replicates/seed11 only if adopted prospectively. No focal-score-selected examples or post-hoc semantic families.

**Stopping rule:** stop before model inference if annotations are absent, human solvability fails its prospectively selected threshold, exact-pair support cannot meet development-derived precision/power, assets/identity are incomplete, or nuisance/positive controls cannot be frozen. A negative accepted-final comparison is scientifically informative; a failed human solvability gate diagnoses the instrument, not the model. GPU cost for the present task is zero. A later bounded frozen inference screen can be capped by measured throughput, but no runtime estimate is invented here. Annotation cost requires actual task timing from development.

**Interpretation matrix:** positive focal-minus-nuisance effect on human gold supports controlled relation discrimination on that population; zero/negative with a successful independent visual control weakens usefulness of current C1 for that semantic task; all models fail while humans pass diagnoses model/protocol limits; humans fail or candidates are multi-valid blocks model conclusions; insufficient exact-pair support means redesign is a new protocol, not rescue by score-based selection. None establishes causal information or globally unseen concepts.
"""

SELECTION = """# Highest-value next question

**Question:** Does frozen C1 distinguish human-verified contrasting relation truth across images with identical ordered object labels, beyond frozen geometry and appearance/context competitors?

**Why selected:** the largest gap is semantic identification, not another score on the current annotations. O utility and geometry competition are already measured; fullscale N4 would sharpen a small predictive residual while retaining ambiguous labels. Human-controlled evaluation can change interpretation whether positive or negative. Existing model-blind E2 development candidates make instrument validation feasible, but no human gold currently exists. [E2,E4,E10]

**Hypothesis:** on accepted independent contrast blocks, a frozen focal model has higher both-correct accuracy than the strongest predeclared nuisance competitor. **Null:** no positive paired advantage. The claim is bounded controlled image-conditioned discrimination, not global semantics. Final effect/precision thresholds and sample size must be fixed from human development, not from focal-model results.

**Decision tree:** human gold absent → design only; development not solvable/support too small → revise evaluation in a new registration or stop; development passes and all controls frozen → freeze an untouched final population; positive final difference with adequate uncertainty/positive-control behavior → evidence for bounded discrimination and consider a new representation objective; negative difference with a successful positive control → current C1 fails the targeted task and further residual training needs a new mechanism; all controls fail → task/model interface inconclusive.

**Rejected now:** fullscale N4 (small resource-trigger failure; still only predictive), residual C1 (premise not identified), seed5678/B2 (implementation replication and nuisance specificity cannot supply semantic truth), architecture isolation (expensive and evaluation remains ambiguous), predicate-disjoint training (large distribution/alias/supervision changes before semantic ground truth), additional decoder sweeps (low decision value). Cross-dataset and synthetic tests are useful later if a controlled semantic target is credible.

**Cost and stop:** this task uses zero GPU-hours. Human development and adjudication are the binding prerequisites; no timing or final N is fabricated. No experiment executes now. Stop before inference until gold, human gate, final sample-size plan, nuisance controls, independent positive visual control and frozen final manifest exist. See `semantic_evaluation_protocol.md`.

**Execution decision:** `NO_NEW_EXPERIMENT_JUSTIFIED_WITH_CURRENT_ASSETS`. This does not reject all future research; it rejects another autonomous compute campaign on present labels. Paper C remains closed; a separately registered Paper D is the recommended path if human development passes.
"""

POSITION = """# Final scientific position

1. **Best hypothesis:** C1 integrates useful object, geometry, appearance and context cues; decodability does not identify their semantic role.
2. **Strongest support:** held-out O/G+O comparisons and direct source data flow; geometry-only readouts are competitive. [E1,E2,E11]
3. **Against an exclusively nuisance claim:** substantial utility beyond O and a small N4-FIT CAL-CHECK increment; neither can prove nuisance exhaustiveness. [E2,E4]
4. **Strongest alternative:** relation-specific image structure may be present but hidden by imperfect nuisance estimators, annotation ambiguity and unsuitable evaluation. This remains alive.
5. **Uncertainty:** fullscale U/S residual, seed robustness, semantic/interactional discrimination and unseen-predicate generalization remain unestablished.
6. **What the repository establishes:** saved-model decodability, restricted held-out predictive comparisons, sensitivity to sampling/calibration and verified narrow interventions.
7. **What it cannot establish:** semanticity, causality, exclusive relation information, full-population G+O+U+S residual or residual-trained C1 benefit.
8. **Highest-value next experiment:** human-verified exact-object-pair contrast evaluation with explicit nuisance and independent visual controls.
9. **Executed:** no. Missing human gold, development yield/discordance and final population prevent a defensible run. No GPU was used.
10. **New result:** independent CPU recomputation matches saved probabilities and identities; no new learned model or scientific endpoint is introduced.
11. **Updated conclusion:** Paper C's predictive evidence is useful but does not justify another month of representation training on the present target. The evidence supports bounded closure, not a proof that relational learning is impossible.
12. **Disposition:** `SPLIT INTO NEW PAPER` for the proposed evaluation-first question; Paper C itself stays `CLOSE`. Continuing the same decoder/nuisance diagnostic loop is not recommended.
"""

# Scores are PI judgments conditional on repository assets, not estimated probabilities
# or evidence of literature-wide novelty. HIGH risk means more likely inconclusive.
DIRECTIONS = [
    (1,"A","Human-verified controlled semantic evaluation","HIGH","MEDIUM","MEDIUM","LOW","HIGH","HIGH","MEDIUM","HIGH","HIGH","HIGH"),
    (2,"M","Object-pair-matched relation evaluation with human gold","HIGH","MEDIUM","MEDIUM","LOW","HIGH","HIGH","MEDIUM","HIGH","HIGH","HIGH"),
    (3,"O","Controlled synthetic relation evaluation","HIGH","MEDIUM","MEDIUM","LOW","HIGH","MEDIUM","MEDIUM","HIGH","HIGH","MEDIUM"),
    (4,"B","Interactional-focused human evaluation","HIGH","MEDIUM","MEDIUM","LOW","HIGH","HIGH","HIGH","HIGH","HIGH","HIGH"),
    (5,"L","Cross-dataset human-validated evaluation","HIGH","MEDIUM","LOW","MEDIUM","HIGH","HIGH","HIGH","HIGH","HIGH","HIGH"),
    (6,"H","Geometry-controlled relation evaluation","HIGH","MEDIUM","MEDIUM","LOW","HIGH","MEDIUM","HIGH","HIGH","MEDIUM","MEDIUM"),
    (7,"I","Appearance-controlled relation evaluation","HIGH","HIGH","LOW","MEDIUM","HIGH","HIGH","HIGH","HIGH","HIGH","HIGH"),
    (8,"J","Scene-context-controlled relation evaluation","HIGH","HIGH","LOW","MEDIUM","HIGH","HIGH","HIGH","HIGH","HIGH","HIGH"),
    (9,"K","Counterfactual data intervention","HIGH","HIGH","LOW","MEDIUM","HIGH","HIGH","HIGH","HIGH","HIGH","HIGH"),
    (10,"C","Predicate-disjoint generalization","HIGH","MEDIUM","LOW","HIGH","HIGH","MEDIUM","HIGH","MEDIUM","MEDIUM","HIGH"),
    (11,"D","Genuinely unseen-predicate text protocol","HIGH","MEDIUM","LOW","HIGH","HIGH","HIGH","HIGH","HIGH","HIGH","HIGH"),
    (12,"F","Explicit nuisance conditioning/full comparator","MEDIUM","LOW","HIGH","MEDIUM","LOW","LOW","MEDIUM","MEDIUM","MEDIUM","MEDIUM"),
    (13,"E","Residual relation expert training","MEDIUM","MEDIUM","MEDIUM","HIGH","MEDIUM","LOW","HIGH","LOW","MEDIUM","MEDIUM"),
    (14,"N","Predicate-frequency matched analysis","MEDIUM","LOW","HIGH","LOW","LOW","LOW","MEDIUM","LOW","LOW","LOW"),
    (15,"G","Representation disentanglement","MEDIUM","MEDIUM","LOW","HIGH","HIGH","MEDIUM","HIGH","LOW","LOW","MEDIUM"),
    (16,"P","Architectural relation-isolated branch","MEDIUM","MEDIUM","LOW","HIGH","HIGH","MEDIUM","HIGH","LOW","LOW","MEDIUM"),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_svg_whitespace(path: Path) -> None:
    """Remove trailing spaces emitted by Matplotlib without changing SVG data."""
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def validate_destinations(repo: Path, out: Path, package: Path, refresh_generated: bool) -> None:
    if out.exists() or package.exists():
        if not refresh_generated:
            raise FileExistsError("Additive build refuses an existing analysis or paper package")
        status = json.loads((out / "final_status.json").read_text())
        if status.get("baseline_commit") != BASELINE or not status.get("analysis_complete"):
            raise ValueError("Refresh accepts only this builder's own generated analysis")
        old = json.loads((out / "hashes.json").read_text())["generated_outputs"]
        for rel, digest in old.items():
            p = repo / rel
            if not (p.is_relative_to(out) or p.is_relative_to(package)) or sha256(p) != digest:
                raise ValueError(f"Refuse refresh of changed or foreign output: {rel}")
        expected = set(old) | {str((out / "hashes.json").relative_to(repo))}
        observed = {str(p.relative_to(repo)) for root in (out, package) for p in root.rglob("*") if p.is_file()}
        if observed != expected:
            raise ValueError("Refuse refresh with unrecognized output files")


def build(repo: Path, out: Path, package: Path, refresh_generated: bool = False) -> None:
    validate_destinations(repo, out, package, refresh_generated)
    verified = verify_evidence(repo)
    created = datetime.now(timezone.utc).isoformat()
    source_hashes = {name: {"path": rel, "sha256": sha256(repo / rel)} for name, rel in SOURCES.items()}
    out.mkdir(parents=True, exist_ok=refresh_generated)
    package.mkdir(parents=True, exist_ok=refresh_generated)
    tables = package / "TABLES"; tables.mkdir(exist_ok=refresh_generated)
    figures = package / "FIGURES"; figures.mkdir(exist_ok=refresh_generated)
    ladder = load(repo, "ladder"); residual = load(repo, "residual"); n4 = load(repo, "n4")
    evidence = [
        ("E0","Artifact/probability identity", "C", "1182/14991 validation; 250/3142 CAL-CHECK", "saved logits and canonical artifacts", "recorded metrics/hash values", "exact recomputation", "no new inference", "audit_metrics"),
        ("E1","Decodability and geometry competition", "A", "10401 images/132556 rows/20016 WPRD cells", "A1-A6 readouts and N1/N2", "frozen text/prior/null", "WPRD", "cell-bootstrap conditional intervals", "ladder"),
        ("E2","Complementary prediction after O/G+O", "A", "FIT75066/943815; CAL8183/102612; VAL1182/14991", "frozen M2+nuisance offset", "O and G+O", "calibrated row LL", "2000 image-bootstrap; exploratory; one split/seed", "residual"),
        ("E3","Historical N4 admissibility", "C", "old1200 train;1078 FIT/122 CAL overlap", "vocabulary/weight provenance", "strict current FIT/CAL discipline", "identity overlap", "historical 2907-label map cannot be reused for current CAL", "vocab"),
        ("E4","Richer nuisance pilot", "B", "FIT5000/63286; CALFIT750/9078; CHECK250/3142", "16900->768->51 N4FIT; frozen fullFIT M2 offset", "N4FIT", "calibrated CALCHECK LL", "2000 image-bootstrap seed20261003; pilot; scale/orientation differ", "n4"),
        ("E5","M2 sampling composition", "B", "same VAL1182/14991;2.5k/5k/10k/15k train", "saved historical checkpoints and balanced15k", "identity-aligned rescoring", "LL/accuracy/macro recall", "2000 images seed11;descriptive;train rows differ", "bridge"),
        ("E6","Seed similarity and transfer", "B", "10231 nonE2 images/129826 rows;2051 image test", "within/cross seed readout", "same decoder applied across representations", "similarity/accuracy", "2000 images seed11;different from residual protocol", "seed"),
        ("E7","Registered path interventions", "C", "750/200/200 smoke;VAL1182/14991", "FULL/A/B seed1234", "matched objective/initialization protocol", "engineering integrity and descriptive metrics", "smoke only;no seed5678 or B2", "smoke_audit"),
        ("E8","Readout-v2", "C", "historical10401/132556/20016 cells", "corrected R2c", "R0", "WPRD", "no corrected paired CI found;older R2 treatment invalid", "readout"),
        ("E9","Geometry/units/Fourier mechanism", "C", "historical diagnostic populations", "source arithmetic and finite-probe recovery", "raw/shuffled geometry", "constant channels and probe R2", "not proof of mathematical information destruction", "fourier"),
        ("E10","Semantic/interactional evaluation", "D", "45 dev blocks/90 images;40 random/80 images", "human independent truth and contrast task", "object/geometry/visual controls proposed", "both-correct block accuracy", "human gold absent;current validation endpoint0 eligible", "e2_manifest"),
        ("E11","Entity scope and architecture", "C", "canonical source commit ec4cca6", "ground_nodes->_forward_impl->forward_pairs", "source data-flow inspection", "scope/gradient consumers", "no independent relation-only pathway", "model_source"),
        ("E12","Unseen-predicate feasibility", "D", "all50 real predicates in all original splits", "alias-aware fresh unseen-supervision design", "current closed-set C1", "design only", "no clean split/training or unseen evaluation executed", "predicate_disjoint"),
        ("E13","Registered C0/C1 geometry treatment", "A", "seed1234 matched20016 WPRD cells;3x12000 training samples", "pixel-space geometry plus Fourier scale .01", "registered locked C0", "delta WPRD +.008261", "cell CI [.003831,.012697];registered +.010 target missed;seed2 also misses", "c1_result"),
        ("E14","Historical conditional N4/N5 pilot", "B", "train1200/14746;VAL1182/14991", "historical N5 nuisance plus rel_feat", "historical N4", "LL3.094027 vs2.742337;delta-.351690", "CI[-.396702,-.304290];small historical train scale and capacity change;not current CAL-safe comparator", "conditional"),
    ]
    evidence_rows=[]
    for ident,hyp,tier,pop,treat,comp,metric,limit,src in evidence:
        evidence_rows.append({"id":ident,"hypothesis":hyp,"tier":tier,"population":pop,"treatment":treat,"comparator":comp,"metric":metric,"uncertainty_and_limitation":limit,"source":SOURCES[src],"source_sha256":source_hashes[src]["sha256"]})
    (out / "evidence_map.json").write_text(json.dumps({"baseline_commit":BASELINE,"created_utc":created,"evidence":evidence_rows,"sources":source_hashes},indent=2)+"\n")
    appendix = "# Evidence appendix\n\nPaths are repository-relative and are original immutable sources. Tiers describe support for the restricted claim, not confirmatory status. Tier A is strong central evidence, B exploratory, C engineering/provenance, D blocked.\n\n"
    for row in evidence_rows:
        appendix += f"## {row['id']} — {row['hypothesis']} (Tier {row['tier']})\n\nPopulation: {row['population']}. Treatment: {row['treatment']}. Comparator: {row['comparator']}. Metric: {row['metric']}. Limitation/uncertainty: {row['uncertainty_and_limitation']}. Source: `{row['source']}`. SHA256: `{row['source_sha256']}`.\n\n"
    appendix += "## Consistency disclosures\n\nThe final row audit supports 12,330/14,991 pairs, superseding the earlier 12,402 brief. Original G+O combined LL 1.3191095828 and independent saved-logit float64 value 1.3191095816 differ by about 1.2e-9; neither historical file is changed. The pilot LL difference independently matches within floating precision. Historical geometry/Fourier rhetoric is qualified as finite-probe recovery here. Full GO calibrated improvement coexists with worse raw LL/accuracy; the contrary earlier briefing is not current evidence.\n"
    appendix += "\nThe C0/C1 registered threshold and exact first-seed values are in `docs/PAPER_C_C0_C1_PREREGISTRATION.md`, `docs/PAPER_C_C1_RESULT.md` and `runs/eval_C1/result.json` (E13). The later seed2 report supersedes the first report's then-current statement that a second seed was not yet run. This historical seed5678 C1 is distinct from the unexecuted seed5678 matched-supervision smoke. Historical conditional N4/N5 (E14) is neither a substitute for fullscale N4-FIT nor a valid current CAL baseline.\n"
    (out / "appendix_evidence.md").write_text(appendix)
    (package / "SUPPLEMENTARY.md").write_text(appendix + "\n" + ARCH + "\n" + CENTRAL)
    full_paper = "# Decodability and nuisance dependence in predicate-supervised scene-graph representations\n\nManuscript-style evidence analysis; a submission-ready claim of novelty is not implied. Baseline commit: `"+BASELINE+"`.\n\n" + "\n".join(SECTIONS.values()) + "\n" + appendix
    (out / "paper_style_analysis.md").write_text(full_paper)
    (package / "PAPER_C_MASTER_ANALYSIS.md").write_text(full_paper)
    for name,text in SECTIONS.items():
        (out / (name + "_draft.md")).write_text(text)
        (package / (name.upper() + ".md")).write_text(text)
    for name,text in {"central_questions.md":CENTRAL,"hostile_review.md":REVIEW,"architecture_and_unseen_protocol.md":ARCH,"semantic_evaluation_protocol.md":E2,"NEXT_QUESTION_SELECTION.md":SELECTION,"FINAL_SCIENTIFIC_POSITION.md":POSITION}.items():
        (out / name).write_text(text)
    (package / "REVIEWER_RESPONSE_PREMORTEM.md").write_text(REVIEW)
    headers=["rank","candidate","direction","scientific_importance","novelty_judgment","feasibility","gpu_cost","data_requirements","annotation_requirements","inconclusive_risk","hypothesis_discrimination","conclusion_change_value","publication_value"]
    ranked=[dict(zip(headers,row)) for row in DIRECTIONS]
    write_csv(out / "direction_ranking.csv",ranked)
    ranktext="# Direction ranking\n\nQualitative PI judgments, not calibrated probabilities or literature-wide novelty findings. HIGH cost/requirements/risk is unfavorable; HIGH importance/discrimination/change/publication value is favorable. Ranking prioritizes identification and feasible prerequisites, not metric excitement. A and M are one selected evaluation-first project; their distinction is semantic gold versus matching design.\n\n| Rank | Direction | Importance | Feasible | GPU cost | Data | Annotation | Inconclusive risk | Discriminates | Changes conclusion | Novelty | Publication |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    for r in ranked:
        ranktext += "| " + " | ".join(str(r[k]) for k in ["rank","direction","scientific_importance","feasibility","gpu_cost","data_requirements","annotation_requirements","inconclusive_risk","hypothesis_discrimination","conclusion_change_value","novelty_judgment","publication_value"]) + " |\n"
    ranktext += "\nRanks 1–4 need human/control data before inference; ranks 5–11 increase distribution/stimulus demands; ranks 12–16 mainly refine predictive mechanisms before semantic identification. Present GPU availability does not change these values. No candidate is executed.\n"
    (out / "direction_ranking.md").write_text(ranktext)
    placement = """# Paper structure and evidence tiers

The five central findings are restricted predictive decoding with competitive geometry (E1), held-out O/G+O contrasts (E2), the bounded richer-nuisance/resource decision (E4), architecture/evaluation identification limits (E10/E11), and the registered C0/C1 threshold miss (E13). A source/provenance finding can be crucial main-text context while remaining Tier C; tiers are not a requirement to include only positive performance results.

| Evidence | Placement | Reason |
| --- | --- | --- |
| E1 A1/A3, geometry and null controls | Main summary; full ladder appendix | Central decoding/nuisance comparison with population and cell-dependence disclosure. |
| E2 full O/G+O residual audit | Main | Strongest current held-out predictive comparison; exploratory status and calibration orientation explicit. |
| E4 N4-FIT pilot | Main bounded result; recipe appendix | Explains resource decision and unresolved full nuisance question; no extrapolation. |
| E10/E11 semantic eligibility and source data flow | Main limitation; detailed scope appendix | Prevents semantic overclaim and motivates evaluation-first selection. |
| E13 C0/C1 preregistered outcome | Main | Reports original hypothesis and missed success threshold honestly. |
| E3 vocabulary/CAL lineage | Appendix and reproducibility supplement | Explains why the old checkpoint is inadmissible. |
| E5 M2 prefix/balanced bridge | Appendix | Sampling sensitivity, not a clean scaling or generalization result. |
| E6 seed similarity/transfer | Appendix | Distinct protocol; no residual robustness claim. |
| E7 FULL/A/B smoke | Supplementary engineering diagnostics | Integrity and narrow path sensitivity; no causal/semantic inference. |
| E8 readout-v2 | Appendix fidelity note | Corrected result and invalid historical treatment must be separated. |
| E9 units/Fourier | Appendix | Source defect and finite-probe recovery; qualified historical rhetoric. |
| E12 unseen-predicate design | Future-work supplement | No experiment performed; current C1 is supervised on all real predicates. |
| E14 old N4/N5 conditional | Appendix or internal only | Earlier motivation; small train scale/capacity and later CAL incompatibility prevent current comparison. |
| Failed, blocked, aborted attempts | Internal archive with reproducibility index | Preserve provenance without inflating the main result count. |

A minimum paper package requires the source-linked tables, original preregistered outcome, current held-out probabilities and calibration provenance, N4 pilot's bounded interpretation, the source data-flow limitations, and the saved claim audit. It does not require another GPU run. Actual venue submission additionally needs external novelty/literature positioning and editorial preparation, neither fabricated by this repository synthesis.
"""
    (out / "paper_structure_recommendation.md").write_text(placement)
    write_csv(tables / "evidence_tiers.csv",evidence_rows)
    ladder_rows=[{"arm":k,"WPRD":v["wprd_macro"],"cells":v["n_cells"],"form":v["form"],"source":SOURCES["ladder"]} for k,v in ladder["arms"].items()]
    write_csv(tables / "readout_ladder.csv",ladder_rows)
    residual_rows=[{"model":m["model"],"calibration":m["calibration"],"log_loss":m["log_loss"],"accuracy":m["accuracy"],"macro_recall":m["macro_recall"],"rows":m["rows"],"population":"fixed_validation","source":SOURCES["residual"]} for m in residual["metrics"]]
    for k,v in n4["CAL_CHECK"].items():
        if isinstance(v,dict) and "log_loss" in v:
            residual_rows.append({"model":k,"calibration":"CAL-FIT" if "calibrated" in k else "raw","log_loss":v["log_loss"],"accuracy":v["accuracy"],"macro_recall":v["macro_recall"],"rows":v["rows"],"population":"pilot_CAL-CHECK","source":SOURCES["n4"]})
    write_csv(tables / "nuisance_and_pilot_metrics.csv",residual_rows)
    effect_rows = []
    for effect in residual["paired_image_cluster_bootstrap"]:
        effect_rows.append({"contrast": effect["contrast"], "delta_calibrated_LL": effect["estimate_nats_per_row"],
                            "CI_low": effect["ci_low_percentile"], "CI_high": effect["ci_high_percentile"],
                            "images": 1182, "rows": 14991, "population": "fixed_validation",
                            "unit": "image", "replicates": effect["bootstrap_replicates"], "seed": effect["seed"],
                            "status": "exploratory_conditional", "source": SOURCES["residual"]})
    pilot_bootstrap = load(repo, "n4_bootstrap")
    effect_rows.append({"contrast": "N4FIT_rel minus N4FIT", "delta_calibrated_LL": pilot_bootstrap["estimate"],
                        "CI_low": pilot_bootstrap["ci95_percentile"][0], "CI_high": pilot_bootstrap["ci95_percentile"][1],
                        "images": 250, "rows": 3142, "population": "pilot_CAL_CHECK",
                        "unit": "image", "replicates": pilot_bootstrap["replicates"], "seed": pilot_bootstrap["seed"],
                        "status": "pilot_only_exploratory_conditional", "source": SOURCES["n4_bootstrap"]})
    write_csv(tables / "residual_effects_and_intervals.csv", effect_rows)
    chronology = [("registered C0/C1", "c1_result"), ("readout-v2 corrected", "readout"),
                  ("A1-A6 and null ladder", "ladder"), ("cross-seed diagnostics", "seed"),
                  ("historical conditional N4/N5", "conditional"), ("M2 common-validation bridge", "bridge"),
                  ("matched-supervision smoke", "smoke_audit"), ("full-scale O/G+O residual", "full"),
                  ("current nuisance ladder", "residual"), ("historical vocabulary forensics", "vocab"),
                  ("bounded N4-FIT pilot", "n4"), ("final evidence audit", "audit_status")]
    timeline_rows = []
    for phase, key in chronology:
        record = load(repo, key)
        timestamp = record.get("created_utc", record.get("timestamp_utc", "NOT_RECORDED"))
        timeline_rows.append({"phase": phase, "recorded_timestamp": timestamp, "source": SOURCES[key],
                              "note": "Phase order; do not infer execution time from a directory name"})
    write_csv(tables / "experimental_timeline.csv", timeline_rows)
    bridge=load(repo,"bridge")
    bridge_rows=[{k:m[k] for k in ["run","training_images","training_rows","validation_images","validation_rows","log_loss","accuracy","macro_recall"]} for m in bridge["model_metrics"]]
    write_csv(tables / "M2_common_validation.csv",bridge_rows)
    prior=load(repo,"audit_metrics")
    predicate_rows=[]
    for row in prior["validation"]["per_predicate"]:
        q={"predicate":row["predicate"],"rows":row["G+O"]["rows"]}
        for model in ["O","G+O","rel_feat","G+O+rel_feat"]:
            for metric in ["log_loss","accuracy","macro_recall"]:
                q[model+"_"+metric]=row[model][metric]
        q["combined_minus_GO_LL"]=row["residual_minus_GO_LL"]
        predicate_rows.append(q)
    write_csv(tables / "per_predicate_saved_metrics.csv",predicate_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none"})
    labels = ["O + rel_feat vs O\nFixed validation, exploratory", "G+O + rel_feat vs G+O\nFixed validation, exploratory", "N4-FIT pilot\nCAL-CHECK, pilot only"]
    effects = [(row["delta_calibrated_LL"], (row["CI_low"], row["CI_high"]), label)
               for row, label in zip(effect_rows, labels)]
    fig,axes=plt.subplots(3,1,figsize=(8,5),layout="constrained")
    for ax,(point,ci,label) in zip(axes,effects):
        ax.errorbar(point,0,xerr=[[point-ci[0]],[ci[1]-point]],fmt="o",color="#245b78",capsize=4)
        ax.axvline(0,color="gray",lw=1);ax.set_yticks([]);ax.set_title(label,loc="left",fontsize=10);ax.set_xlabel("Calibrated log-loss delta (nats/row; negative favors combination)")
    fig.savefig(figures/"residual_intervals.svg");plt.close(fig)
    normalize_svg_whitespace(figures/"residual_intervals.svg")
    rows=sorted(predicate_rows,key=lambda r:r["G+O_log_loss"]*r["rows"],reverse=True)[:10]
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    ax.barh([r["predicate"] for r in rows][::-1],[r["G+O_log_loss"]*r["rows"] for r in rows][::-1],color="#245b78")
    ax.set_xlabel("Total calibrated G+O negative log-loss (nats)")
    ax.set_title("Ten largest total-loss contributors; prevalence affects this ranking")
    fig.savefig(figures/"total_loss_by_predicate.svg");plt.close(fig)
    normalize_svg_whitespace(figures/"total_loss_by_predicate.svg")
    (figures/"README.md").write_text("# Figure provenance\n\n`residual_intervals.svg` uses stored paired intervals, with separate axes/population captions to avoid implying a matched causal ladder. `total_loss_by_predicate.svg` multiplies saved predicate mean NLL by row count; it is a descriptive prevalence-sensitive ranking, not a newly selected endpoint. Source tables and `tools/build_paper_c_master_analysis.py` reproduce both; no model fitting.\n")
    (out / "independent_verification.json").write_text(json.dumps(verified,indent=2)+"\n")
    readme="""# Paper C post-closeout research analysis

This additive bundle is a manuscript and research-decision reconstruction from existing evidence. It uses CPU saved-logit consistency checks and reproducible tables/figures. No learned model, split, annotation, inference, extraction or GPU experiment is added.

Start with `paper_style_analysis.md`, `hostile_review.md`, `NEXT_QUESTION_SELECTION.md`, and `FINAL_SCIENTIFIC_POSITION.md`. The paper-oriented uppercase drafts, tables and figures are in repository-root `paper_package/`. `evidence_map.json` contains original artifact hashes and population/estimator limits; `independent_verification.json` records independent saved-logit metric and identity checks.

Decision: no autonomous experiment with current assets. The selected future question requires human gold and a frozen powered evaluation population. Paper C stays closed; the evaluation-first question belongs to a separately registered project. No claim of no residual or equivalence is made.

Reproduce into NEW, nonexistent destinations using:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/build_paper_c_master_analysis.py --output runs/paper_c_master_research_analysis_<UTC_TIMESTAMP> --package <new_package_path>
```

The builder refuses existing output paths and checks all source and immutable artifacts before writing. Analysis inputs must be available locally; a Git-only clone does not include all payloads.
"""
    (out/"README.md").write_text(readme)
    (package/"README.md").write_text(readme)
    status={"status":"PAPER_C_NO_NEW_EXPERIMENT_JUSTIFIED","analysis_complete":True,"baseline_commit":BASELINE,"created_utc":created,"selected_question":"human-verified object-pair controlled relation discrimination","experiment_executed":False,"gpu_hours":0,"learned_models_fit":0,"new_splits":0,"human_gold_status":"MISSING","recommendation":"SPLIT_INTO_NEW_PAPER; Paper C remains CLOSE"}
    (out/"final_status.json").write_text(json.dumps(status,indent=2)+"\n")
    outputs={}
    for root in [out,package]:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p != out/"hashes.json": outputs[str(p.relative_to(repo))]=sha256(p)
    (out/"hashes.json").write_text(json.dumps({"sources":source_hashes,"generated_outputs":outputs,"immutable":verified["immutable_hashes"]},indent=2)+"\n")
    print(json.dumps({"status":status["status"],"output":str(out),"package":str(package),"generated_files":len(outputs),"GPU_used":False},indent=2))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--refresh-generated",action="store_true",help="Refresh only a verified draft previously generated by this tool; refuse changed or foreign files")
    args=parser.parse_args()
    repo=args.repo.resolve()
    out=args.output if args.output.is_absolute() else repo/args.output
    package=args.package if args.package.is_absolute() else repo/args.package
    if not out.resolve().is_relative_to(repo) or not package.resolve().is_relative_to(repo):
        raise ValueError("Analysis outputs must stay within the repository")
    if not str(out.relative_to(repo)).startswith("runs/paper_c_master_research_analysis_"):
        raise ValueError("Use a new paper_c_master_research_analysis run directory")
    build(repo,out,package,args.refresh_generated)


if __name__ == "__main__":
    main()
