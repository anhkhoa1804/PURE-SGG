# Paper Tables v0.1

## Table 1. Historical WPRD readout ladder

Historical aggregate WPRD; each row is one saved arm evaluated over 20,016 eligible WPRD cells from 10,401 images / 132,556 relation rows, 50 predicates. The cell aggregation—not an independent model sample—is the metric unit. A5a and A5b differ in fit protocol. Status: historical. Exact source: `paper_package/TABLES/readout_ladder.csv`; underlying `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/result.json`.

| Arm | Definition (saved) | WPRD |
|---|---|---:|
| A1 | Frozen baseline text-logit channel | 0.574988 |
| A2 | Linear readout | 0.580128 |
| A3 | GELU MLP, 768→768→51 | 0.585669 |
| A4 | Cosine readout | 0.575834 |
| A5a | 19-D geometry, cross-fit | 0.588144 |
| A5b | 19-D geometry, train-fit | 0.596235 |
| A6 | Linear fused rel_feat + geometry | 0.585503 |
| N1 | Shuffled-label null | 0.505451 |
| N2 | Pair-prior control | 0.500000 |

## Table 2. Geometry and learned-head comparison

Same historical WPRD population and source as Table 1. These are descriptive estimator comparisons; A5b's fit population differs, and no new pairwise CI is implied. Source: `paper_package/TABLES/readout_ladder.csv`.

| Estimator | WPRD | Interpretation limit |
|---|---:|---|
| A1 frozen baseline | 0.574988 | Historical reference |
| A2 linear head | 0.580128 | Decodability, not semantic isolation |
| A3 GELU head | 0.585669 | Greater readout capacity; not semantic evidence |
| A5a geometry cross-fit | 0.588144 | Geometry is a strong competing predictor |
| A5b geometry train-fit | 0.596235 | Different fitting protocol from A5a |
| A6 feature + geometry | 0.585503 | Fusion did not exceed A5b descriptively; no causal decomposition |

Historical C0/C1 context is not a clean units-only comparison: C0 WPRD 0.566727 is in `runs/eval_C0/result.json`; C1 WPRD 0.574988 is in `runs/eval_C1/result.json`, but Fourier settings differ. The corrected R1 result is unavailable (`research_analysis/R1_artifact_audit.md`).

## Table 3. Corrected nested nuisance analysis

The extension holds the calibrated parent fixed and adds the fitted nonnegative frozen `rel_feat` offset; alpha=0 exactly recovers the parent. All arms use 250 CAL-CHECK images / 3,142 rows. LL is relation-row mean calibrated/final LL. CIs are exploratory paired image-cluster bootstrap intervals (2,000 resamples; seed 20261003). Source: `research_analysis/R2_nested_nuisance_analysis.json`, `research_analysis/R2_nested_nuisance_analysis.md` §8.

| Parent | Extension | LL (parent → extension) | ΔLL | Exploratory 95% CI | Accuracy (parent → extension) | Macro recall (parent → extension) |
|---|---|---:|---:|---:|---:|---:|
| O | O + rel_feat | 1.598961 → 1.280646 | −0.318314 | [−0.357666, −0.276130] | 0.663590 → 0.682050 | 0.227013 → 0.200346 |
| G+O | G+O + rel_feat | 1.315260 → 1.255252 | −0.060008 | [−0.074317, −0.044480] | 0.667091 → 0.668364 | 0.137086 → 0.135990 |

## Table 4. Predicate-level heterogeneity

Selected largest-support G+O classes to show that class-level changes are mixed; this is support ordering, not a predeclared head/tail group. Recall and precision are one-vs-rest. Full predicate list for both contrasts is in `research_analysis/R2_per_predicate_effects.csv`; same CAL-CHECK population, status exploratory. Source: that CSV.

| Predicate | Support | Baseline recall | Extended recall | Δ recall | Baseline precision | Extended precision | Δ precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| on | 1,124 | 0.946619 | 0.948399 | +0.001779 | 0.620408 | 0.620128 | −0.000280 |
| has | 418 | 0.892344 | 0.901914 | +0.009569 | 0.722868 | 0.716730 | −0.006138 |
| in | 404 | 0.529703 | 0.514851 | −0.014851 | 0.798507 | 0.790875 | −0.007633 |
| wearing | 258 | 0.821705 | 0.837209 | +0.015504 | 0.812261 | 0.805970 | −0.006290 |
| of | 249 | 0.526104 | 0.522088 | −0.004016 | 0.723757 | 0.734463 | +0.010706 |
| with | 78 | 0.025641 | 0.012821 | −0.012821 | 0.200000 | 0.200000 | 0.000000 |
| behind | 71 | 0.478873 | 0.478873 | 0.000000 | 0.618182 | 0.653846 | +0.035664 |
| holding | 69 | 0.391304 | 0.405797 | +0.014493 | 0.675000 | 0.666667 | −0.008333 |
| near | 68 | 0.014706 | 0.029412 | +0.014706 | 0.125000 | 0.285714 | +0.160714 |
| next to | 51 | 0.156863 | 0.117647 | −0.039216 | 0.258065 | 0.222222 | −0.035842 |

## Table 5. Claim and limitation summary

| Claim | Evidence status | Main limitation |
|---|---|---|
| Predicate scores are decodable | Supported descriptively | Decodability does not identify semantic content |
| WPRD captures within-pair discrimination | Narrowly supported | Eligibility, dependence, no planted shortcut or strong-VLM ceiling |
| rel_feat adds utility beyond G+O | Exploratory bounded support | 250-image CAL-CHECK pilot; no seed uncertainty |
| Uniform predicate gain | Not supported | Recall/precision changes mixed; macro recall declines slightly |
| Geometry semantic effect | Missing | R1 blocked; no endpoint |
| Open-vocabulary generalization | Missing | No predicate-disjoint evaluation |

Sources: `research_analysis/paper/CLAIM_EVIDENCE_MATRIX.md`, R2/R3 audits, and `research_analysis/R1_artifact_audit.md`.
