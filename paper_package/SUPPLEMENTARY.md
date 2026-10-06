# Evidence appendix

Paths are repository-relative and are original immutable sources. Tiers describe support for the restricted claim, not confirmatory status. Tier A is strong central evidence, B exploratory, C engineering/provenance, D blocked.

## E0 — Artifact/probability identity (Tier C)

Population: 1182/14991 validation; 250/3142 CAL-CHECK. Treatment: saved logits and canonical artifacts. Comparator: recorded metrics/hash values. Metric: exact recomputation. Limitation/uncertainty: no new inference. Source: `runs/paper_c_final_evidence_audit_20261003T185925Z/posthoc_metrics.json`. SHA256: `641c63be540a625d6b43008257934a1e89cd4306a6f7728e90e7512687e80bce`.

## E1 — Decodability and geometry competition (Tier A)

Population: 10401 images/132556 rows/20016 WPRD cells. Treatment: A1-A6 readouts and N1/N2. Comparator: frozen text/prior/null. Metric: WPRD. Limitation/uncertainty: cell-bootstrap conditional intervals. Source: `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/result.json`. SHA256: `846e8f1dac516a0b3b43dee8a2209bc1cb9709f418225f2bd47adf267146cccf`.

## E2 — Complementary prediction after O/G+O (Tier A)

Population: FIT75066/943815; CAL8183/102612; VAL1182/14991. Treatment: frozen M2+nuisance offset. Comparator: O and G+O. Metric: calibrated row LL. Limitation/uncertainty: 2000 image-bootstrap; exploratory; one split/seed. Source: `runs/paper_c_nuisance_ladder_v2_20261003T092837Z/nuisance_ladder_v2_results.json`. SHA256: `182f644d024345c26e7b949072da580030b40e0bd5a95639293d5b95e9fb7477`.

## E3 — Historical N4 admissibility (Tier C)

Population: old1200 train;1078 FIT/122 CAL overlap. Treatment: vocabulary/weight provenance. Comparator: strict current FIT/CAL discipline. Metric: identity overlap. Limitation/uncertainty: historical 2907-label map cannot be reused for current CAL. Source: `runs/paper_c_nuisance_vocab_contract_20261003T114634Z/vocabulary_provenance.json`. SHA256: `517ee396c7fabdebc3a1ed699508da3c597368f655607062c69e61802014e1aa`.

## E4 — Richer nuisance pilot (Tier B)

Population: FIT5000/63286; CALFIT750/9078; CHECK250/3142. Treatment: 16900->768->51 N4FIT; frozen fullFIT M2 offset. Comparator: N4FIT. Metric: calibrated CALCHECK LL. Limitation/uncertainty: 2000 image-bootstrap seed20261003; pilot; scale/orientation differ. Source: `runs/paper_c_n4fit_pilot_training_20261003T182118Z/pilot_results.json`. SHA256: `05a112234b1f2590ccf17e847c15df04b16200e5fb1d9ae39109494da6dfa7ac`.

## E5 — M2 sampling composition (Tier B)

Population: same VAL1182/14991;2.5k/5k/10k/15k train. Treatment: saved historical checkpoints and balanced15k. Comparator: identity-aligned rescoring. Metric: LL/accuracy/macro recall. Limitation/uncertainty: 2000 images seed11;descriptive;train rows differ. Source: `runs/paper_c_same_validation_checkpoint_rescore_20261002T082100Z/same_validation_checkpoint_rescoring.json`. SHA256: `f328b7cea362e526eda197a7752195fe35ec3e561106159e612e2f77e9cff7c5`.

## E6 — Seed similarity and transfer (Tier B)

Population: 10231 nonE2 images/129826 rows;2051 image test. Treatment: within/cross seed readout. Comparator: same decoder applied across representations. Metric: similarity/accuracy. Limitation/uncertainty: 2000 images seed11;different from residual protocol. Source: `runs/l4_breakthrough_20260930/seed_stability/fit_retry1/result.json`. SHA256: `5fd9e1b57d9303f205e86a08f3dcc90e2b940c162e2d59282fc1992994e8b0e7`.

## E7 — Registered path interventions (Tier C)

Population: 750/200/200 smoke;VAL1182/14991. Treatment: FULL/A/B seed1234. Comparator: matched objective/initialization protocol. Metric: engineering integrity and descriptive metrics. Limitation/uncertainty: smoke only;no seed5678 or B2. Source: `runs/paper_c_matched_supervision_smoke_20261002T111542Z/c1_smoke_final_audit.json`. SHA256: `1697a2e8cd53832a36e19c33814ce4fa755cc465311a1a79e1f49520322ebcbc`.

## E8 — Readout-v2 (Tier C)

Population: historical10401/132556/20016 cells. Treatment: corrected R2c. Comparator: R0. Metric: WPRD. Limitation/uncertainty: no corrected paired CI found;older R2 treatment invalid. Source: `runs/eval_readout_v2_R2c_20260911T023722Z/result.json`. SHA256: `d4d568eb81016173640a3a9f89e7eba50414178b1df29957ee667a589f7501cb`.

## E9 — Geometry/units/Fourier mechanism (Tier C)

Population: historical diagnostic populations. Treatment: source arithmetic and finite-probe recovery. Comparator: raw/shuffled geometry. Metric: constant channels and probe R2. Limitation/uncertainty: not proof of mathematical information destruction. Source: `docs/FOURIER_NULL_AND_INVERTIBILITY_RESULT.md`. SHA256: `96d835f68e1700ff36c5fb322c8972ce232fa7d8148dc1271a76ff292f9eeac3`.

## E10 — Semantic/interactional evaluation (Tier D)

Population: 45 dev blocks/90 images;40 random/80 images. Treatment: human independent truth and contrast task. Comparator: object/geometry/visual controls proposed. Metric: both-correct block accuracy. Limitation/uncertainty: human gold absent;current validation endpoint0 eligible. Source: `runs/e2_development_20260912/FROZEN_CANDIDATE_MANIFEST.json`. SHA256: `d4fd069e5b83c5c32184a418ca7386e02a39c8b96801414ecdc7e6d78d9fc4eb`.

## E11 — Entity scope and architecture (Tier C)

Population: canonical source commit ec4cca6. Treatment: ground_nodes->_forward_impl->forward_pairs. Comparator: source data-flow inspection. Metric: scope/gradient consumers. Limitation/uncertainty: no independent relation-only pathway. Source: `runs/paper_c_canonical_train_relfeat_20260930T062248Z/code/openvocab_rel/models/relational_model.py`. SHA256: `6ccb855a20122f3917b87bcca2f561a1bc3d271f16cd140b5b0a448bb45558db`.

## E12 — Unseen-predicate feasibility (Tier D)

Population: all50 real predicates in all original splits. Treatment: alias-aware fresh unseen-supervision design. Comparator: current closed-set C1. Metric: design only. Limitation/uncertainty: no clean split/training or unseen evaluation executed. Source: `docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md`. SHA256: `df462f4d5d384debeda70a07bf98b16f40e2745716fc01e2bdc6be151043aeb0`.

## E13 — Registered C0/C1 geometry treatment (Tier A)

Population: seed1234 matched20016 WPRD cells;3x12000 training samples. Treatment: pixel-space geometry plus Fourier scale .01. Comparator: registered locked C0. Metric: delta WPRD +.008261. Limitation/uncertainty: cell CI [.003831,.012697];registered +.010 target missed;seed2 also misses. Source: `runs/eval_C1/result.json`. SHA256: `8895f6d6bfd14f96bf178002e993dea0dc36d6eea8ad8395e750d0c456eda236`.

## E14 — Historical conditional N4/N5 pilot (Tier B)

Population: train1200/14746;VAL1182/14991. Treatment: historical N5 nuisance plus rel_feat. Comparator: historical N4. Metric: LL3.094027 vs2.742337;delta-.351690. Limitation/uncertainty: CI[-.396702,-.304290];small historical train scale and capacity change;not current CAL-safe comparator. Source: `runs/paper_c_decision_pilot_20261001T123842Z/conditional_pilot_20261001T171120Z/conditional_pilot_results.json`. SHA256: `862704b18c0556053187d2876fedc9f163395f710418ae4615ace4562fbd6758`.

## Consistency disclosures

The final row audit supports 12,330/14,991 pairs, superseding the earlier 12,402 brief. Original G+O combined LL 1.3191095828 and independent saved-logit float64 value 1.3191095816 differ by about 1.2e-9; neither historical file is changed. The pilot LL difference independently matches within floating precision. Historical geometry/Fourier rhetoric is qualified as finite-probe recovery here. Full GO calibrated improvement coexists with worse raw LL/accuracy; the contrary earlier briefing is not current evidence.

The C0/C1 registered threshold and exact first-seed values are in `docs/PAPER_C_C0_C1_PREREGISTRATION.md`, `docs/PAPER_C_C1_RESULT.md` and `runs/eval_C1/result.json` (E13). The later seed2 report supersedes the first report's then-current statement that a second seed was not yet run. This historical seed5678 C1 is distinct from the unexecuted seed5678 matched-supervision smoke. Historical conditional N4/N5 (E14) is neither a substitute for fullscale N4-FIT nor a valid current CAL baseline.

# Architecture, losses and predicate-disjoint feasibility

The immutable `relational_model.py` defines `ProgressiveEdgeConditionedLayer` near line 35, node grounding at 345, pair decoding at 431, `_forward_impl` at 660 and text-predicate scoring at 859. Grounding projects whole-image visual tokens and box queries, uses global bias and node attention/pooling. `_forward_impl` applies a two-layer `context_mixer` across object nodes around line 695 before selecting subject/object rows. Thus node appearance and other-object context already mix upstream of pair operations. Pair decoding combines subject/object features, Fourier geometry and fusion_gate, includes SPOA state when configured and `rel_seed`, performs two edge updates conditioned on incoming relation state, subject/object, visual tokens and geometry, then bilinear/output normalization. The final 768-D rel_feat has no clean relation-only branch. Each artifact's saved config determines which optional pathways execute. [E11]

The earliest explicit pair conditioning occurs in pair selection/subject-object fusion and geometry conditioning; SPOA's predicate-conditioned visual readout strengthens it. Edge layers receive pair-specific state and geometry as well as contextualized node features and image tokens. Object appearance, global scene structure, measured geometry and lexical supervision can all flow into the output. Train-source predicate CE consumes learned relation outputs; text predicate CE uses relation projections against text embeddings. Grounding, SPOA/counterfactual, relationness, KL/rank and configured auxiliary losses can also route gradients upstream; enabled terms/weights must be read from checkpoint/config, not assumed active merely because code exists. [E11]

A changes geometry conditioning at one fusion gate, leaving geometry elsewhere. B changes pair-state alignment at both edge interfaces, while preserving target unary/geometry/visual tensors and earlier context mixing. Objective formulas can stay fixed while gradients change. These diagnostic treatments do not identify exclusive semantic content. A distinct branch would need explicit separated inputs, declared information barriers, shared matched supervision and controls; even then separation by architecture would not establish semantic identification without controlled labels and evaluation. A stronger architecture claim is therefore not the first missing experiment. [E7,E11]

The existing predicate-disjoint audit establishes that all 50 real predicates occur in each original train/validation/test split; the 51st model placeholder `relation` is not a real predicate. Metadata lists three orphan labels (`around`, `growing on`, `says`) and cannot define the trained vocabulary. `wearing`/`wears` and other aliases require joint assignment. Current C1 has seen every evaluation predicate via predicate and text-predicate supervision, so rescoring it on a renamed held-out set is not unseen-predicate evaluation. [E12]

A future unseen-predicate protocol is technically designable: freeze alias-aware predicate groups before training; remove all held-out predicate supervision and text-conditioned relation losses; audit frequency priors, samplers, prompts, prototypes, metadata and checkpoint initialization; keep object splits and image boundaries explicit; separate development predicates from final held-out predicates; score frozen text embeddings without fitting unseen prototypes. Removing annotated relationships does not make visually present unseen concepts absent, so task-unseen must be distinguished from globally unseen. Distribution matching cannot be guaranteed when common or rare predicates are withheld; the historical audit shows severe imbalance. This requires a fresh model and protocol and does not win the current next-direction ranking. No split or implementation is created here.

# Central scientific questions

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
