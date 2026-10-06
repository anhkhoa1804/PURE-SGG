# Decodability and nuisance dependence in predicate-supervised scene-graph representations

Manuscript-style evidence analysis; a submission-ready claim of novelty is not implied. Baseline commit: `7d6eaf2098ad601d4d573d374f99e0feccc03252`.

# Abstract

Predicate prediction from a scene-graph representation does not identify which visual or statistical cues support that prediction. We reconstruct a controlled research program on a frozen, predicate-supervised C1 representation and distinguish decodability from evidence for relation-specific content. On 20,016 historical within-pair discrimination cells, a GELU readout reached WPRD 0.585669 compared with 0.574988 for the frozen text readout, while geometry-only readouts reached 0.588144 (cross-fit) and 0.596235 (train-fit). On a separate fixed validation population of 1,182 images and 14,991 labeled rows, combining frozen rel_feat with an ordered-object-pair prior reduced calibrated log-loss by 0.366033 nats per row; after adding geometry the increment was 0.005129, with an image-cluster interval including zero. A separately held-out, leakage-safe N4-FIT feasibility pilot incorporating object identity, geometry, CLIP crops and global context improved log-loss by 0.002446, with a pilot-only interval below zero, but did not meet its registered resource criterion for fullscale reconstruction. These contrasts differ in population and offset construction and are not a strictly nested causal ladder. Architecture and provenance audits identify substantial nuisance pathways and prevent several tempting cross-experiment comparisons. The evidence establishes predictive decodability and utility relative to specified baselines, while full-population residual beyond appearance/context and semantic relational discrimination remain unresolved. We recommend closing the present predictive hypothesis and making human-verified controlled evaluation a separately registered project before further model development. [E1–E8]

# Introduction

Scene-graph predicate labels combine spatial, object, appearance and context regularities. A representation trained on those labels may support accurate prediction while primarily integrating those regularities. Paper C initially asked whether repairing the inherited geometry pathway improved C1 over C0 on within-pair relation discrimination. Its later operational question became whether a frozen predicate-supervised representation retained usable predictive structure after progressively stronger nuisance estimators were considered. These are related but distinct questions. A decoder improvement does not itself identify a semantic mechanism.

The program matters because the proposed constructive method—a residual relation expert trained against a nuisance offset—requires a residual worth targeting. Without that premise, another training campaign could optimize a weak or misidentified effect. The evidence therefore tests alternative explanations rather than treating any positive readout as proof of relational content: ordered object-pair frequency, geometry, decoder capacity, sampling composition, readout implementation, seed transfer, and appearance/context. The last alternative was tested only in a bounded pilot. [E1–E10]

This manuscript is an evidence reconstruction, not a claim of a new state-of-the-art method. Its contribution is the measured contraction of a particular predictive contrast, the disclosure of implementation and lineage constraints, and the distinction between a resource decision and a scientific null. No global novelty claim is certified by this repository-only review. Five findings deserve the main text: registered C0/C1 performance and its threshold miss; readout decodability with competitive geometry; the O and G+O held-out contrasts; the narrowly interpretable N4-FIT pilot; and the architecture/evaluation limits that prevent a semantic conclusion. Sampling, seed transfer, readout-v2, Fourier recovery and intervention smoke belong in the appendix.

The preferred explanation is that C1 integrates several predictive cues. The strongest alternative to that explanation is that useful image-conditioned relation structure exists but is poorly identified by the current nuisance estimators and labels. Current evidence cannot distinguish these fully. The most valuable next question concerns human-verified discrimination when object labels are held fixed and measured nuisance predictors are explicit competitors. [E7–E13]

# Methods

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

# Results

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

# Discussion

The strongest conclusion is that rel_feat supports closed-set predictive decoding and complements an ordered-pair prior under the tested scoring protocol. Geometry is a strong competing predictor, and the full G+O increment is small and uncertain. The N4-FIT pilot indicates a small conditional benefit after its richer nuisance estimator, but does not identify semantic content or resolve the full nuisance question. The scientific lesson concerns identification: measured decoder utility depends on comparator strength, calibration, population, and architecture lineage. [E1–E4]

The best current hypothesis is that C1 is an integrator of object, visual, geometry and scene cues. Its source allows all these pathways into rel_feat. The best competing hypothesis is that useful relational discrimination survives those cues, but the current semantic evaluation and nuisance estimators do not identify it adequately. Predictive performance alone cannot decide between them. A neural nuisance expert is also fallible: failure to condition away a measured effect may reflect underfitting, vocabulary coverage, misspecified geometry, or calibration. Conversely, a strong nuisance fit does not mathematically remove semantic content. [E2,E3,E11]

Scientific significance and practical value should be separated. The pilot interval below zero indicates a conditional predictive difference on its population. Its .002446-nat change has no demonstrated deployment or semantic utility and did not satisfy the frozen spending criterion. There is no universal effect-size threshold established here. Calling the effect negligible in an absolute sense, or reporting an equivalence result, would require a new registered criterion. The paper can honestly report the effect, uncertainty and resource decision without either inflation or erasure. [E4]

The highest-value next question is evaluation: does a frozen visual model track human-verified relation truth in exact ordered-object-pair contrast blocks beyond explicit geometry and appearance/context competitors? Existing E2 development candidates offer a model-blind starting instrument, not gold labels. A human development phase can reveal whether the task is solvable, whether predicates co-occur rather than contrast, and whether nuisance-matched support exists. Architecture redesign before this evidence would change several things while leaving the semantic target unidentified. [E10–E12]

Paper C should close as a bounded predictive and reproducibility investigation. A future semantic study should be a separately registered Paper D. Its success would strengthen a narrower claim of controlled image-conditioned relation discrimination; it would not automatically establish causality, general semantic understanding, or unseen-predicate transfer. No new experiment is executed in this branch because the highest-value question lacks human gold and a powered final evaluation population. Existing CPU artifacts are used for the manuscript and independent consistency checks. [E10]

# Limitations

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

# Conclusion

Paper C establishes protocol-specific decodability and predictive utility beyond an ordered-object-pair prior. Adding geometry greatly reduces the measured increment; the remaining full G+O contrast is uncertain. A richer leakage-safe nuisance pilot yields a small benefit that does not justify its registered fullscale resource expense. These findings support a restrained account of nuisance-sensitive predictive representations. They leave semantic relation discrimination and full-population residual beyond appearance/context unresolved. The constructive residual-training hypothesis is untested rather than disproved. The current direction should remain closed; a separately registered, human-verified controlled evaluation has greater expected scientific value than another model or extraction run with the present labels.

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
