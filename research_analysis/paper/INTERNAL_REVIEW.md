# Internal Review

## Reviewer 1 — Empirical SGG

### Overall assessment

The manuscript has a defensible audit thesis, but its strongest empirical result is a bounded predictive comparison, not evidence that the representation encodes semantic relations. The work can be an empirical measurement paper if the historical readout ladder, WPRD definition, and current nested pilot are clearly separated by population and purpose.

### Strengths

- The paper does not equate decoder performance with semantic understanding.
- The corrected R2 fixes the parent and gives the added branch an exact zero-effect setting.
- Historical results and the blocked R1 endpoint are preserved rather than silently revised.
- The pair-prior WPRD control addresses one concrete shortcut: a score constant within an ordered pair cannot discriminate predicates within that pair.

### Major concerns

1. The paper spans several populations and experiment generations. The historical A1–A6 ladder, R3 correlation arms, corrected R2 CAL-CHECK pilot, N4-FIT pilot, and R1 attempt are not one matched experiment. The text needs a compact provenance/population table and must not compare their magnitudes as if they were.
2. The biological/semantic interpretation remains underidentified: geometry, local appearance, scene context, annotation regularities, and pair-conditioned frequency are not jointly controlled in the primary R2.
3. WPRD is a custom conditional statistic with partial construct validation. Its narrow operational interpretation is plausible, but its general benchmark validity is not established.
4. Predicate-level changes are mixed. The paper should show them, not imply a broad gain from aggregate loss.

### Minor concerns

- Define “within-pair” with the actual unit and eligible-cell rule at first use.
- Distinguish “macro recall” from pooled accuracy wherever both are reported.
- A5a and A5b have different fit protocols; the historical ladder is descriptive, not a controlled capacity ranking.

### Required revisions

- Put R2's 250-image/3,142-row CAL-CHECK population in the abstract and results lead.
- State explicitly that the O and G+O contrasts are nested offsets evaluated on the same check rows, while the older independent decoder contrast is not the corrected estimate.
- Include a predicate table or figure with support, baseline recall, extended recall, and change; no invented head/tail cutoffs.
- Move broad WPRD correlation claims out of the main argument; report n=12 dependent arms and the absent controls.
- Keep the geometry units-contract finding separate from any performance consequence.

### Fatal concerns, if any

No fatal data-integrity defect is evident in the saved corrected R2 artifact. The absence of a semantic evaluation is fatal only to a semantic-understanding claim, not to a narrowly framed measurement/predictive audit.

### Evidence needed to resolve concerns

The current paper can resolve presentation issues using saved artifacts. Broader semantic identification would require evidence not present here (for example, a controlled semantic evaluation); no such experiment is authorized in this review.

## Reviewer 2 — Statistics and Evaluation

### Overall assessment

R2 is substantially better identified than the earlier independent-decoder contrast. Its uncertainty is correctly clustered by image for the saved paired predictions. Nonetheless, this is a small exploratory check set with one fitted model and one split; the reported interval is not a complete sampling distribution over training, calibration, or datasets.

### Strengths

- Same CAL-CHECK rows and frozen parent predictions are used in each paired contrast.
- CAL-FIT (750 images/9,078 rows) is distinct from CAL-CHECK (250/3,142); the added scalar is fitted before checking.
- The exact alpha=0 parent point is included, with a same-objective parent fallback.
- The report preserves mixed accuracy/macro-recall changes and an image-cluster bootstrap.

### Major concerns

1. The CI covers image resampling conditional on the single fitted parent, offset, and split. It does not account for refitting variation, seed variation, or population shift.
2. The effective inferential unit is image cluster, while the point estimate is pooled row-weighted. Images contribute different numbers of rows; this weighting must remain explicit.
3. The check set is modest and the CI's exclusion of zero should not be translated into a confirmatory or population-wide significance claim.
4. The WPRD arm-correlation permutation p-values use dependent deterministic arms. They are not valid independent-model inference.
5. The historical validation and pilot numbers have different fitting/calibration regimes and must not be directly differenced.

### Minor concerns

- Call the interval an “exploratory paired image-cluster percentile interval,” not a generic confidence interval without qualification.
- Give the bootstrap repetition count and seed in the main methods or a prominent table note.
- State that rows are pooled within each resampled image draw, preserving the row-weighted estimand.

### Required revisions

- Use “exploratory” at every first presentation of R2, including abstract/table caption.
- Avoid “significant,” “robust,” or “generalizes” for this result.
- Report both raw and calibrated/final LL only with their distinct definitions; identify the primary calibrated contrast.
- State explicitly that model-fitting uncertainty is not captured.
- Identify WPRD correlation n=12 and dependence in the same paragraph as the coefficients.

### Fatal concerns, if any

No fatal defect is established for the bounded R2 estimand as recorded. It is not sufficient for a broad generalization claim or a full-population nuisance conclusion.

### Evidence needed to resolve concerns

For the current paper: accurate estimand and limitation language, already supported by artifacts. For population-wide uncertainty: repeated fits or external populations would be needed, but these are outside this closed experimental phase.

## Reviewer 3 — Representation Learning and Semantics

### Overall assessment

The paper is strongest as a caution about what predictive scores and decodability do not identify. It does not establish semantic specificity. The current representation may integrate object appearance, geometry, contextual features, pair priors, and other cues; the saved analyses do not isolate a relation-only pathway.

### Strengths

- The manuscript distinguishes predictive utility from semantics and causality.
- It does not call the current C1 open-vocabulary.
- It treats the geometry implementation defect as a construction finding and the unfinished R1 endpoint as unavailable.
- It acknowledges that a residual after G+O can be complementary image-conditioned signal rather than relation-specific content.

### Major concerns

1. “Relation feature” is a project name, not proof that the tensor contains relation-specific information. Avoid wording that semantically reifies `rel_feat`.
2. WPRD within-pair discrimination is not equivalent to human semantic correctness; pair-conditioned ranking can remain sensitive to appearance and context.
3. R2's scalar offset tests predictive complementarity under a fixed parent, not conditional information in a causal or representation-theoretic sense.
4. There is no predicate-disjoint/open-vocabulary protocol or human-verified interactional evaluation; current validation has zero eligible interactional-family rows.
5. The geometry correction did not reach evaluation, so no corrected geometry performance claim is available.

### Minor concerns

- Prefer “frozen `rel_feat` predictor/log-probability offset” to “relational branch.”
- Define “conditional” as conditional on this parent, fitted protocol, and check population.
- Do not call the residual “semantic” even when its point estimate is negative LL.

### Required revisions

- Replace broad “relational quality” phrasing with “predicate-score prediction/discrimination” unless explicitly framed as the aspiration being audited.
- Add a limitations paragraph stating which nuisance channels remain uncontrolled in R2.
- Report no open-vocabulary or interactional advantage.
- State that no clean relation-only branch is established by the architecture analysis.

### Fatal concerns, if any

The missing semantic identification is fatal to a semantic-representation contribution. It is not fatal to an empirically careful measurement/audit paper, provided the title, abstract, and contribution stay within that boundary.

### Evidence needed to resolve concerns

The present review can narrow wording and expose the gap. Resolving semantic specificity would require a separately designed evaluation; it is not supplied by the existing repository and is not authorized here.
