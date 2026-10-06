# R2 — Nested nuisance analysis

## Design fixed before scoring

The comparison is nested in the predictive-logit sense: an ordered subject/object category pair prior is fit as a count table on the registered full FIT images, then held fixed. The only added parameter is a nonnegative scalar multiplying the frozen rel_feat predictor’s log-probabilities. That scalar is fit on CAL-FIT and evaluated on image-disjoint CAL-CHECK. Thus the O-only model is exactly the nested model at alpha zero; this avoids comparing independently fit decoder architectures. It does not claim the pair prior is statistically optimal or that the residual is semantic.

## Populations and identity rules

| Population | Images | Rows | Use |
|---|---:|---:|---|
| Registered FIT | 75,066 | 943,815 | Fit the pair-prior count table only |
| CAL-FIT | 750 | 9,078 | Fit alpha only |
| CAL-CHECK | 250 | 3,142 | Held-out evaluation only |

FIT/CAL-FIT/CAL-CHECK are image-disjoint under the frozen manifests. CAL-FIT and CAL-CHECK were also checked for pair-row identity disjointness by exact key joins. No validation outcome or E2 row was used. Source relation keys are `(image_id, subject_index, object_index, normalized predicate)`; object-pair predictors use the ordered lowercased first object labels. Saved prediction targets were checked against the frozen 51-class predicate map.

The pair prior uses Laplace alpha=1 over the frozen 51-class vocabulary. Unseen ordered object pairs back off to the FIT-global 51-way predicate prior with the same smoothing. CAL-CHECK support: **2,610/3,142** rows; 532 rows back off. CAL-FIT support: 7,445/9,078.

## Model and calibration

`z_O = log P_FIT(y | ordered subject class, object class)`. `z_nested = z_O + alpha * log_softmax(z_relfeat)`. Alpha is one nonnegative scalar fit by LBFGS on CAL-FIT (softplus parameterization; same optimizer settings as the registered scalar-alpha pilot routine). No extra temperature is fitted; this keeps the O logits fixed and makes the nested comparison directly interpretable. The saved CAL-CHECK sidecar omitted standalone rel_feat logits, so its frozen log-probability offset was algebraically recovered from saved raw N4/combined logits and CAL-FIT-only frozen alpha/temperature. This inversion is row-shift invariant for softmax/CE and is recorded in the analysis code; no CHECK labels enter it.

The count table contained **197,661** observed ordered noun pairs (10,080,711 smoothed probability entries; 9,883,050 free probability degrees of freedom) plus the FIT-global backoff. Added residual capacity: one scalar. Seed for the image-cluster bootstrap: **20261003**, 2,000 replicates, paired row-loss differences, image resampling with all rows retained.

## Results

| CAL-CHECK metric | O only | O + frozen rel_feat | Difference |
|---|---:|---:|---:|
| Log-loss | 1.807878 | 1.185767 | **−0.622110** |
| Accuracy | 0.663590 | 0.679504 | +0.015913 |
| Macro recall (46 observed predicate classes) | 0.227013 | 0.184865 | −0.042148 |

Fitted alpha = **0.3976523**. Image-cluster percentile 95% CI for held-out row-mean log-loss difference: **[−0.672765, −0.569993]**. The contrast is large for log-loss and accuracy, while macro recall declines; therefore this is not a uniform tail-class improvement.

## Interpretation and limitations

R2 establishes, on this bounded CAL-CHECK population and for this frozen readout, conditional predictive utility beyond an FIT-only ordered pair prior under a genuinely nested score construction. It removes the independent-decoder parameterization objection for this contrast. It does not separate appearance, scene context, geometry, annotation co-occurrence, or other visual cues from relation-specific signal. CAL-CHECK is only 250 images and the pair prior was fitted on a larger population than the bounded N4-FIT pilot model; the estimand is the O-prior contrast specified here, not full N4. The result does not establish semantics or causality.

This strengthens the case that C1 representations contain usable within-image predictive signal beyond object-pair counts. It does not alone justify a separate geometry paper. The R1 decision also depends on whether WPRD can support the narrow geometry endpoint and whether the prior C0/C1 intervention bundled factors.

Machine-readable details, exact source hashes, counts and CI replicates are in [R2_nested_nuisance_analysis.json](R2_nested_nuisance_analysis.json); computation is reproducible with `tools/portfolio_gate_analysis.py`.
