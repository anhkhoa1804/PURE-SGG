# Hostile Reviewer Attack

## 1. “The result is just object-pair prior.”

- **Evidence supporting objection:** The ordered pair prior has strong held-out predictive performance; see `runs/paper_c_fullscale_residual_audit_20261003T084403Z/` and R2 source summaries.
- **Fatal?** Major, not fatal to a predictive audit paper.
- **Current answer:** Corrected nested R2 fixes the O/G+O parent and tests an offset. The G+O + `rel_feat` arm improves calibrated LL by 0.060008 nats/row on CAL-CHECK.
- **Missing evidence:** Full-scale identity-matched semantic evaluation and full G+O+U+S comparison.
- **Manuscript mitigation:** Present O and G+O as strong controls; claim only bounded incremental prediction.

## 2. “WPRD is not a valid measure.”

- **Evidence supporting objection:** It uses eligible within-pair cells and a row cap; test cell vectors are unavailable; general construct validity is incomplete.
- **Fatal?** Major for broad metric claims; not fatal if narrowly defined.
- **Current answer:** Pair-prior-only score is 0.5 and WPRD is coherent as conditional within-pair score discrimination.
- **Missing evidence:** Planted-shortcut and strong-VLM ceiling checks; independent model-family construct validation.
- **Mitigation:** Define the exact statistic and eligible population; do not call it semantic quality.

## 3. “The correlation p-values are pseudoreplicated.”

- **Evidence:** n=12 deterministic arms from shared checkpoint/cache lineage; `research_analysis/R3_WPRD_construct_validity.md` §4.
- **Fatal?** Fatal to inferential correlation claims; not to descriptive table reporting.
- **Current answer:** We agree; p-values are not independent-model inference.
- **Missing evidence:** Independent model-family sample and dependence-aware model-level uncertainty.
- **Mitigation:** Report correlations descriptively or omit them from the main result.

## 4. “CAL-CHECK is too small for R2.”

- **Evidence:** 250 images / 3,142 rows; single fitted pipeline.
- **Fatal?** Major for population-wide claims; not fatal to an explicitly exploratory bounded estimate.
- **Current answer:** The paired image-cluster interval is exploratory and conditional; it does not quantify training-seed uncertainty.
- **Missing evidence:** Independent validation population and fit replication, neither available under the closed protocol.
- **Mitigation:** Keep pilot population in title/abstract/results and avoid confirmatory language.

## 5. “The interval excludes zero, so why call it exploratory?”

- **Evidence:** Corrected R2 G+O contrast interval is [−0.074317, −0.044480].
- **Fatal?** Not fatal; interpretation risk is major if mislabeled.
- **Current answer:** The interval is conditional on one selected analysis, fit and split; the research path was exploratory and no seed uncertainty is represented.
- **Missing evidence:** Preregistered independent replication.
- **Mitigation:** Label “exploratory image-cluster 95% interval,” state what is conditioned on, and do not use it to assert generalization.

## 6. “The gain is concentrated in head predicates.”

- **Evidence:** Highest-support classes dominate row counts; per-class changes vary. For G+O, `on` support is 1,124, `has` 418, `in` 404; see CSV.
- **Fatal?** Major for a broad class-general claim.
- **Current answer:** We do not establish a head-predicate mechanism. Support ordering is descriptive only; there are no predeclared frequency groups.
- **Missing evidence:** A preregistered, independently powered frequency-stratified analysis.
- **Mitigation:** Show the full per-predicate table and report heterogeneous metrics, no uniform gain.

## 7. “Geometry is confounded and the corrected run is absent.”

- **Evidence:** Historical comparisons bundled input units/Fourier settings; R1 stopped before endpoint.
- **Fatal?** Major for geometry performance claims.
- **Current answer:** The units-contract defect is verified, but its corrected effect is unknown.
- **Missing evidence:** Completed R1 endpoint; explicitly not part of this paper-construction task.
- **Mitigation:** Make geometry a measurement/input-visibility audit and disclose the blocked R1.

## 8. “A same-seed rerun cannot estimate robustness.”

- **Evidence:** R1 had only seed 1234 and did not complete.
- **Fatal?** Not applicable to a result that is not claimed; major if claiming robustness.
- **Current answer:** No corrected performance estimate and no between-seed uncertainty are reported.
- **Missing evidence:** Independent seeds; not authorized.
- **Mitigation:** Avoid robust/reproducible geometry-effect wording.

## 9. “You are equating predictive residue with semantics.”

- **Evidence:** R2 optimizes log-loss on predicate labels; no human semantic ground truth.
- **Fatal?** Fatal to semantic conclusion.
- **Current answer:** Predictive utility is the estimand; semantics remain unestablished.
- **Missing evidence:** Human-verified semantic evaluation capable of distinguishing relation from appearance/context shortcuts.
- **Mitigation:** Explicit distinction throughout abstract and discussion.

## 10. “Open-vocabulary claims are unsupported.”

- **Evidence:** No predicate-disjoint evaluation; supervised predicate labels overlap evaluation.
- **Fatal?** Fatal to an open-vocabulary claim.
- **Current answer:** We make no such claim.
- **Missing evidence:** A separately registered unseen-predicate protocol; not authorized here.
- **Mitigation:** Remove open-vocabulary language except as a limitation.

## 11. “The corrected nested model could still capitalize on calibration.”

- **Evidence:** R2's final metric is calibrated; calibration and alpha are fit on CAL-FIT.
- **Fatal?** Major methodological concern, addressed but not eliminated.
- **Current answer:** Parent and extension use same fitting split; alpha=0 recovers parent; selected CAL-FIT objective is no worse; CAL-CHECK is untouched for fit. Primary contrast also has raw LL delta −0.032009, with exploratory image CI [−0.046121,−0.016764].
- **Missing evidence:** External independent confirmation; all numbers remain conditional on this split/model fit.
- **Mitigation:** Report raw and final metrics and precisely describe the calibration procedure.

## 12. “This is several populations presented as one ladder.”

- **Evidence:** Historical WPRD, full validation nuisance results, corrected R2 CAL-CHECK, and N4-FIT pilot have different samples and fitting protocols.
- **Fatal?** Major if directly ranked; manageable with explicit provenance.
- **Current answer:** The manuscript separates historical, full-validation, and pilot evidence; no cross-population effect decomposition is claimed.
- **Missing evidence:** A fully matched full-scale nuisance ladder.
- **Mitigation:** Put population and status next to every value; avoid a single monotone ladder plot across mismatched samples.
