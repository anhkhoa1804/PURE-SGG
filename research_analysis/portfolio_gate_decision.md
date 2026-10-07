# Portfolio gate decision

## Decision: `R1_BLOCKED`; portfolio `MERGE_A_B_AND_FOLD_C`

The earlier R2 headline `ΔLL = −0.622110` is retracted for the intended nested question. It used raw O logits, omitted the primary G+O contrast, and its O-only number was incorrectly compared in prose to a calibrated full-validation O result.

Corrected R2 uses fixed FIT-trained O and G+O parent scores, temperatures and one nonnegative rel_feat offset fitted only on CAL-FIT, and the same 250-image / 3,142-row CAL-CHECK for all arms. The primary G+O→G+O+rel_feat calibrated LL change is **−0.060008** (image-cluster exploratory 95% CI **[−0.074317, −0.044480]**). Accuracy rises slightly, but macro recall falls slightly; this is not a uniform predicate gain. The secondary O→O+rel_feat change is **−0.318314** (CI **[−0.357666, −0.276130]**). See `R2_nested_nuisance_analysis.md` and its JSON/CSV for exact values and protocol.

This supports a bounded pilot predictive increment for the frozen readout beyond the registered G+O predictor. It does not test U/S, does not give full-population inference, and does not establish semantic specificity. U/S remain unavailable under the clean identity-safe protocol.

R3 completeness is mixed: exact correlation sample size is recovered (12 dependent scoring arms, one underlying checkpoint/cache family); prior-only and random-null controls exist; planted-shortcut and strong-VLM ceiling controls are absent; dependence-aware inferential treatment remains incomplete. WPRD remains interpretable only as a narrow within-pair discrimination statistic.

R1 was not run and is outside this correction. The previous R1 resource gate recorded active PID 24793; do not terminate it without first confirming command, owner/process tree, and relation to this workload. Any future R1 must have a separately frozen same-seed fixed-geometry design, preregistered success margin/metric/stopping rule/baseline/artifact paths, and a fresh `nvidia-smi` gate. The prior R1 blocked status is not a null result, and this correction does not authorize R1.

Portfolio decision remains **`MERGE_A_B_AND_FOLD_C`**: the corrected bounded R2 estimate does not establish the distinct geometry contribution needed for a separate C paper, while the WPRD validity limitations remain. Do not start a new experiment campaign from this correction.
