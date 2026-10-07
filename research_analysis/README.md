# SGG research portfolio gate

## Verdict

The portfolio decision is to **merge A/B and fold C into the measurement/audit paper**. Corrected R2 finds a bounded nested predictive increment beyond the saved G+O predictor on CAL-CHECK; it is pilot-only and not semantic evidence. The earlier `−0.622110` R2 headline is retracted for the intended contrast. WPRD remains a narrow within-pair endpoint, and its arm-correlation p-values are not model-level inference. R1 was not run in this correction; no outcome is inferred, and no semantic or open-vocabulary result is established.

## What was investigated

This gate asks whether C1’s geometry result is separable from nuisance prediction and whether WPRD is sufficiently interpretable to evaluate one controlled geometry condition. The research program’s prior evidence includes frozen rel_feat readouts, pair priors, geometry controls, WPRD audits, interventions, and a bounded N4-FIT pilot.

## Newly tested

R2 reuses the FIT-only pair prior, the registered FIT-only G+O model, and existing bounded pilot predictions. Parent temperatures and a nonnegative residual coefficient are fit only on 750 CAL-FIT images; all four arms are evaluated on the same 250-image / 3,142-row CAL-CHECK set. R3 audits saved WPRD tables and their dependence limits. No R1 experiment was launched in this correction.

## Headline R2 result

| CAL-CHECK contrast | Calibrated parent LL | Calibrated extended LL | Delta (image-cluster exploratory 95% CI) |
|---|---:|---:|---:|
| O → O + rel_feat | 1.598961 | 1.280646 | −0.318314 [−0.357666, −0.276130] |
| G+O → G+O + rel_feat | 1.315260 | 1.255252 | −0.060008 [−0.074317, −0.044480] |

The former raw O/CAL-CHECK delta `−0.622110` is retained only as a retracted historical computation, not as an R2 headline. Accuracy rises while macro recall falls in both corrected contrasts; the gain is not uniform. These are pilot-only results and do not isolate appearance/context or establish semantics.

## Prior evidence and boundary

- O→O+rel_feat: held-out log-loss delta −0.366033, image CI [−0.396638, −0.336165].
- G+O→G+O+rel_feat: −0.005129, CI [−0.026360, +0.015863].
- N4-FIT pilot: −0.002446, exploratory CI [−0.002627, −0.002284]; pilot-only and not full-population inference.
- Full 83k-image N4 was not run. C1 residual training, seed 5678 and B2 remain unauthorized.

## What remains uncertain

R2 only conditions on ordered object-pair counts; the residual can include geometry, appearance, scene context, and other image cues. WPRD has only one underlying checkpoint/cache family in its twelve scoring arms. The previous C0→C1 training intervention bundled geometry units and Fourier scale. There is no final interactional-family endpoint or human semantic test.

## Publication consequence

Current portfolio recommendation: `MERGE_A_B_AND_FOLD_C`. Do not present WPRD as a general SGG quality metric or use R2 as evidence of semantic relation representation. See [FINAL_REPORT.md](FINAL_REPORT.md), [portfolio_gate_decision.md](portfolio_gate_decision.md), [R2](R2_nested_nuisance_analysis.md), and [R3](R3_WPRD_construct_validity.md).

## Reproduction

CPU-only corrected R2 computation: `CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/r2_nested_correction.py`. It reads frozen manifests, canonical training JSONL, saved FIT model weights, and saved pilot predictions; validation is used only to reproduce saved parent metrics. Exact identities, hashes, split counts, and CI replicates are recorded in `R2_nested_nuisance_analysis.json`.
