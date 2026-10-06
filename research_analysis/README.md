# SGG research portfolio gate

## Verdict

The portfolio decision is to **merge A/B and fold C into the measurement/audit paper**. R2 found predictive residual beyond an ordered noun-pair prior on a bounded held-out set. WPRD is interpretable for the narrow within-pair endpoint, but its arm-correlation p-values are not model-level inference. A single C1a isolation run was scientifically justified and registered, but is **blocked** because the fresh L4 check found an active workload. No R1 outcome is inferred, and no semantic or open-vocabulary result is established.

## What was investigated

This gate asks whether C1’s geometry result is separable from nuisance prediction and whether WPRD is sufficiently interpretable to evaluate one controlled geometry condition. The research program’s prior evidence includes frozen rel_feat readouts, pair priors, geometry controls, WPRD audits, interventions, and a bounded N4-FIT pilot.

## Newly tested

R2 reuses the frozen full-scale FIT/CAL image split and existing pilot predictions. A 51-class ordered pair prior is fit only on FIT; one nonnegative residual coefficient is fit only on 750 CAL-FIT images; evaluation uses 250 disjoint CAL-CHECK images. R3 rechecks saved WPRD tables and computes descriptive leave-one-arm-out correlation ranges. R1 was registered as a single C1a condition; its pre-run resource gate found an active L4 workload, so it was not launched.

## Headline R2 result

| CAL-CHECK | O only | O + rel_feat | Delta |
|---|---:|---:|---:|
| Log-loss | 1.807878 | 1.185767 | −0.622110 |
| Accuracy | 0.663590 | 0.679504 | +0.015913 |
| Macro recall | 0.227013 | 0.184865 | −0.042148 |

Image-cluster 95% CI for log-loss delta: [−0.672765, −0.569993]. Bounded-pilot result; it does not isolate geometry, appearance, or semantics.

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

CPU-only R2/R3 computation: `.venv/bin/python tools/portfolio_gate_analysis.py`. It reads frozen manifests, canonical training JSONL, and saved pilot predictions; it does not read validation outcomes. Exact artifact hashes and split counts are recorded in the JSON outputs.
