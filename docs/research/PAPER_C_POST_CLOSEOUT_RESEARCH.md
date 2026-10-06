# Paper C post-closeout research analysis

Baseline: `7d6eaf2098ad601d4d573d374f99e0feccc03252` on
`research/pure-complete-readout-v2`. The closeout branch and its research
conclusion are preserved. This analysis is on `research/paper-c-next-direction`.

The current direction remains **PAPER_C_DIRECTION_CLOSED**. The branch adds a
manuscript-style synthesis, formal reviewer premortem, ranked research options,
an evaluation-first protocol proposal, reproducible tables/figures and CPU
checks of saved logits. No learned model, dataset split, human annotation,
feature extraction, C1 inference or GPU experiment was added.

The analysis bundle is
`runs/paper_c_master_research_analysis_20261006T031506Z/`; the paper drafts,
tables and figures are under `paper_package/`. Start with
`NEXT_QUESTION_SELECTION.md` and `FINAL_SCIENTIFIC_POSITION.md` in that run.
`evidence_map.json` records original sources, hashes, populations, treatments,
comparators, uncertainty and limitations. `independent_verification.json`
checks saved 51-output predictions on 1,182 / 14,991 validation identities and
250 / 3,142 pilot CAL-CHECK identities.

Two interpretive clarifications matter. The O/G+O and N4-FIT effects use
different populations, model fitting scales and offset orientations; their
magnitudes are useful context and were used for a resource rule, but they are
not a matched causal decomposition. The historical Fourier recovery audit
supports a finite-probe recoverability result, not a proof of mathematical
information destruction. The G+O combination's raw loss and accuracy worsen
while its calibrated loss improves slightly; the pilot has small improvements
in both. No historical number or report was edited.

The selected next question asks whether frozen C1 follows human-verified
contrasting relation truth with identical ordered object labels beyond frozen
geometry and appearance/context competitors. This addresses an evaluation
bottleneck: existing E2 candidates are model-blind but annotation is pending,
and the fixed residual validation has zero eligible rows for its frozen
interactional endpoint. No human result can be generated autonomously from
those metadata.

Execution decision: **NO_NEW_EXPERIMENT_JUSTIFIED_WITH_CURRENT_ASSETS**. The
best next project needs human development, adjudication, measured candidate
yield and paired discordance, a final sample-size plan and frozen controls
before scoring. It should be separately registered as a new Paper D. This
does not claim the constructive residual method is disproved or that all
future work has low value. It means another autonomous compute campaign on
the present labels does not resolve the largest scientific uncertainty.

No existing closeout text was overwritten. The manuscript makes predictive
claims only, retains the blocked fullscale nuisance question, and does not
claim semantic, causal, unseen-predicate or interactional superiority.

Validation on this branch: `.venv/bin/python -m pytest` completed with 615
passed, 1 skipped and 25 warnings; the five focused audit-builder tests also
passed after the final generated-table and whitespace changes. Python compile,
CLI help, artifact SHA checks and `git diff --check` passed. No GPU was used.
