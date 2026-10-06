# Paper C status

**Official state: `PAPER_C_DIRECTION_CLOSED` (2026-10-03).** The current
research direction is complete as an exploratory diagnostic program. No next
Paper C experiment is authorized by the recorded evidence.

## Final scope

The work examined whether frozen predicate-supervised `rel_feat` is decodable
and whether its predictive contribution persists after nuisance predictors
are added. The evidence is conditional on specified populations, feature
pipelines, and decoder families. It does not identify semantic relational
content.

## State of the evidence

- WPRD readouts A1–A6 show above-null decodability under the registered
  evaluation, including geometry-only performance and a geometry-plus-feature
  fusion arm. These are diagnostic readout results, not proof of relation
  semantics.
- `O + rel_feat` reduced calibrated held-out log-loss by 0.3660331323 nats/row
  (image-clustered 95% CI [−0.3966375880, −0.3361649032]).
- `G+O + rel_feat` changed calibrated held-out log-loss by −0.0051288329
  (95% CI [−0.0263599217, +0.0158628067]); the interval includes zero.
- The final identity-checked pair-prior audit supports 12,330 / 14,991
  validation rows (82.25%). The earlier 12,402 summary is superseded; see
  `runs/paper_c_final_evidence_audit_20261003T185925Z/posthoc_metrics.json`
  and the consistency note in `evidence_index.json`.
- The bounded FIT-only N4-FIT pilot reported 1.5984459578 versus 1.5959997348
  CAL-CHECK log-loss, delta −0.0024462230, with an exploratory image-cluster
  interval [−0.0026268349, −0.0022835433]. Its resource rule did not justify
  fullscale replay.
- Full-population N4-FIT was not run; therefore residual beyond full-population
  G+O+U+S remains **not established**.
- Seed-1234 FULL/A/B completed as a smoke/engineering check. It was not a
  confirmatory performance experiment. No seed-5678 replication or B2 run is
  included in the authorized final evidence.
- The current fixed validation set has zero eligible rows for the frozen
  interactional-family endpoint.

## Prohibited conclusions

Do not state that this work proves relational semantics, causal information,
open-vocabulary relation understanding, interactional-family superiority, or
information unavailable from the image. See the
[claim audit](PAPER_C_CLAIM_AUDIT.md) and
[closeout](PAPER_C_CLOSEOUT.md).
