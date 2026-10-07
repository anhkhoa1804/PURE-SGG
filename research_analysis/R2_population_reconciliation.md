# R2 Population Reconciliation

The rows below are distinct analyses, not interchangeable estimates.

| Quantity | Old result | Current previous-R2 result | Same population? | Same labels? | Same object-pair support? | Same backoff? | Same calibration? | Same split? | Same row count? | Same image count? | Same fitting regime? | Comparable? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| O pair prior LL | 1.632362; validation, 14,991 rows / 1,182 images | 1.807878; CAL-CHECK, 3,142 rows / 250 images | No | Yes, same frozen 51 classes | Same lookup/rule; supported rows differ with population | Yes, FIT-global Laplace-1 | No: full-CAL T vs raw | No | No | No | No: count table same, calibration regime differs | No |
| O + rel_feat LL | 1.266329; validation, 14,991 / 1,182 | 1.185767; CAL-CHECK, 3,142 / 250 | No | Yes, same 51 classes | Same O support rule; supported rows differ | Yes | No: full-CAL calibration vs CAL-FIT fitted offset and uncalibrated O | No | No | No | No: calibration/fitting regime differs | No |

The previous O-only result is the same underlying FIT count lookup and backoff as the canonical pair prior, but it is an *uncalibrated score on a different held-out population*. It is worse numerically than the calibrated validation result without implying a changed or degraded prior. The earlier O+rel values also differ in evaluation population and calibration regime. Neither pair of numbers may be directly subtracted.

The corrected R2 instead uses the same CAL-CHECK rows for O, G+O, O+rel_feat, and G+O+rel_feat; the parent logits are temperature-calibrated on CAL-FIT and remain fixed. Full structured fields and the historical retraction are in [R2_population_reconciliation.json](R2_population_reconciliation.json).
