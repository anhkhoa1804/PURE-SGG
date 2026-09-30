# E2 human statistical fidelity audit

The human solvability unit is the contrast block. Each block contributes the
mean of its available independent annotator both-correct indicators. The point
estimate is the mean of those block means, and the confidence interval is a
5,000-replicate bootstrap resampling blocks with seed 11. Annotator-response
Wilson intervals are retained as descriptive diagnostics only. This avoids
treating three responses on the same block as three independent scientific
units.

The analysis is currently `BLOCKED_HUMAN_INPUT` because all three annotation
files are absent. No human estimate is reported. The implementation rejects
missing/uncertain block choices from the primary human gate and reports family
support separately.
