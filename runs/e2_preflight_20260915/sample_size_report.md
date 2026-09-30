# E2 sample-size preflight

The paired simulation uses 100,000 replicates, seed `20260911`, two-sided
alpha `0.05`, target paired gain `0.05`, 80% power, expected 95% CI half-width
`0.05`, and N rounded to the smallest multiple of 25. The paired outcome is
simulated through four joint cells, so baseline/model responses are not treated
as independent Bernoulli samples.

For the grid used here, required N ranges from 325 to 1,250 across plausible
baseline and discordance settings. At baseline 0.60, delta 0.05, required N
is 350, 625, 950, and 1,250 for discordance 0.10, 0.20, 0.30, and 0.40.
The 45-block development packet is therefore an instrument check and cannot
support the final confirmatory sample by itself. Final N must be fixed after
human-only attrition and discordance estimates, before focal model scoring.
