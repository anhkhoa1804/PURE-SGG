# E2 M3/M4/M5 training-lineage audit

The earlier scorer fit all three nuisance baselines from the cached validation
dump while excluding accepted image IDs. That is a holdout-style diagnostic,
but it does not satisfy the registered train-only lineage rule. This was a
real pre-scoring defect.

The scorer is now repaired: `--train` defaults to
`datasets_vg150_clean/train.jsonl`, and M3 geometry class means, M4 ordered
noun backoff counts, and M5 relation frequencies are fit only by streaming
that declared artifact. If an evaluation image ID appears in that source, the
iterator fails loudly. The cached validation dump remains read-only and is
used only for M1 lookup and accepted-block evaluation metadata.

The current train artifact is present with SHA256
`306fc0db96ca716bbc78c2cf32d49fdc695ab0074004a99c7399c850a1c605a4`.
