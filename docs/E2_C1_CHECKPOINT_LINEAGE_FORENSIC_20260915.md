# E2 C1 checkpoint lineage forensic audit

`checkpoints/C1_seed1234.pt` with expected hash
`79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854` is absent
from accessible storage. The available demo checkpoint is a different
historical artifact and is not an allowed substitution.

Current E2 development can use the accepted `runs/eval_C1/pair_logits.pt`
directly as `M1_cached_C1`; fresh C1 inference is not required for that
descriptive readout. The checkpoint is therefore
`NOT_REQUIRED_FOR_CURRENT_DEVELOPMENT`. It is `REQUIRED_BEFORE_FINAL` only if
fresh M1 image inference is made part of the final estimand or if publication
requires a fully reconstructible fresh-inference lineage.
