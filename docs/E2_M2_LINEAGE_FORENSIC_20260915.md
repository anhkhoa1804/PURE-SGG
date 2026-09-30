# E2 M2 lineage forensic audit

Original search status: `NOT_AVAILABLE` for a canonical train-derived M2.
Corrected status after inspecting the cached dump: `AVAILABLE_CLEAN` for a
validation-population held-out probe; `NOT_AVAILABLE` for a canonical
train-derived model.

The repository contains the accepted cached C1/R0 pair dump and the matched R2
pair dump. No standalone train-split `rel_feat` tensor, decoder head, or
serialized canonical train-derived fit was found under the repository or the
accessible project root. The representation decoder ladder source documents a
separate train JSONL input, but source code is not a saved training
representation.

The accepted C1 dump nevertheless contains `rel_feat`, image IDs, pair slots,
GT predicate targets, and the exact 51-column vocabulary for all 10,401
validation images. Excluding all 170 unique development and random-cohort
images in the frozen E2 packet leaves 10,231 fitting images and 129,826 finite
768-dimensional target rows. The executable audit at
`runs/e2_preflight_20260915/m2_reconstruction_audit.json` resolves all 90
development candidate sides and verifies that duplicate pair slots are
content-identical.

The consequence is now narrow and explicit: M2 cannot be reported as a
canonical train-derived or train-to-validation generalizing model. A clean
`M2_valheldout` can be fit and scored as a validation-population held-out E2
probe, provided it is frozen before accepted-block scoring. This does not block
human annotation or packet freezing and does not require restoring the C1
checkpoint.

The current required fit artifact and audit are the pre-scoring outputs under
`runs/e2_preflight_20260915/`. Reusing accepted A3 predictions remains
prohibited. A future publication-grade canonical M2 would still require a
train-split representation dump or fresh C1 checkpoint, but that asset is not
required for the amended E2 held-out probe.
