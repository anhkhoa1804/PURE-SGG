# M2 asset forensic report

Canonical train-derived status: **NOT_AVAILABLE**.

Validation-held-out status: **AVAILABLE_CLEAN**; see
`m2_reconstruction_audit.json` and `m2_validation_heldout_decoder.pt`.

The repository and accessible project roots contain the accepted evaluation
dumps and decoder source, but no standalone train-split `rel_feat`
representation or canonical train-derived decoder head. The source-only
decoder implementation is not a fitting artifact. The cached validation dump
does support the amended `M2_valheldout` probe after excluding all 170 frozen
development and random-cohort images. Reusing accepted A3 predictions remains
prohibited.
