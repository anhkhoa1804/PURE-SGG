# Portfolio gate decision

## Decision: `R1_BLOCKED`

R2 completed without identity or split failures. A FIT-only ordered-pair prior was held fixed, a single residual coefficient was calibrated on CAL-FIT, and O-only versus O+rel_feat was scored on disjoint CAL-CHECK rows. This resolves the stated independent-decoder identification concern for the pair-prior contrast.

R3 does **not** validate WPRD as a general SGG quality measure. It supports the narrower WPRD construct as within-pair predicate discrimination: the object-pair prior is exactly chance, its statistic is explicitly conditioned within ordered object-class groups, and the saved p70 geometry removal changes the metric on paired cells. This is enough to interpret one narrow geometry comparison, not to make semantic claims.

The prior C0→C1 training intervention changed both `geom_input_pixel_space` and `geom_fourier_scale`. Existing launcher `scripts/train/run_c0_c1.sh` defines C1a as `pixel_space=true, Fourier_scale=1.0`, which differs from C0 only in pixel-space geometry input. The confound is concrete; a single C1a run at seed 1234, followed by the existing full-split endpoint and paired comparison against frozen C0, can determine whether the geometry-only condition moves WPRD. It is bounded to one training arm and one evaluation, no sweep or second seed.

The narrow R1 question is scientifically justified, but the required fresh L4 check found an active workload (PID 24793, 5,802/23,034 MiB, 15% utilization). Per the no-competition rule the run was not launched and was not retried. Thus R1 is execution-blocked, not a null result. Its predeclared protocol and interpretation rule are preserved in `R1_fixed_geometry_C1.md` and its machine-readable companion.

Portfolio decision for the current evidence: **`MERGE_A_B_AND_FOLD_C`**. Without the geometry-only result, C1’s geometry effect remains the already-reported weak bundled effect and should not be advanced as a separate paper. Do not infer that C1a would be null or positive.
