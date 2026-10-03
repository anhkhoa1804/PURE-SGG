# Paper C experimental timeline

This is a map, not a replacement for preregistrations and result artifacts.
Historical files remain unchanged.

1. **A1–A6 and null baselines.** Registered readout ladder compared frozen
   text, linear/MLP, cosine, geometry, fusion, shuffled-label, and prior arms.
   Result: `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/`.
2. **Geometry and readout diagnostics.** Geometry decodability, geometry
   fusion, estimator matching, and later falsification work are recorded in
   dated docs and `runs/p57`–`runs/p69` families. These established important
   nuisance alternatives, not relation semantics.
3. **Readout-v2 forensics and cross-seed exploration.** Treatment fidelity,
   invalid pilot findings, and limited seed checks are preserved under the
   corresponding `docs/PAPER_C_READOUT_V2_*` and run paths. Do not collapse
   their distinct populations/protocols.
4. **M2 sampling diagnostics.** Prefix and balanced sampling were examined;
   balanced 15k used 15,000 images / 206,669 rows and scored 1,182 validation
   images / 14,991 rows. Historical prefix metrics used a different validation
   population, so they were not directly comparable without identity-safe
   rescoring.
5. **FULL/A/B interventions.** Seed-1234 smoke used FULL, geometry-at-gate A,
   and within-image pair-state alignment B. It passed its engineering smoke
   audit (`SMOKE_GO`) but was not an effect-estimation experiment. The
   matched-supervision seed-5678 replication and B2 intervention were not run.
6. **Fullscale residual audit.** The O and G+O levels showed a large and a
   small/uncertain residual contrast, respectively. Full G+O+U+S was blocked
   by missing identity-safe U/S population coverage; see
   `runs/paper_c_nuisance_ladder_v2_20261003T092837Z/`.
7. **N4 vocabulary forensic and contract amendment.** The historical
   2,907-label vocabulary/checkpoint overlapped current CAL; the current
   leakage-safe pilot therefore froze FIT-only vocabulary and a 16,900-wide
   model. See `runs/paper_c_nuisance_vocab_contract_20261003T114634Z/` and
   `runs/paper_c_n4fit_contract_20261003T115720Z/`.
8. **Bounded N4-FIT pilot and strategic stop.** 5,000 FIT / 1,000 CAL images
   were extracted; 750 CAL-FIT images calibrated and fitted alpha, while 250
   CAL-CHECK images evaluated. The exploratory delta was −.0024462230, smaller
   than the prior G+O reference, so approximately 11 GPU-hours of fullscale
   reconstruction were not justified. No validation was scored and no C1
   retraining was authorized.

Official end state: `PAPER_C_DIRECTION_CLOSED`; see
[closeout](PAPER_C_CLOSEOUT.md).
