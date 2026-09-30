# Paper C E2 implementation-fidelity audit

Authority for the prospective development packet is the 2026-09-12 additive
population redesign amendment. Earlier draft clauses that require a hard
geometry-matched primary population are superseded for development; the
geometry vector remains a recorded nuisance baseline.

| Protocol requirement | Intended rule | Actual implementation | Status | Evidence | Required fix |
|---|---|---|---|---|---|
| Frozen population | 45 model-blind blocks | frozen manifest and packet hashes match | PASS | `FROZEN_CANDIDATE_MANIFEST.json` | none |
| Image independence | no dev reuse or dev/random overlap | executable block audit passes | PASS | `block_independence_report.json` | none |
| Candidate selection | no model outputs | field/AST leakage audit passes | PASS | `candidate_leakage_report.json` | rerun preflight after packet changes |
| Relation whitelist | six frozen pairs | packet and selection flow match | PASS | frozen manifest | none |
| Human truth | three independent annotators | files absent; app isolates current annotator | BLOCKED_HUMAN_INPUT | annotation directory | complete human input |
| Adjudication | no silent majority promotion | accepted state requires single-valid visible truth | PASS | `tools/e2_adjudicate.py` | retain raw records |
| Human inference | block-clustered | block means are bootstrap units | PASS | `tools/e2_human_analysis.py` | none |
| M1 | cached exact pair/vocabulary lookup | implemented as `M1_cached_C1`; identical duplicate cache slots are resolved only after channel equality verification | PASS | `tools/e2_score_models.py`, `tools/e2_cached_dump.py` | no fresh-checkpoint substitution |
| M2 | fresh clean held-out representation fit | `M2_valheldout` fit from 129,826 cached validation rows after excluding all 170 frozen development/random-cohort images; canonical train-derived M2 remains unavailable | PASS WITH AMENDMENT | `m2_reconstruction_audit.json`, `m2_validation_heldout_decoder.pt`, additive lineage amendment | use amended name and claim boundary |
| M3 | train-only geometry nuisance fit | repaired to explicit `--train` JSONL | PASS | scorer and geometry tests | score only after gold freeze |
| M4/M5 | train-only object/frequency fit | repaired to explicit `--train` JSONL | PASS | scorer and lineage tests | score only after gold freeze |
| Geometry claim | measured nuisance, not removed | 19D vector retained and scored | PASS | amendment and scorer | no causal wording |
| Provenance | current image state accurate | stale historical claim retained, current state added | PASS | reconciliation report | none |
| GPU policy | no GPU during preflight | CPU-only execution | PASS | environment capture | M6 requires bounded pilot later |

The canonical train-derived M2 lineage remains unavailable, but that is no
longer a blocker for the amended E2 focal comparison. The amended comparison
uses `M2_valheldout` versus M3 and M4, with the narrower validation-population
held-out interpretation recorded in
`docs/PAPER_C_E2_M2_LINEAGE_AMENDMENT_20260915.md`.
