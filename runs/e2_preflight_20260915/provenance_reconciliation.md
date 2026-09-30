# E2 provenance reconciliation

The original `provenance.json` recorded the metadata-only state before selected images were restored. That historical claim is retained; the current observed state is recorded additively below.

| Claim | Observed | Status |
|---|---|---|
| images | `170` | **STALE_HISTORICAL_CLAIM** |
| candidate_sha256 | `"ca53f3c0496e5323f408a6bc84d08c7991947ecae37d7d5aec2f7b2964523f80"` | **VERIFIED** |
| image_manifest | `"d17b1ec99841da0959f5a46c4c9cee5516b497144a06a12cd72c656a0aee819b"` | **VERIFIED** |
| image_decode | `{"observed_errors": 0, "status": "VERIFIED"}` | **VERIFIED** |
| model_blind_selection | `true` | **VERIFIED** |
