# E2 asset migration forensic audit

| Artifact | Current status | Evidence | E2 consequence |
|---|---|---|---|
| `train.jsonl` | `PRESENT_EXACT` | local SHA256 verified | required for M3/M4/M5 fitting |
| `validation.jsonl` | `MISSING` | no local match; historical hash only | publication lineage gap; not current packet blocker |
| cached C1 pair dump | `PRESENT_EXACT` | local SHA256 verified | sufficient for M1 cached readout |
| frequency prior | `PRESENT_DIFFERENT/UNKNOWN` for historical variants | local file exists; historical manifests record another artifact | not used for E2 nuisance fitting |
| current predicate vocabulary | `PRESENT_EXACT` | local LF SHA256 verified | used for exact cached M1 mapping |
| `C1_seed1234.pt` | `MISSING` | no local or accessible project match | fresh M1 unavailable |
| selected E2 images | `PRESENT_EXACT` for packet manifest | 170 files decode successfully | sufficient for development annotation |
| full image corpus | `MISSING` | no local corpus found | not restored; unnecessary for current development |

Historical machine paths and Google Drive references are documented in the
repository, but no mounted or copied source artifact was found in the current
accessible roots. Missing artifacts remain missing rather than being
reconstructed under historical names.
