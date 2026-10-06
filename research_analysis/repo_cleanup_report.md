# Repository cleanup report

| PATH | ACTION | REASON | REPLACED_BY | PROVENANCE_STATUS |
|---|---|---|---|---|
| 159 top-level `runs/` directories and their contents | Retained | Historical/failed/blocked/completed run status cannot safely be inferred from naming; many are untracked user artifacts | Inventory index only | UNSAFE_TO_DELETE |
| Untracked C0/C1/C1-seed5678 and evaluation outputs | Retained | Research outputs/checkpoints/logs visible in initial status; not modified by this task | None | UNSAFE_TO_DELETE |
| Pre-existing untracked `tools/l4_*` / `tests/test_l4_*` | Retained | Identity/lineage/seed audit code and tests; this task did not own or supersede them | None | CURRENT_ANALYSIS / USER DATA |
| Canonical checkpoint, datasets, source snapshot, representations | Retained unchanged | Explicit immutable provenance contract | None | FROZEN_HISTORICAL |
| `.gitignore` | No change | No isolated generated file was shown to be tracked unnecessarily; broad patterns could hide research evidence | None | No cleanup warranted |
| New `tools/portfolio_gate_analysis.py` | Added | Reproducible CPU R2 and R3 computations with identity checks | N/A | CURRENT_ANALYSIS |
| New `research_analysis/` package | Added | Human-readable, machine-readable portfolio decision record outside run logs | N/A | CURRENT_ANALYSIS |

No files were deleted, moved, overwritten, or rewritten in `runs/`, `checkpoints/`, `datasets_vg150_clean/`, or the canonical source snapshot. The existing untracked run artifacts leave `git status` nonempty; they are intentionally not staged. No cleanup candidate met the required proof of irrelevance and safety. This is a conservative, provenance-preserving cleanup rather than deletion for visual tidiness.
