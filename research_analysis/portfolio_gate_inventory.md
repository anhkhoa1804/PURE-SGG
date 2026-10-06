# Portfolio gate inventory

Initial state: branch `research/paper-c-next-direction`, HEAD `0b306a4020e41356f29e5053a902b065d6ae94e1`, tracking `origin/research/paper-c-next-direction`. The tracked worktree was clean. The repository contains 498 tracked files and 159 top-level directories under `runs/`; the initial status also showed many untracked run outputs and markers. Those are retained as user research data.

| Area | Classification | Findings / action |
|---|---|---|
| Canonical source, checkpoint, train representation and datasets | FROZEN_HISTORICAL | Immutable; hash/commit checked before closeout. |
| `openvocab_rel/` | ACTIVE | Model, geometry, training and configuration implementation; no algorithm changes in this gate. |
| `tools/within_pair_discrimination.py`, geometry evaluators, prior/residual tooling | CURRENT_ANALYSIS / HISTORICAL | Source methods retained; historical behavior is not edited. |
| `runs/paper_c_fullscale_residual_audit_*`, N4-FIT and A/B smoke | FROZEN_HISTORICAL | Manifests, predictions, reports and negative/blocked outcomes are evidence. |
| `runs/p49`, `p53`, `p61`, `p70` | FROZEN_HISTORICAL | Sources for the WPRD dependence and intervention audit. |
| Tests | ACTIVE | Existing geometry contract tests already cover pixel-scale units and normalized-box collapse; targeted suite was run before this task. |
| Untracked `tools/l4_*` and `tests/test_l4_*` | CURRENT_ANALYSIS | Pre-existing identity/lineage/seed audit utilities; preserved and included in full CPU test discovery, not modified. |
| New `tools/portfolio_gate_analysis.py` | CURRENT_ANALYSIS | Reproducible CPU-only R2/R3 computations; no validation scoring. |
| Untracked run directories / checkpoints | UNSAFE_TO_DELETE | Preserved because their provenance status cannot be downgraded from a status listing alone. |
| Transient caches/build files | CANDIDATE_FOR_CLEANUP | No clearly isolated tracked candidate found that is safe to delete; no broad `.gitignore` change. |

Detailed paths and classifications are in [portfolio_gate_inventory.json](portfolio_gate_inventory.json). No historical result or checkpoint was modified or removed.
