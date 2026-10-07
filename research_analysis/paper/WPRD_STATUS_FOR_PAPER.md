# WPRD Status for Paper

## Editorial verdict

Retain WPRD only as a specialized within-pair discrimination endpoint, with its eligible population and dependence limitations stated. Remove or qualify any claim that it is a general SGG quality or semantic-understanding metric. Do not use cross-arm p-values as independent-model evidence or upgrade Paper A from those correlations alone.

| Audit item | Status | Verified evidence |
|---|---|---|
| Spearman effective population | 12 complete deterministic scoring arms in each saved correlation table | `runs/p49_metric_grounding/corr.json`, `runs/p53_metric_grounding_vg150only/corr.json`, `runs/p61_test_metric_grounding/corr.json`; R3 audit §3 |
| Independence/dependence | Dependent arms, shared checkpoint/cache lineage; not 12 independent model fits | `research_analysis/R3_WPRD_construct_validity.md` §§3–4 |
| Planted-shortcut sensitivity | NOT DONE | No matched planted relational signal artifact; R3 completeness table |
| Strong-VLM ceiling | NOT AVAILABLE | No strong-VLM scoring arm or construct-validity ceiling artifact; R3 completeness table |
| Dependence-aware significance | PARTIAL | Leave-one-arm-out descriptive checks exist; no independent-model cluster inference; historical arm-permutation p-values do not preserve scorer-family dependence |
| Prior/null controls | DONE, narrow | Prior-only WPRD = 0.5; random null near 0.5 in the three stored tables; only object-pair-constant shortcut is controlled |
| Geometry responsiveness | Diagnostic exists | p70 no-geometry comparison uses 20,016 paired validation cells, ΔWPRD −0.008675; does not establish semantic validity (`runs/p70_geometry_causal_ablation/ablation_wprd.json`) |
| Eligible sample details | Partial by split | Historical validation reports 20,016 cells; test correlation JSON omits per-arm cells/cell vectors, so exact test cell population cannot be recovered from that JSON alone |

The often-quoted held-out-test correlations are +0.9142 for R@50 and −0.7273 for mR@50, with nominal arm-permutation p values in the historical artifact. Their analysis unit is the scoring arm (n=12), and the arms share lineage. Correlation is descriptive; the p-values are not valid independent-model inference. Source: `runs/paper_c_final_evidence_audit_20261003T185925Z/` and `research_analysis/R3_WPRD_construct_validity.md` §2.

## Hard editorial consequence

- **Can remain:** “WPRD measures within-pair predicate-score discrimination over eligible cells under this implementation.”
- **Must be removed:** “WPRD proves semantic relational understanding,” or that arm correlations establish an independent relationship with general SGG quality.
- **Must be qualified:** the association of WPRD with R@50/mR@50, prior robustness, and geometry response. Identify n=12 dependent arms, shared lineage, and missing construct-validity controls.
