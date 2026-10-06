# Paper C post-closeout research analysis

This additive bundle is a manuscript and research-decision reconstruction from existing evidence. It uses CPU saved-logit consistency checks and reproducible tables/figures. No learned model, split, annotation, inference, extraction or GPU experiment is added.

Start with `paper_style_analysis.md`, `hostile_review.md`, `NEXT_QUESTION_SELECTION.md`, and `FINAL_SCIENTIFIC_POSITION.md`. The paper-oriented uppercase drafts, tables and figures are in repository-root `paper_package/`. `evidence_map.json` contains original artifact hashes and population/estimator limits; `independent_verification.json` records independent saved-logit metric and identity checks.

Decision: no autonomous experiment with current assets. The selected future question requires human gold and a frozen powered evaluation population. Paper C stays closed; the evaluation-first question belongs to a separately registered project. No claim of no residual or equivalence is made.

Reproduce into NEW, nonexistent destinations using:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/build_paper_c_master_analysis.py --output runs/paper_c_master_research_analysis_<UTC_TIMESTAMP> --package <new_package_path>
```

The builder refuses existing output paths and checks all source and immutable artifacts before writing. Analysis inputs must be available locally; a Git-only clone does not include all payloads.
