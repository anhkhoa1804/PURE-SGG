# E2 development candidate packet

Status: **annotation pending; not scientific result evidence**.

This packet was generated from the accepted C1 pair dump using cached image IDs, object labels, predicate labels, and boxes only. No model scores, WPRD values, decoder errors, or VLM outputs were read by candidate construction.

- `development_candidates.jsonl`: 45 deterministic candidate blocks.
- `random_cohort_candidates.jsonl`: 40 deterministic comparison candidates.
- `candidate_pool.jsonl`: full model-blind whitelist candidate pool.
- `human_annotation_tasks.jsonl`: 270 pending image-level tasks (3 annotators × 90 images).
- `human_block_tasks.jsonl`: 135 pending 2×2 solvability tasks (3 annotators × 45 blocks).
- `FROZEN_CANDIDATE_MANIFEST.json`: immutable candidate and image identity manifest.
- `annotation_validation.json`: current validation status; currently `ANNOTATION_PENDING`.
- `overlays/`: highlighted subject/object images (`S` red, `O` blue).
- `images/image_manifest.json`: per-image source URL, size, and SHA256.
- `manifest_sha256.txt`: hashes for packet artifacts.

The full validation JSONL is unavailable locally. Images were restored only for this packet using the documented VG_100K/VG_100K_2 URL convention; the image hashes and successful URLs are recorded. This does not establish the missing validation-file byte identity.

Dataset labels are candidate-generation metadata only. Three independent human annotations and adjudication are required before block inclusion or model scoring.

Launch one independent local session per annotator:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python tools/e2_annotation_app.py \
  --packet runs/e2_development_20260912 --annotator annotator_1 --port 8765
```

Use separate ports for `annotator_2` and `annotator_3`. Each session writes only its own file under `annotations/`.
