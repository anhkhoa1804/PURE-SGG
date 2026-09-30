# Geometry leakage audit

Status: **PASS for the pre-scoring code path**.

M3 now follows the explicit graph `datasets_vg150_clean/train.jsonl →
geometry standardization/class means → frozen scores on accepted blocks`.
The accepted block IDs, human gold, M1, M2, and M6 outputs are not used in
fitting. Evaluation IDs are rejected if they appear in the declared training
source. Geometry-conflict selection remains downstream of human gold and the
frozen M3 model and is secondary only.
