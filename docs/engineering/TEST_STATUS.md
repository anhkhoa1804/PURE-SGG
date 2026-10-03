# Test status

Date: 2026-10-03 UTC. Environment: Python 3.10.12; PyTorch 2.9.1+cu129.
All commands were run with `CUDA_VISIBLE_DEVICES=''`; no model training or GPU
work was launched.

| Command | Result | Notes |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q` before adding `pytest.ini` | FAIL during collection | Pytest descended into copied `runs/**/code/tests` snapshots, causing duplicate module-name import mismatches and a missing module in the snapshot. No test assertions ran to completion. |
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/ -q` | PASS — 611 passed, 25 warnings, 346.96 s | Explicit active-tree run. Workspace included three pre-existing untracked `tests/test_l4_*.py` files; they were not staged by this closeout. |
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q` after `pytest.ini` | PASS — 611 passed, 25 warnings, 315.62 s | Confirms the ordinary documented invocation uses the scoped test tree. |
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m compileall -q openvocab_rel tools scripts` | PASS | Syntax check only; no module execution/inference. |
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/prepare_vg150_subset.py --help` | PASS | Help path only. |
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/validate_dataset.py --help` | PASS | Help path only. |
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m openvocab_rel.train --help` | PASS | Help path only; no training launched. |
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python -c 'import openvocab_rel; import openvocab_rel.config; import openvocab_rel.datasets.vg150_loader; import openvocab_rel.evals; import openvocab_rel.models.relational_model; import openvocab_rel.train'` | PASS | Core package imports only. |

`pytest.ini` sets `testpaths=tests` and excludes run/data/checkpoint trees so
the documented bare pytest invocation does not treat embedded research source
snapshots as the live suite. Optional/historical dependencies and real-data
paths are outside the offline suite’s scope. Requirements declare PyTorch
2.10+, while the observed environment is 2.9.1+cu129; this mismatch is a
known limitation, not hidden by the local pass/fail result.
