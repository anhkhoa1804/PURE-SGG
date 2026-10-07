# R1 resource and process check

Timestamp: 2026-10-07 02:08:20 UTC (preflight; a second fresh check is required immediately before each GPU launch).

- Device: NVIDIA L4, 23,034 MiB; driver 580.178.04; CUDA 13.0.
- Preflight utilization: 4%; reported memory: 0 MiB used.
- `nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv`: no running compute process found.
- PID 24793: both `ps -fp 24793` and the explicit `ps -o ... -p 24793` returned no process row. It was not signalled or otherwise touched. Since the PID was absent, no ownership/provenance inference is made about any prior process.
- Disk at preflight: 24 GiB available on `/mnt/research-work`.
- Canonical C1 checkpoint SHA256 verified as `79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854`.
- Canonical train JSONL SHA256 verified as `306fc0db96ca716bbc78c2cf32d49fdc695ab0074004a99c7399c850a1c605a4`.
- Canonical source snapshot HEAD verified as `ec4cca6ff01ab46929b7b428220fe0df2ba00f2a`.

The initial check did not itself authorize a launch; the immediate pre-training and during-run resource events are recorded below.

## Immediate pre-training check and during-run stop

At 2026-10-07 02:18:08 UTC, a fresh `nvidia-smi` again showed 0/23,034 MiB, 4% utilization, no compute apps, and PID 24793 absent. Training launched at 02:18:22 UTC. At 02:57:19 UTC, while training PID 17600 was active, GPU utilization reached 100% and a separate PID 32655 appeared, running `experiments/005_omnidocbench/prepare.py` with 186 MiB allocated. The R1 process alone received SIGINT and exited with status 130; PID 32655 and PID 24793 were not signalled. No full evaluator was launched. Detailed stop evidence is in `runs/R1_fixed_geometry_C1a_20261007T021152Z/stop_record.json`.
