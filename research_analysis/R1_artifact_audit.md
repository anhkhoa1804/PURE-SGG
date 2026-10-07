# R1 Artifact Audit

## Status

R1 is `BLOCKED_TECHNICAL`. Training began, completed epoch index 0, and wrote an intermediate checkpoint. The run was interrupted during the launcher's subsequent in-training validation diagnostic when a separate GPU workload appeared. The full evaluator was never launched. No corrected-C1 WPRD, delta, confidence interval, or materiality conclusion exists.

The recorded preflight shows the L4 idle at 0 MiB before training; the contention record at 2026-10-07 02:57:19 UTC shows 100% utilization and a separate process, PID 32655, running `experiments/005_omnidocbench/prepare.py`. The record describes separate command trees from R1. It does not establish the process owner's identity, so no ownership claim is made. R1 PID 17600 alone received SIGINT and exited 130. PID 24793 was absent at preflight and was not signalled.

## Stage reached

- The sole registered C1a training invocation started at 2026-10-07 02:18:22 UTC, with seed 1234 and the frozen protocol.
- Epoch index 0 completed its training loop. The run used 12,000 samples per epoch, batch size 6, and gradient accumulation 4: 2,000 microbatches and 500 scheduled optimizer-update boundaries for the completed epoch. The last periodic progress line is `ep=0 st=1975`; progress was logged every 25 microbatches.
- The epoch-0 checkpoint was saved before the in-training 60-batch validation diagnostic. The trace ends in object-label candidate inference with `KeyboardInterrupt`.
- No subsequent training epoch completed. The configured three-epoch run did not complete.
- The separate full validation evaluator in `tools/c0_c1_evaluate.py` was not launched; the `evaluation/` output and result are absent.
- The training log contains no WPRD endpoint. Any intermediate training loss/progress values are optimization diagnostics only and are not an R1 result.

## Artifact classifications

The run directory is preserved without deleting or rewriting its files. Hashes below are the recorded SHA256 values in the run's `hashes.json`; the canonical checkpoint and dataset hashes were independently rechecked for this audit.

| Artifact | Classification | Audit disposition |
|---|---|---|
| `README.md` | `DIAGNOSTIC_ONLY` | Describes the incomplete run and explicitly warns that no final metric exists. |
| `checkpoint.pt` | `PARTIAL_NONFINAL` | 5,293,175,545-byte epoch-0 checkpoint; SHA256 `1bebd6104688c4c50484fe0d1a507caed2e23c23f5592628490ff697fd0cda82`. Not a completed three-epoch model; not evaluated; never cite as corrected C1. |
| `commands.txt` | `REPRODUCIBILITY_METADATA` | Records the one training invocation and the evaluator command that was not run. |
| `compute_apps_pretrain.txt` | `DIAGNOSTIC_ONLY` | Pre-launch compute-app query recorded zero active compute processes. |
| `contention_process.txt` | `DIAGNOSTIC_ONLY` | Contemporaneous record of the later GPU contention, commands, reported allocation, and R1-only interruption. Owner identity is not recorded. |
| `disk_pretrain.txt` | `DIAGNOSTIC_ONLY` | Pre-launch disk capacity snapshot. |
| `environment.json` | `REPRODUCIBILITY_METADATA` | Environment and runtime provenance. |
| `hashes.json` | `REPRODUCIBILITY_METADATA` | Hash manifest; explicitly does not self-hash. |
| `nvidia_smi_pretrain.txt` | `DIAGNOSTIC_ONLY` | Immediate pre-training L4 snapshot: 0 MiB reported, 4% utilization, no compute process. |
| `pid_24793_pretrain.txt` | `DIAGNOSTIC_ONLY` | Preflight process query; PID 24793 absent. |
| `resource_usage.json` | `REPRODUCIBILITY_METADATA` | Preflight and interruption resource timeline; reports about 40 minutes observed runtime and no retry. |
| `stop_record.json` | `DIAGNOSTIC_ONLY` | Terminal stop status `BLOCKED_TECHNICAL`; records exit 130 and no evaluation/retry. |
| `training.log` | `PARTIAL_NONFINAL` | Partial training and diagnostic trace ending in `KeyboardInterrupt`; not a result report. |
| `training/provenance.txt` | `REPRODUCIBILITY_METADATA` | Seed, arm, geometry flags, commit, initialization hash, and start time. |
| `training_config.json` | `REPRODUCIBILITY_METADATA` | Frozen treatment, comparator, train recipe, evaluation plan, and decision margin. |

No artifact in this run qualifies as `VALID_FINAL_EVIDENCE`. There are no recorded `TRANSIENT` artifacts requiring removal; the large checkpoint is retained as `PARTIAL_NONFINAL` for provenance.

## Frozen baseline integrity

- `checkpoints/C1_seed1234.pt`: SHA256 `79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854` — verified unchanged.
- `datasets_vg150_clean/train.jsonl`: SHA256 `306fc0db96ca716bbc78c2cf32d49fdc695ab0074004a99c7399c850a1c605a4` — verified unchanged.

## Decision boundary

R1's predefined +0.010 WPRD materiality threshold cannot be evaluated. The result is unavailable, not zero, not below threshold, and not evidence for or against a geometry effect. No R1 rerun is authorized by this closure pass.
