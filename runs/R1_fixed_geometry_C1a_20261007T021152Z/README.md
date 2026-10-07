# R1 — fixed geometry units-contract run

One bounded seed-1234 C1a run, registered in `research_analysis/R1_protocol_freeze.md` before launch.

Treatment is C1a: pixel-scale box inputs with Fourier scale held at 1.0. Primary comparator is the frozen C0 arm (same Fourier scale, normalized boxes). Historical C1 (pixel-scale boxes, Fourier scale 0.01) is context only because it changes two geometry factors.

The one training invocation was interrupted during its epoch-0 in-training diagnostic when unrelated GPU PID 32655 appeared. A partial epoch-0 checkpoint and complete partial training log are preserved. The full evaluator and paired comparison were not launched; this directory contains no final metric. The checkpoint is provenance-only, not a completed model. The canonical prior runs and checkpoints were not output targets.
