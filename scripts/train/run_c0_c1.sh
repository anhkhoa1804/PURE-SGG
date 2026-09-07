#!/usr/bin/env bash
# Paper C: the C0 / C1 paired intervention.
#
# Pre-registered in docs/PAPER_C_C0_C1_PREREGISTRATION.md.
#
# Both arms are THIS SCRIPT with one argument changed. Every other setting --
# data, split, seed, optimizer, LR, schedule, batch, accumulation, budget,
# losses, sampler, evaluator -- is pinned identically below, so the only thing
# that can differ between C0 and C1 is the geometry contract.
#
#   ARM=C0   geom_input_pixel_space=false  geom_fourier_scale=1.0   (historical)
#   ARM=C1   geom_input_pixel_space=true   geom_fourier_scale=0.01
#   ARM=C1a  geom_input_pixel_space=true   geom_fourier_scale=1.0   (diagnostic)
#   ARM=C1b  geom_input_pixel_space=false  geom_fourier_scale=0.01  (diagnostic)
#
# Usage: ARM=C0 SEED=1234 bash scripts/train/run_c0_c1.sh
set -Eeuo pipefail

ARM="${ARM:?set ARM to C0, C1, C1a or C1b}"
SEED="${SEED:-1234}"
EPOCHS="${EPOCHS:-3}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-12000}"
FOURIER_SCALE="${FOURIER_SCALE:-0.01}"
RUN_NAME="${RUN_NAME:-${ARM}_seed${SEED}}"
OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"
SAVE_PATH="${SAVE_PATH:-checkpoints/${RUN_NAME}.pt}"

case "${ARM}" in
  C0)  PIXEL=false; SCALE=1.0 ;;
  C1)  PIXEL=true;  SCALE="${FOURIER_SCALE}" ;;
  C1a) PIXEL=true;  SCALE=1.0 ;;
  C1b) PIXEL=false; SCALE="${FOURIER_SCALE}" ;;
  *) echo "unknown ARM=${ARM}" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON="${PYTHON:-.venv/bin/python3}"
mkdir -p checkpoints runs logs "${OUT_DIR}"

# Provenance, recorded before the run rather than reconstructed after it.
{
  echo "arm=${ARM}"
  echo "seed=${SEED}"
  echo "epochs=${EPOCHS}"
  echo "samples_per_epoch=${SAMPLES_PER_EPOCH}"
  echo "geom_input_pixel_space=${PIXEL}"
  echo "geom_fourier_scale=${SCALE}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l)"
  echo "resume_from=checkpoints/demo_best/pure_best_adapt_light_mR50.pt"
  echo "ckpt_sha256=$(sha256sum checkpoints/demo_best/pure_best_adapt_light_mR50.pt | cut -d' ' -f1)"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
} > "${OUT_DIR}/provenance.txt"

ARGS=(
  -u -m openvocab_rel.train
  --stage 3
  --gpu_preset l4_24gb
  --run_name "${RUN_NAME}"
  --out_dir "${OUT_DIR}"
  --save_path "${SAVE_PATH}"
  --save_metrics_json "${OUT_DIR}/metrics.jsonl"
  --seed "${SEED}"
  # ---- data: identical for both arms -------------------------------------
  --vg150_root datasets_vg150_clean
  --vg150_enabled true
  --vg150_source local-jsonl
  --device cuda
  --clip_input_res 336
  --max_images 0
  --samples_per_epoch "${SAMPLES_PER_EPOCH}"
  --max_objects 32
  --max_pairs 64
  --learned_prune_k 64
  --use_all_pairs false
  --negative_pair_ratio 2.0
  # ---- optimisation: identical for both arms -----------------------------
  --epochs "${EPOCHS}"
  --batch_size 12
  --accum_steps 2
  --num_workers 4
  --lr 2e-5
  --lr_schedule cosine
  --warmup_steps 300
  --amp true
  --amp_dtype bf16
  --channels_last true
  --gradient_checkpointing true
  --freeze_clip false
  --progressive_unfreeze false
  # ---- objective: the historical stage-3 recipe, identical for both arms --
  --train_objective full
  --explicit_spoa_enabled true
  --adaptive_calibration_enabled true
  --adaptive_prior_enabled true
  --bias_residual_enabled true
  --predicate_label_relaxation_enabled true
  --text_conditioned_projection_enabled true
  --text_conditioned_projection_residual 0.35
  --relationness_enabled true
  --lambda_predicate_ce 1.2
  --lambda_spoa_alignment 0.75
  --lambda_dense_grounding 0.25
  --lambda_counterfactual 0.05
  --lambda_text_predicate_ce 0.4
  --lambda_relationness 0.15
  --lambda_calibration_reg 0.001
  --lambda_calibration_kl 0.03
  --lambda_calibration_rank 0.08
  --calibration_rank_margin 0.25
  --predicate_ce_loss focal
  --predicate_ce_gamma 1.0
  --predicate_ce_weight_power 0.5
  --predicate_ce_max_weight 5.0
  --predicate_metadata_path configs/predicate_metadata_vg150.json
  # ---- resume from the frozen historical checkpoint ----------------------
  --resume true
  --resume_from checkpoints/demo_best/pure_best_adapt_light_mR50.pt
  --reset_epoch true
  # ---- in-training eval: cheap, for stability only. The scientific
  #      evaluation is a separate full-split pass, see the registration. ----
  --eval_every 1
  --eval_batches 60
  --eval_fast_mode false
  --eval_sgg_predicate_score_mode ensemble
  --eval_sgg_predicate_ensemble_alpha 0.0
  --eval_sgg_use_gt_pairs true
  --eval_sgg_use_relationness false
  --eval_sgg_grounding_dino_enabled false
  --freq_bias_enabled true
  --freq_bias_path datasets_vg150_clean/frequency_prior_train.json
  --freq_bias_alpha 3.75
  --freq_bias_smoothing 1.0
  --bayes_calibration_weight 0.0
  --log_every 25
  # Both arms are evaluated at a FIXED budget (the final epoch), never at a
  # per-arm selected best epoch -- that selection would differ between arms and
  # contaminate the paired comparison. The four best_* checkpoints are therefore
  # pure waste here (~5 GB each) and are suppressed identically for both arms.
  --save_best_checkpoints false
  # ================= THE ONLY TWO FLAGS THAT DIFFER ======================
  --geom_input_pixel_space "${PIXEL}"
  --geom_fourier_scale "${SCALE}"
  # =======================================================================
)

if [[ "$#" -gt 0 ]]; then ARGS+=("$@"); fi

PYTHONUNBUFFERED=1 "${PYTHON}" "${ARGS[@]}" 2>&1 | tee "logs/${RUN_NAME}.log"
echo "finished_utc=$(date -u +%FT%TZ)" >> "${OUT_DIR}/provenance.txt"
