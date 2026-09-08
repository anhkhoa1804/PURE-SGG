#!/usr/bin/env bash
# Paper C: Readout v2 training (P = trainable predicate prototypes).
#
# Pre-registered in docs/PAPER_C_READOUT_V2_PREREGISTRATION.md.
#
# Geometry is FIXED to the validated C1 contract and is never varied by this
# script (no ARM switch, unlike run_c0_c1.sh) -- geometry is a controlled
# variable here, not the subject of this experiment. Every model parameter
# except the new predicate_prototypes tensor is frozen by train.py's own
# readout_v2_enabled code path (openvocab_rel/train.py, right after the
# resume block) -- this script does not need to (and must not) pass any
# flag that changes the backbone.
#
# Usage:
#   PILOT=1 bash scripts/train/run_readout_v2.sh          # ~250 images, 1 epoch
#   bash scripts/train/run_readout_v2.sh                  # full budget -- NOT
#     registered yet; do not run until a post-pilot amendment fixes the
#     budget (docs/PAPER_C_READOUT_V2_PREREGISTRATION.md section 12/21).
set -Eeuo pipefail

SEED="${SEED:-1234}"
BASE_CKPT="${BASE_CKPT:-checkpoints/C1_seed1234.pt}"
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-0.5}"
LR="${LR:-2e-3}"

if [[ "${PILOT:-0}" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-250}"   # ~3,000 GT-positive pairs at this dataset's ~12.7 pairs/image density
  RUN_NAME="${RUN_NAME:-readout_v2_pilot_seed${SEED}}"
else
  echo "Full-budget run is NOT registered yet -- see" >&2
  echo "docs/PAPER_C_READOUT_V2_PREREGISTRATION.md section 12/21." >&2
  echo "Set PILOT=1 to run the registered pilot, or register a full-budget" >&2
  echo "amendment first." >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"
SAVE_PATH="${SAVE_PATH:-checkpoints/${RUN_NAME}.pt}"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "BASE_CKPT not found: ${BASE_CKPT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON="${PYTHON:-.venv/bin/python3}"
mkdir -p checkpoints runs logs "${OUT_DIR}"

{
  echo "experiment=readout_v2"
  echo "seed=${SEED}"
  echo "epochs=${EPOCHS}"
  echo "samples_per_epoch=${SAMPLES_PER_EPOCH}"
  echo "readout_v2_lambda_anchor=${LAMBDA_ANCHOR}"
  echo "readout_v2_lr=${LR}"
  echo "geom_input_pixel_space=true (fixed, C1 contract)"
  echo "geom_fourier_scale=0.01 (fixed, C1 contract)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain | wc -l)"
  echo "resume_from=${BASE_CKPT}"
  echo "ckpt_sha256=$(sha256sum "${BASE_CKPT}" | cut -d' ' -f1)"
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
  --epochs "${EPOCHS}"
  --batch_size 12
  --accum_steps 1
  --num_workers 4
  # --lr is overridden internally by readout_v2_lr once readout_v2_enabled
  # is applied (train.py reassigns base_lr right after the resume block --
  # every other parameter is frozen, so its nominal LR is inert regardless).
  --lr "${LR}"
  --lr_schedule cosine
  --warmup_steps 0
  --amp true
  --amp_dtype bf16
  --channels_last true
  --gradient_checkpointing false
  --freeze_clip true
  --progressive_unfreeze false
  # ---- objective: PRE-GPU AUDIT CORRECTION (see
  #      docs/PAPER_C_READOUT_V2_PREREGISTRATION.md CORRECTION block).
  #      An earlier version of this script copied these three flags from
  #      run_c0_c1.sh's TRAINING recipe. That was wrong: explicit_spoa_enabled
  #      and text_conditioned_projection_enabled are not loss-side knobs --
  #      they change what tools/readout_v2_evaluate.py's forward pass computes
  #      (explicit_spoa_enabled changes rel_feat itself; text_conditioned_
  #      projection_enabled changes what adaptive_predicate_logits scores),
  #      and the eval tool hardcodes both FALSE (matching every existing R0/C1
  #      WPRD number in this programme, which was also always evaluated with
  #      both false -- c0_c1_evaluate.py hardcodes them false too, so "true"
  #      here never matched even the base checkpoint's OWN evaluated rel_feat,
  #      let alone R2's). predicate_label_relaxation_enabled=true is a THIRD,
  #      unregistered active term on l_readout_v2_ce (it reaches
  #      _predicate_ce_loss via the pred_sim_matrix argument). All three are
  #      corrected to false here so train-time rel_feat/readout-input/loss
  #      are proven identical to what tools/readout_v2_evaluate.py computes --
  #      see tests/test_readout_v2.py's runtime-resolved parity tests.
  --train_objective full
  --explicit_spoa_enabled false
  --adaptive_calibration_enabled true
  --adaptive_prior_enabled true
  --bias_residual_enabled true
  --predicate_label_relaxation_enabled false
  --text_conditioned_projection_enabled false
  --text_conditioned_projection_residual 0.35
  --relationness_enabled true
  # Every lambda below is left at the base run's own value for forward-pass
  # parity, but the base run's own frozen parameters cannot move, so only
  # lambda_predicate_ce=0 (see below) actually matters for what trains.
  --lambda_predicate_ce 0.0
  --lambda_spoa_alignment 0.0
  --lambda_dense_grounding 0.0
  --lambda_counterfactual 0.0
  --lambda_text_predicate_ce 0.0
  --lambda_relationness 0.0
  --lambda_calibration_reg 0.0
  --lambda_calibration_kl 0.0
  --lambda_calibration_rank 0.0
  --calibration_rank_margin 0.25
  --predicate_ce_loss focal
  --predicate_ce_gamma 1.0
  --predicate_ce_weight_power 0.5
  --predicate_ce_max_weight 5.0
  --predicate_metadata_path configs/predicate_metadata_vg150.json
  # ---- Readout v2 -----------------------------------------------------
  --readout_v2_enabled true
  --readout_v2_lambda_anchor "${LAMBDA_ANCHOR}"
  --readout_v2_lr "${LR}"
  # ---- resume from the validated, FIXED C1 geometry checkpoint --------
  --resume true
  --resume_from "${BASE_CKPT}"
  --reset_epoch true
  # ---- in-training eval: cheap, stability only -------------------------
  --eval_every 1
  --eval_batches 20
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
  --log_every 10
  --save_best_checkpoints false
  # ================= GEOMETRY: FIXED, NEVER VARIED ========================
  --geom_input_pixel_space true
  --geom_fourier_scale 0.01
  # =========================================================================
)

if [[ "$#" -gt 0 ]]; then ARGS+=("$@"); fi

PYTHONUNBUFFERED=1 "${PYTHON}" "${ARGS[@]}" 2>&1 | tee "logs/${RUN_NAME}.log"
echo "finished_utc=$(date -u +%FT%TZ)" >> "${OUT_DIR}/provenance.txt"
