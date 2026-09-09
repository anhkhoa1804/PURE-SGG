"""Pure planning, error-metric, and decision logic for the G-D audit.

This module has no model or CUDA side effects.  It is intentionally separate
from the GPU runner so its tests can execute on CPU and cannot accidentally
launch an experiment.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch

GD_STATUSES = (
    "GD-EXPLAINED-INERT",
    "GD-SCIENTIFICALLY-MATERIAL",
    "GD-UNRESOLVED",
)


def planned_gpu_matrix() -> List[Dict[str, Any]]:
    """Return the minimum bounded matrix required by the forensic question.

    The four rows corresponding to the original proposal are retained.  The
    repeats test same-condition determinism; the padded rows test batch-shape
    dependence; and the explicit checkpoint-P row audits a load-order
    ambiguity in the current evaluator construction.
    """
    return [
        {
            "condition": "X_R0_noP_9",
            "checkpoint": "C1_seed1234.pt",
            "evaluator": "R0/original",
            "prototype_registration": "none",
            "prototype_adaptation": "bypassed",
            "batch": "final_9",
            "repeat_of": None,
            "resolves": ["baseline", "A", "B"],
        },
        {
            "condition": "X_R0_noP_9_repeat",
            "checkpoint": "C1_seed1234.pt",
            "evaluator": "R0/original",
            "prototype_registration": "none",
            "prototype_adaptation": "bypassed",
            "batch": "final_9",
            "repeat_of": "X_R0_noP_9",
            "resolves": ["F"],
        },
        {
            "condition": "Y_R0_P_registered_9",
            "checkpoint": "C1_seed1234.pt",
            "evaluator": "R0/original_plus_registration_only",
            "prototype_registration": "registered_never_before_rel_feat",
            "prototype_adaptation": "bypassed",
            "batch": "final_9",
            "repeat_of": None,
            "resolves": ["G"],
        },
        {
            "condition": "Q_R2_noP_9",
            "checkpoint": "readout_v2_pilot_seed1234_v3.pt",
            "evaluator": "R0/original",
            "prototype_registration": "none",
            "prototype_adaptation": "bypassed",
            "batch": "final_9",
            "repeat_of": None,
            "resolves": ["A", "B", "G"],
        },
        {
            "condition": "Z_R2_P_registered_9",
            "checkpoint": "readout_v2_pilot_seed1234_v3.pt",
            "evaluator": "R2/construction_actual_load_order",
            "prototype_registration": "registered_never_before_rel_feat",
            "prototype_adaptation": "bypassed",
            "batch": "final_9",
            "repeat_of": None,
            "resolves": ["A", "B", "G"],
        },
        {
            "condition": "Z_R2_P_registered_9_repeat",
            "checkpoint": "readout_v2_pilot_seed1234_v3.pt",
            "evaluator": "R2/construction_actual_load_order",
            "prototype_registration": "registered_never_before_rel_feat",
            "prototype_adaptation": "bypassed",
            "batch": "final_9",
            "repeat_of": "Z_R2_P_registered_9",
            "resolves": ["F"],
        },
        {
            "condition": "X_R0_noP_12padded",
            "checkpoint": "C1_seed1234.pt",
            "evaluator": "R0/original",
            "prototype_registration": "none",
            "prototype_adaptation": "bypassed",
            "batch": "final_9_padded_to_12",
            "repeat_of": None,
            "resolves": ["C"],
        },
        {
            "condition": "Z_R2_P_registered_12padded",
            "checkpoint": "readout_v2_pilot_seed1234_v3.pt",
            "evaluator": "R2/construction_actual_load_order",
            "prototype_registration": "registered_never_before_rel_feat",
            "prototype_adaptation": "bypassed",
            "batch": "final_9_padded_to_12",
            "repeat_of": None,
            "resolves": ["C", "G"],
        },
        {
            "condition": "Zp_R2_checkpointP_registered_9",
            "checkpoint": "readout_v2_pilot_seed1234_v3.pt",
            "evaluator": "R2/construction_explicit_checkpoint_P",
            "prototype_registration": "registered_never_before_rel_feat",
            "prototype_adaptation": "bypassed",
            "batch": "final_9",
            "repeat_of": None,
            "resolves": ["A", "G", "checkpoint_P_load_order"],
        },
    ]


def tensor_error_metrics(actual: torch.Tensor, reference: torch.Tensor,
                         eps: float = 1.0e-12) -> Dict[str, Any]:
    """Compute exact and relative error summaries for aligned tensors."""
    a = actual.detach().double().cpu()
    b = reference.detach().double().cpu()
    if tuple(a.shape) != tuple(b.shape):
        return {"shape_mismatch": True, "shape_actual": list(a.shape),
                "shape_reference": list(b.shape)}
    d = a - b
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {"shape_mismatch": False, "finite": False,
                "nonfinite_actual": int((~torch.isfinite(a)).sum()),
                "nonfinite_reference": int((~torch.isfinite(b)).sum())}
    ad = d.abs()
    rel = ad / b.abs().clamp_min(float(eps))
    differing = d != 0
    return {
        "shape_mismatch": False,
        "finite": True,
        "shape": list(a.shape),
        "element_count": int(d.numel()),
        "changed_element_count": int(differing.sum().item()),
        "changed_fraction": float(differing.float().mean().item()) if d.numel() else 0.0,
        "changed_row_count": int(differing.reshape(a.shape[0], -1).any(1).sum()) if a.ndim > 1 and a.numel() else int(differing.sum()),
        "absolute_quantiles": {str(q): float(torch.quantile(ad.flatten(), q)) for q in (0.0, 0.5, 0.9, 0.99, 1.0)} if d.numel() else {},
        "relative_denominator_floor": eps,
        "max_abs": float(ad.max().item()) if d.numel() else 0.0,
        "mean_abs": float(ad.mean().item()) if d.numel() else 0.0,
        "rms_abs": float(torch.sqrt((d * d).mean()).item()) if d.numel() else 0.0,
        "max_relative": float(rel.max().item()) if d.numel() else 0.0,
        "mean_relative": float(rel.mean().item()) if d.numel() else 0.0,
    }


def classify_gd(*, reproduction_complete: bool,
                causal_explanation_proven: bool,
                endpoint_operationally_irrelevant: bool,
                endpoint_materially_changed: bool) -> str:
    """Apply the locked three-way decision rule; never invent a fourth state."""
    if not reproduction_complete:
        return "GD-UNRESOLVED"
    if endpoint_materially_changed:
        return "GD-SCIENTIFICALLY-MATERIAL"
    if causal_explanation_proven and endpoint_operationally_irrelevant:
        return "GD-EXPLAINED-INERT"
    return "GD-UNRESOLVED"


def required_condition_fields() -> List[str]:
    return [
        "device", "dtype", "autocast_dtype", "TF32",
        "torch_deterministic_algorithms", "cudnn_deterministic",
        "cudnn_benchmark", "batch_shape", "image_ids",
        "pixel_tensor_sha256", "per_image_pixel_sha256",
        "module_train_eval_flags", "relevant_module_names",
        "parameter_hashes", "buffer_hashes", "prototype_parameter_hash",
        "prototype_read_events", "rng_state_before", "rng_state_after",
        "rel_feat_sha256", "rel_feat_vs_baseline", "text_logits",
        "model_logits", "cls_logits", "same_condition_repeatability",
        "final_batch_wprd",
    ]
