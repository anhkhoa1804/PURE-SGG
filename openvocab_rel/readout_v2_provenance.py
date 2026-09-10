"""Paper C -- Readout v2 treatment-fidelity provenance (strategy S1).

Registered in docs/PAPER_C_R2_TREATMENT_FIDELITY_AMENDMENT_2026-09-10.md
(sections F.2, F.3, G). Nothing here is reachable unless
``cfg.readout_v2_prototype_source == "checkpoint"``; the default
``"reinit_E"`` path is byte-unmodified from the pre-amendment behaviour.

The defect this module corrects
-------------------------------
``train.py``'s resume filter keeps only keys already present in
``mdl.state_dict()``.  ``predicate_prototypes`` is created lazily by
``init_readout_v2()``, which runs *after* the resume block, so a trained
prototype tensor in the checkpoint is necessarily absent from the model at
resume time and is dropped.  Under S1 that skip is **expected and correct**
-- see ``EXPECTED_SKIPPED_KEYS``.  The defect was never the skip; it was the
absence of any subsequent restoration.  This module supplies exactly that
restoration, plus the provenance record that makes it auditable.

Treatment fidelity is decided from parameter provenance ALONE.  No endpoint
value, and no relationship between the endpoint and the baseline, is
consulted here -- see the amendment's section F.3.3.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

# The trained pilot checkpoint this amendment registers as R2c's treatment
# parameter (amendment section F.1).  Recomputed and cross-checked against
# tools/run_final_batch_gd_cuda.py's own recorded ``checkpoint_P_hash``.
REGISTERED_CHECKPOINT_P_SHA256 = (
    "294d01cccb7bc78233d52de2c5b532e261b1782f92a77cf72f9b768954943e5c"
)
PROTOTYPE_KEY = "predicate_prototypes"
EXPECTED_P_SHAPE: Tuple[int, int] = (51, 768)
EXPECTED_P_DTYPE = torch.float32
# Amendment section F.3.4: under S1 this key is EXPECTED to be skipped by the
# resume filter.  No other model key may be.
EXPECTED_SKIPPED_KEYS: Tuple[str, ...] = (PROTOTYPE_KEY,)

PROTOTYPE_SOURCE_REINIT_E = "reinit_E"
PROTOTYPE_SOURCE_CHECKPOINT = "checkpoint"
PROTOTYPE_SOURCES = (PROTOTYPE_SOURCE_REINIT_E, PROTOTYPE_SOURCE_CHECKPOINT)
# Value written into the provenance record when S1 restoration actually ran.
PROTOTYPE_SOURCE_LABEL_CHECKPOINT = "explicit_checkpoint_P"
PROTOTYPE_SOURCE_LABEL_REINIT_E = "registered_from_E_after_resume"


class ReadoutV2ProvenanceError(RuntimeError):
    """Fail-closed provenance violation.

    Raised, never warned.  Every call site treats this as fatal: a run that
    cannot certify which tensor it scored with is not a run.
    """


def tensor_hash(tensor: torch.Tensor) -> Dict[str, Any]:
    """Digest a tensor exactly the way tools/run_final_batch_gd_cuda.py does.

    Identical scheme (dtype tag + shape tag + raw bytes) so hashes recorded
    by the G-D runner, by this module, and by any future audit are directly
    comparable rather than merely similar.
    """
    cpu = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode())
    digest.update(json.dumps(list(cpu.shape)).encode())
    digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"sha256": digest.hexdigest(), "dtype": str(cpu.dtype), "shape": list(cpu.shape)}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_skipped_keys(skipped: Iterable[str]) -> Dict[str, Any]:
    """Classify the resume filter's skipped keys against S1's expectation.

    Returns the full observed list, the registered expectation, and the
    unexpected remainder.  The expected ``predicate_prototypes`` skip is
    surfaced as expected -- never filtered out, never downgraded to a count
    (amendment section F.3.4 / section G).
    """
    observed = sorted(str(k) for k in skipped)
    expected = [k for k in observed if k in EXPECTED_SKIPPED_KEYS]
    unexpected = [k for k in observed if k not in EXPECTED_SKIPPED_KEYS]
    return {
        "skipped_keys_by_name": observed,
        "skipped_keys_expected": list(EXPECTED_SKIPPED_KEYS),
        "skipped_keys_expected_observed": expected,
        "skipped_keys_unexpected": unexpected,
        "prototype_skip_was_expected": PROTOTYPE_KEY in observed,
    }


def assert_no_unexpected_skips(audit: Mapping[str, Any]) -> None:
    """Amendment section F.3.2 condition 7 -- fail closed on any other key."""
    unexpected = list(audit.get("skipped_keys_unexpected") or [])
    if unexpected:
        raise ReadoutV2ProvenanceError(
            "resume skipped model tensors beyond the expected Readout v2 prototype: "
            f"{unexpected}. Under strategy S1 exactly {list(EXPECTED_SKIPPED_KEYS)} may be "
            "skipped (it is restored and hash-verified afterwards); anything else means "
            "the backbone did not fully load."
        )


def extract_checkpoint_prototypes(
    model_state: Optional[Mapping[str, torch.Tensor]],
    *,
    expected_sha256: Optional[str] = REGISTERED_CHECKPOINT_P_SHA256,
    expected_shape: Optional[Sequence[int]] = EXPECTED_P_SHAPE,
    expected_dtype: Optional[torch.dtype] = EXPECTED_P_DTYPE,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Read and certify the checkpoint's trained prototype tensor.

    Amendment section F.3.1 assertions 1-2 and section F.3.2 condition 6.
    ``expected_sha256=None`` records the hash without asserting it, for
    synthetic fixtures; R2c always passes the registered value.
    """
    if model_state is None:
        raise ReadoutV2ProvenanceError(
            "--readout_v2_prototype_source checkpoint requires a resumed checkpoint, "
            "but no model state was loaded (is --resume true set?)."
        )
    if PROTOTYPE_KEY not in model_state:
        raise ReadoutV2ProvenanceError(
            f"checkpoint has no '{PROTOTYPE_KEY}' tensor; it does not carry a trained "
            "Readout v2 prototype matrix and cannot instantiate the R2c treatment."
        )
    checkpoint_p = model_state[PROTOTYPE_KEY].detach().clone()
    if expected_shape is not None and tuple(checkpoint_p.shape) != tuple(expected_shape):
        raise ReadoutV2ProvenanceError(
            f"checkpoint '{PROTOTYPE_KEY}' shape {tuple(checkpoint_p.shape)} != "
            f"registered {tuple(expected_shape)}."
        )
    if expected_dtype is not None and checkpoint_p.dtype != expected_dtype:
        raise ReadoutV2ProvenanceError(
            f"checkpoint '{PROTOTYPE_KEY}' dtype {checkpoint_p.dtype} != "
            f"registered {expected_dtype}."
        )
    if not bool(torch.isfinite(checkpoint_p).all()):
        raise ReadoutV2ProvenanceError(
            f"checkpoint '{PROTOTYPE_KEY}' contains non-finite values."
        )
    info = tensor_hash(checkpoint_p)
    if expected_sha256 and info["sha256"] != expected_sha256:
        raise ReadoutV2ProvenanceError(
            f"checkpoint '{PROTOTYPE_KEY}' sha256 {info['sha256']} != registered "
            f"{expected_sha256}. Refusing to run: the treatment parameter is not the "
            "one this amendment registers."
        )
    return checkpoint_p, info


def p_vs_e_summary(P: torch.Tensor, E: torch.Tensor) -> Dict[str, Any]:
    """Displacement of the installed prototypes from their CLIP init.

    Reports the same quantities the preregistration's section 13 mechanistic
    endpoint and section 15 stop conditions are written against, so the
    record is directly comparable to what Gate 1 already certified.
    """
    p = P.detach().float().cpu()
    e = E.detach().float().cpu()
    delta = p - e
    row_norm = delta.norm(dim=-1)
    cos = (
        torch.nn.functional.normalize(p, dim=-1)
        * torch.nn.functional.normalize(e, dim=-1)
    ).sum(-1)
    one_minus_cos = 1.0 - cos
    return {
        "equal": bool(torch.equal(p, e)),
        "max_abs_diff": float(delta.abs().max()),
        "mean_abs_diff": float(delta.abs().mean()),
        "row_l2_mean": float(row_norm.mean()),
        "row_l2_max": float(row_norm.max()),
        "one_minus_cos_mean": float(one_minus_cos.mean()),
        "one_minus_cos_min": float(one_minus_cos.min()),
        "one_minus_cos_max": float(one_minus_cos.max()),
        "rows_moved_gt_1e-6": int((one_minus_cos > 1e-6).sum()),
        "n_rows": int(p.shape[0]),
    }


def restore_checkpoint_prototypes(
    module: torch.nn.Module,
    checkpoint_p: torch.Tensor,
    *,
    checkpoint_hash: Mapping[str, Any],
    anchor_E: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Install the trained prototypes into an already-initialized parameter.

    ``init_readout_v2(E)`` must have run first, so ``_readout_v2_anchor``
    still holds E and the optimizer param group added in train.py already
    references this exact Parameter object.  The copy is therefore done in
    place under ``no_grad`` -- rebinding the attribute or assigning to
    ``.data`` would silently detach the optimizer's reference and, for the
    anchor, redefine the registered ``L_anchor``.

    Returns the section G provenance fields.  Raises on any mismatch.
    """
    if not hasattr(module, PROTOTYPE_KEY):
        raise ReadoutV2ProvenanceError(
            "restore_checkpoint_prototypes() requires init_readout_v2() to have run first."
        )
    param = getattr(module, PROTOTYPE_KEY)
    if tuple(param.shape) != tuple(checkpoint_p.shape):
        raise ReadoutV2ProvenanceError(
            f"installed prototype shape {tuple(param.shape)} != checkpoint "
            f"{tuple(checkpoint_p.shape)}."
        )
    with torch.no_grad():
        param.copy_(checkpoint_p.to(device=param.device, dtype=param.dtype))

    installed = tensor_hash(param)
    if installed["sha256"] != checkpoint_hash["sha256"]:
        raise ReadoutV2ProvenanceError(
            "post-restore provenance assertion FAILED: installed prototype sha256 "
            f"{installed['sha256']} != checkpoint {checkpoint_hash['sha256']}."
        )

    record: Dict[str, Any] = {
        "prototype_source": PROTOTYPE_SOURCE_LABEL_CHECKPOINT,
        "checkpoint_P_hash": dict(checkpoint_hash),
        "installed_P_hash": installed,
        "installed_P_matches_checkpoint": True,
    }

    anchor = getattr(module, "_readout_v2_anchor", None)
    if anchor is not None:
        record["anchor_hash"] = tensor_hash(anchor)
        if anchor_E is not None:
            record["anchor_equals_E"] = bool(
                torch.equal(anchor.detach().cpu(), anchor_E.detach().cpu())
            )
    if anchor_E is not None:
        equals_e = bool(torch.equal(param.detach().cpu().float(), anchor_E.detach().cpu().float()))
        record["installed_P_equals_E"] = equals_e
        record["E_hash"] = tensor_hash(anchor_E)
        record["P_vs_E"] = p_vs_e_summary(param, anchor_E)
        if equals_e:
            # Amendment section F.3.1 assertion 4.  A checkpoint whose P never
            # moved would independently be a preregistration section 15 failure.
            raise ReadoutV2ProvenanceError(
                "installed prototype is exactly equal to E. The checkpoint's trained "
                "prototypes did not move from their CLIP initialization, so this run "
                "would not instantiate the R2c treatment."
            )
    return record


def write_provenance_record(path: str, record: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)
