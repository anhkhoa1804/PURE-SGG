#!/usr/bin/env python3
"""Bounded CUDA runner for the Paper C final-batch G-D forensic matrix.

This is intentionally separate from the historical R0/R2 evaluators.  It
launches only the final nine validation examples (or those nine plus three
duplicates for the batch-shape control), never a full validation pass.  It
must be run with a new versioned output path; it refuses to overwrite any
artifact.

The runner is not executed by the CPU audit.  Before model construction it
checks ``nvidia-smi`` and refuses to compete with any reported compute
process.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gd_forensic_protocol import (  # noqa: E402
    classify_gd,
    planned_gpu_matrix,
    required_condition_fields,
    tensor_error_metrics,
)

DATA_ROOT = ROOT / "datasets_vg150_clean"
FINAL_START = 10392
FINAL_COUNT = 9
BATCH_SIZE = 12
FINAL_IDS = ['2385790', '2353744', '2390749', '2343247', '2405812',
             '2390381', '2335326', '2406736', '2361448']
FINAL_PAIR_COUNTS = [1, 2, 7, 12, 9, 6, 10, 11, 7]
IMAGE_FIELDS = ('image_id', 'pairs', 'pair_index', 'model_logits', 'prior_rows',
                'text_logits', 'cls_logits', 'adaptive_logits', 'obj_labels',
                'subj_label', 'obj_label', 'obj_boxes', 'gt_subj_idx', 'gt_obj_idx',
                'gt_pred', 'gt_subj_label', 'gt_obj_label', 'rel_feat')


def _slots(layout):
    if layout not in ('final_9', 'final_9_padded_to_12'):
        raise ValueError(layout)
    return list(range(9)) + ([0, 1, 2] if layout == 'final_9_padded_to_12' else [])


def _assert_population(ids, counts, layout):
    slots = _slots(layout)
    if list(map(str, ids)) != [FINAL_IDS[i] for i in slots]:
        raise ValueError('Wrong final-nine image identity/order')
    if list(counts) != [FINAL_PAIR_COUNTS[i] for i in slots]:
        raise ValueError('Wrong final-nine pair counts; expected exactly 65 original rows')


def _trim_dump(payload, layout):
    _assert_population(payload['image_id'], [len(x) for x in payload['pairs']], layout)
    result = dict(payload)
    for key in IMAGE_FIELDS:
        if key in result:
            if len(result[key]) != len(_slots(layout)):
                raise ValueError(f'Unaligned image field: {key}')
            result[key] = result[key][:9]
    result.update(n_images=9, n_pairs=65)
    if len(result['pred_vocab']) != 51:
        raise ValueError('Expected full 51-column vocabulary')
    return result


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def _digest_bytes(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> Dict[str, Any]:
    x = tensor.detach().contiguous()
    cpu = x.cpu()
    return {
        "sha256": _digest_bytes([
            str(cpu.dtype).encode(), json.dumps(list(cpu.shape)).encode(),
            cpu.reshape(-1).view(torch.uint8).numpy().tobytes(),
        ]),
        "dtype": str(cpu.dtype), "shape": list(cpu.shape),
    }


def _object_hashes(module: torch.nn.Module, include_values: bool = True) -> Dict[str, Any]:
    """Hash parameters and buffers without changing their device or dtype."""
    params: Dict[str, Any] = {}
    buffers: Dict[str, Any] = {}
    for name, value in module.named_parameters():
        if include_values:
            params[name] = dict(_tensor_hash(value), requires_grad=value.requires_grad,
                                device=str(value.device))
        else:
            params[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    for name, value in module.named_buffers():
        if include_values:
            buffers[name] = _tensor_hash(value)
        else:
            buffers[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    plain = {}
    excluded = set(torch.nn.Module().__dict__) | {'_parameters', '_buffers', '_modules'}
    def encode(x):
        if isinstance(x, torch.Tensor):
            return _tensor_hash(x)
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, (list, tuple)):
            return [encode(v) for v in x]
        if isinstance(x, dict):
            return {str(k): encode(v) for k, v in x.items()}
        if hasattr(x, 'to_dict'):
            return encode(x.to_dict())
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            return encode(dataclasses.asdict(x))
        return {'type': type(x).__qualname__, 'uninspected': True}
    for name, child in module.named_modules():
        plain[name] = {k: encode(v) for k, v in vars(child).items()
                       if k not in excluded and not callable(v)}
    return {"parameters": params, "buffers": buffers, 'plain_module_state': plain}


def _module_modes(module: torch.nn.Module) -> Dict[str, bool]:
    return {name or "<root>": bool(child.training)
            for name, child in module.named_modules()}


def _rng_hash(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return _tensor_hash(value)["sha256"]
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _rng_state(cuda: bool) -> Dict[str, Any]:
    out = {
        "python_random_sha256": _rng_hash(random.getstate()),
        "torch_cpu_sha256": _rng_hash(torch.get_rng_state()),
    }
    if cuda:
        out["torch_cuda_sha256"] = [
            _rng_hash(x) for x in torch.cuda.get_rng_state_all()
        ]
    try:
        import numpy as np
        state = np.random.get_state()
        out['numpy_state'] = [state[0], state[1].tolist(), *state[2:]]
        out["numpy_sha256"] = hashlib.sha256(json.dumps(out['numpy_state']).encode()).hexdigest()
    except Exception:
        out["numpy_sha256"] = None
    out['python_state'] = random.getstate()
    out['torch_cpu_state_hex'] = torch.get_rng_state().numpy().tobytes().hex()
    if cuda:
        out['torch_cuda_state_hex'] = [x.cpu().numpy().tobytes().hex() for x in torch.cuda.get_rng_state_all()]
    return out


def _runtime_settings(device: torch.device, cfg: Any) -> Dict[str, Any]:
    from openvocab_rel.evals import _resolve_amp_dtype
    return {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "autocast_enabled": bool(getattr(cfg, "amp", False)),
        "autocast_dtype": str(getattr(cfg, "amp_dtype", "unknown")),
        'effective_autocast_dtype': str(_resolve_amp_dtype(cfg, device)),
        'effective_autocast_enabled': bool(getattr(cfg, 'amp', False)) and device.type == 'cuda',
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _final_dataset_patch(batch_layout: str):
    """Restrict only the validation JSONL dataset created by train.main."""
    from openvocab_rel.datasets import vg150_loader

    cls = vg150_loader.VG150JSONLDataset
    original = cls.__init__

    def patched(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if str(getattr(self, "split", "")).lower() in {"val", "valid", "validation"}:
            rows = list(self.rows[FINAL_START:FINAL_START + FINAL_COUNT])
            if [str(r['image_id']) for r in rows] != FINAL_IDS:
                raise ValueError('Validation source indices 10392..10400 no longer match historical IDs')
            if batch_layout == "final_9_padded_to_12":
                rows.extend(rows[:3])
            if len(rows) != (12 if batch_layout.endswith("12") else FINAL_COUNT):
                raise RuntimeError(f"final forensic slice has {len(rows)} rows for {batch_layout}")
            self.rows = rows
            self.cfg.samples_per_epoch = 0

    cls.__init__ = patched

    def restore() -> None:
        cls.__init__ = original

    return restore


class Instrument:
    def __init__(self, batch_layout: str):
        self.batch_layout = batch_layout
        self.events: List[Dict[str, Any]] = []
        self.first_forward = True
        self.before: Dict[str, Any] = {}
        self.after: Dict[str, Any] = {}
        self.input: Dict[str, Any] = {}
        self.rel_feat: Optional[torch.Tensor] = None
        self.rel_feat_per_image: List[torch.Tensor] = []
        self.runtime: Dict[str, Any] = {}
        self.forward_count = 0
        self._original_adaptive = None
        self._adaptive_owner = None

    def prepare(self, original, cfg, batch, processor, device):
        result = original(cfg, batch, processor, device)
        if self.first_forward:
            pixel, boxes, pairs = result
            _assert_population([ex['image_id'] for ex in batch], [len(p) for p in pairs], self.batch_layout)
            slots = _slots(self.batch_layout)
            for i in range(9, len(slots)):
                if not torch.equal(pixel[i], pixel[slots[i]]):
                    raise ValueError('Padding pixel is not an exact duplicate')
            self.input = {
                "image_ids": [str(ex.get("image_id", "")) for ex in batch],
                "batch_size": len(batch),
                "pixel": _tensor_hash(pixel),
                "per_image_pixel_sha256": [_tensor_hash(x) for x in pixel],
                "pixel_device": str(pixel.device),
                "pixel_dtype": str(pixel.dtype),
                "pixel_shape": list(pixel.shape),
                "obj_boxes_224_shape": [list(x.shape) for x in boxes],
                "pair_counts": [len(x) for x in pairs],
                'source_indices': [FINAL_START + i for i in slots],
                'duplicate_slots': [{'slot': i, 'duplicates_slot': slots[i]} for i in range(9, len(slots))],
                'original9_pixel': _tensor_hash(pixel[:9]),
                'obj_boxes_224': [_tensor_hash(x) for x in boxes],
                'pairs': [[[int(p[0]), int(p[1])] for p in ps] for ps in pairs],
                'historical_pixel_identity': 'UNPROVEN_FROM_HISTORICAL_DUMPS',
            }
            self.events.append({"event": "prepared_input"})
        return result

    def _install_prototype_guard(self, model: torch.nn.Module) -> None:
        owner = model.module if hasattr(model, "module") else model
        if not hasattr(owner, "adaptive_predicate_logits"):
            return
        original = owner.adaptive_predicate_logits
        self._original_adaptive = original
        self._adaptive_owner = owner

        def guarded(rel_feats):
            self.events.append({
                "event": "predicate_prototypes_read",
                "before_rel_feat_returned": not any(
                    e.get("event") == "rel_feat_returned" for e in self.events
                ),
                "rel_feat_shape": list(rel_feats.shape),
            })
            return original(rel_feats)

        owner.adaptive_predicate_logits = guarded

    def forward(self, original, cfg, model, clip_model, processor, batch, device, *args, **kwargs):
        self.forward_count += 1
        if self.forward_count != 1:
            raise RuntimeError('Only one relational forward is permitted per condition')
        if self.first_forward:
            self.runtime = _runtime_settings(device, cfg)
            self.before = {
                "model": _object_hashes(model.module if hasattr(model, "module") else model),
                "clip": _object_hashes(clip_model.module if hasattr(clip_model, "module") else clip_model),
                "model_modes": _module_modes(model.module if hasattr(model, "module") else model),
                "clip_modes": _module_modes(clip_model.module if hasattr(clip_model, "module") else clip_model),
                "rng": _rng_state(device.type == "cuda"),
            }
            self._install_prototype_guard(model)
            if any(self.before['model_modes'].values()) or any(self.before['clip_modes'].values()):
                raise RuntimeError('All model/CLIP modules must be in eval mode')
            self.events.append({"event": "forward_start"})
        owner = model.module if hasattr(model, 'module') else model
        cls = type(owner)
        original_getattr = cls.__getattr__
        def guarded_getattr(obj, name):
            if obj is owner and name in ('predicate_prototypes', '_readout_v2_anchor'):
                self.events.append({'event': 'upstream_prototype_attribute_read', 'name': name})
                raise RuntimeError('Prototype/anchor read before rel_feat returned')
            return original_getattr(obj, name)
        cls.__getattr__ = guarded_getattr
        try:
            result = original(cfg, model, clip_model, processor, batch, device, *args, **kwargs)
        finally:
            cls.__getattr__ = original_getattr
        if self.first_forward:
            rels = result[1]
            self.events.append({"event": "rel_feat_returned"})
            self.rel_feat_per_image = [x.detach().cpu().clone() for x in rels]
            self.rel_feat = torch.cat(self.rel_feat_per_image, dim=0)
            self.after = {
                "model": _object_hashes(model.module if hasattr(model, "module") else model),
                "clip": _object_hashes(clip_model.module if hasattr(clip_model, "module") else clip_model),
                "rng": _rng_state(device.type == "cuda"),
                'model_modes': _module_modes(owner),
                'clip_modes': _module_modes(clip_model),
            }
            self.first_forward = False
        return result

    def restore(self) -> None:
        if self._adaptive_owner is not None and self._original_adaptive is not None:
            self._adaptive_owner.adaptive_predicate_logits = self._original_adaptive


@contextlib.contextmanager
def _instrument_evals(inst: Instrument):
    import openvocab_rel.evals as ev
    old_prepare = ev._prepare_eval_batch
    old_forward = ev._forward_eval_batch

    def prepare(cfg, batch, processor, device):
        return inst.prepare(old_prepare, cfg, batch, processor, device)

    def forward(cfg, model, clip_model, processor, batch, device, *args, **kwargs):
        return inst.forward(old_forward, cfg, model, clip_model, processor, batch, device, *args, **kwargs)

    ev._prepare_eval_batch = prepare
    ev._forward_eval_batch = forward
    try:
        yield
    finally:
        ev._prepare_eval_batch = old_prepare
        ev._forward_eval_batch = old_forward
        inst.restore()


def _checkpoint_p(path: Path) -> Optional[torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = ckpt.get("model", ckpt)
    value = model.get("predicate_prototypes") if isinstance(model, dict) else None
    return value.detach().cpu() if isinstance(value, torch.Tensor) else None


class _FinishedBoundedEval(Exception):
    pass


def _install_checkpoint_p(model, checkpoint_p):
    if checkpoint_p is None:
        raise ValueError('Explicit checkpoint-P control requires a saved prototype tensor')
    installed = model.predicate_prototypes
    if installed.shape != checkpoint_p.shape or installed.dtype != checkpoint_p.dtype:
        raise ValueError('Checkpoint P shape/dtype differs; refusing implicit conversion')
    with torch.no_grad():
        installed.copy_(checkpoint_p.to(installed.device))
    if _tensor_hash(installed) != _tensor_hash(checkpoint_p):
        raise ValueError('Installed P does not exactly match checkpoint P')


def _validate_historical_population(first9):
    """Read-only exact annotation check; never compare on counts alone."""
    checks = {}
    fields = ('image_id', 'pairs', 'pair_index', 'obj_labels', 'obj_boxes',
              'subj_label', 'obj_label', 'gt_subj_idx', 'gt_obj_idx', 'gt_pred',
              'gt_subj_label', 'gt_obj_label')
    for name in ('eval_C1', 'eval_readout_v2_R2_pilot_v3'):
        historic = torch.load(ROOT / 'runs' / name / 'pair_logits.pt', map_location='cpu', weights_only=False)
        for field in fields:
            for i, current in enumerate(first9[field]):
                previous = historic[field][FINAL_START+i]
                same = torch.equal(current, previous) if isinstance(current, torch.Tensor) else current == previous
                if not same:
                    raise ValueError(f'Historical population mismatch: {name}/{field}/{i}')
        checks[name] = 'EXACT_MATCH'
        del historic
    return checks


def _run_one(plan: Dict[str, Any], args: argparse.Namespace, work: Path) -> Dict[str, Any]:
    import openvocab_rel.evals as ev
    from openvocab_rel import train as train_mod
    from openvocab_rel.models.relational_model import RelationalModel

    ckpt = ROOT / "checkpoints" / plan["checkpoint"]
    cond_dir = work / plan["condition"]
    cond_dir.mkdir(parents=True, exist_ok=False)
    dump = cond_dir / "pair_logits.pt"
    batch_layout = plan["batch"]
    restore_dataset = _final_dataset_patch(batch_layout)
    inst = Instrument(batch_layout)
    original_init = RelationalModel.init_readout_v2
    original_load = RelationalModel.load_state_dict
    load_audit = {}
    checkpoint_p = _checkpoint_p(ckpt)
    explicit_p = plan['condition'].startswith('Zp_')
    registration_only = plan['condition'].startswith('Y_')
    if explicit_p and checkpoint_p is None:
        raise ValueError('Missing checkpoint P')

    def init_readout(self, E):
        original_init(self, E)
        if explicit_p:
            _install_checkpoint_p(self, checkpoint_p)

    if explicit_p:
        RelationalModel.init_readout_v2 = init_readout

    def audited_load(self, state, *a, **kw):
        saved = torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)
        saved_model = saved.get('model', saved)
        skipped = sorted(set(saved_model) - set(state))
        expected_skipped = ['predicate_prototypes'] if checkpoint_p is not None else []
        if skipped != expected_skipped:
            raise ValueError(f'Unexpected skipped checkpoint tensors: {skipped}')
        load_audit.update(skipped_keys=skipped, loaded_keys=sorted(state),
                          prototype_existed_at_resume='predicate_prototypes' in self.state_dict())
        result = original_load(self, state, *a, **kw)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f'Incomplete common model load: {result}')
        return result
    RelationalModel.load_state_dict = audited_load

    argv = [
        "--stage", "3", "--gpu_preset", "l4_24gb", "--eval_only", "true",
        "--epochs", "0", "--resume", "true",
        "--resume_from", str(ckpt), "--vg150_root", str(DATA_ROOT),
        "--vg150_enabled", "true", "--vg150_source", "local-jsonl",
        "--device", "cuda", "--batch_size", str(BATCH_SIZE),
        "--num_workers", "0", "--clip_input_res", "336",
        "--samples_per_epoch", "0", "--eval_batches", "1",
        "--eval_fast_mode", "false", "--torch_compile", "false",
        '--seed', '1234',
        "--explicit_spoa_enabled", "false", "--text_conditioned_projection_enabled", "false",
        "--relationness_enabled", "false", "--eval_sgg_use_relationness", "false",
        "--eval_sgg_predicate_score_mode", "ensemble",
        "--eval_sgg_predicate_ensemble_alpha", "0.0",
        "--adaptive_calibration_enabled", "true", "--bayes_calibration_weight", "0.0",
        "--freq_bias_enabled", "true", "--freq_bias_path",
        str(DATA_ROOT / "frequency_prior_train.json"), "--freq_bias_alpha", "3.75",
        "--freq_bias_smoothing", "1.0", "--eval_sgg_use_gt_pairs", "true",
        "--eval_sgg_grounding_dino_enabled", "false",
        "--eval_sgg_dump_pair_logits_path", str(dump),
        "--eval_sgg_dump_rel_feat", "true",
        "--geom_input_pixel_space", "true", "--geom_fourier_scale", "0.01",
        "--readout_v2_enabled", "true" if plan["prototype_registration"] != "none" and not registration_only else "false",
        "--run_name", f"paper_c_gd_{plan['condition']}", "--out_dir", str(cond_dir),
        "--save_metrics_json", str(cond_dir / "metrics.jsonl"),
    ]

    original_core = train_mod._run_core_evals
    setup = {}
    def bounded_core(_args, cfg, metrics, model, clip, processor, loader, device, *rest, **kw):
        if device.type != 'cuda':
            raise RuntimeError('CPU fallback is forbidden for the GPU matrix')
        if registration_only:
            # Registration only: use the recorded R0 E, no extra CLIP forward,
            # no anchor buffer, and no change to existing requires_grad flags.
            cached = torch.load(ROOT / 'runs/eval_C1/pair_logits.pt', map_location='cpu', weights_only=False)
            model.register_parameter('predicate_prototypes', torch.nn.Parameter(cached['pred_emb'].to(device).clone()))
            del cached
        has_p = 'predicate_prototypes' in dict(model.named_parameters())
        if has_p != (plan['prototype_registration'] != 'none'):
            raise RuntimeError('Prototype registration does not match condition')
        setup.update(effective_config=dict(vars(cfg)),
                     installed_P=_tensor_hash(model.predicate_prototypes) if has_p else None,
                     prototype_requires_grad=model.predicate_prototypes.requires_grad if has_p else None)
        if explicit_p and setup['installed_P'] != _tensor_hash(checkpoint_p):
            raise ValueError('Checkpoint P changed between installation and evaluation')
        ev.eval_sgg_standard(cfg, model, clip, processor, loader, device, max_batches=1)
        raise _FinishedBoundedEval()
    train_mod._run_core_evals = bounded_core
    original_step = torch.optim.AdamW.step
    def forbidden_step(*a, **kw):
        raise RuntimeError('Optimizer steps are forbidden in the forensic matrix')
    torch.optim.AdamW.step = forbidden_step
    # Retain the corresponding historical wrapper's observational patches.
    import importlib
    wrapper = importlib.import_module('tools.readout_v2_evaluate' if plan['condition'].startswith(('Z_', 'Zp_')) else 'tools.c0_c1_evaluate')
    from openvocab_rel import geometry as geom
    import openvocab_rel.models.relational_model as relmod
    old_fp, old_g, old_rg = relmod.ProgressiveRelationalDecoder.forward_pairs, geom.geom_feats_torch, relmod.geom_feats_torch
    relmod.ProgressiveRelationalDecoder.forward_pairs = wrapper._patched_fp
    geom.geom_feats_torch = wrapper._patched_geom
    relmod.geom_feats_torch = wrapper._patched_geom
    try:
        with _instrument_evals(inst):
            try:
                train_mod.main(argv)
            except _FinishedBoundedEval:
                pass
    finally:
        restore_dataset()
        RelationalModel.init_readout_v2 = original_init
        RelationalModel.load_state_dict = original_load
        train_mod._run_core_evals = original_core
        torch.optim.AdamW.step = original_step
        relmod.ProgressiveRelationalDecoder.forward_pairs = old_fp
        geom.geom_feats_torch, relmod.geom_feats_torch = old_g, old_rg

    payload = torch.load(dump, map_location="cpu", weights_only=False)
    first9 = _trim_dump(payload, batch_layout)
    historical_population = _validate_historical_population(first9)
    trimmed_dump = cond_dir / "pair_logits_first9.pt"
    torch.save(first9, trimmed_dump)
    rel_first9 = torch.cat(inst.rel_feat_per_image[:FINAL_COUNT], dim=0)
    result: Dict[str, Any] = {
        "condition": plan["condition"], "status": "COMPLETED",
        "work_dir": str(cond_dir), "dump": str(dump),
        "trimmed_dump_first9": str(trimmed_dump),
        "runtime": inst.runtime,
        "input": inst.input, "before": inst.before, "after": inst.after,
        "prototype_read_events": inst.events,
        "prototype_checkpoint_present": checkpoint_p is not None,
        'checkpoint_P_hash': _tensor_hash(checkpoint_p) if checkpoint_p is not None else None,
        'setup': setup,
        'historical_population': historical_population,
        'pid': os.getpid(),
        'forward_count': inst.forward_count,
        'argv': argv,
        'optimizer_steps': 0,
        'adaptation_bypassed': True,
        'downstream_adaptive_scoring': 'allowed only after guarded rel_feat return; no prototype update',
        "prototype_source": "explicit_checkpoint_P" if explicit_p else ('cached_R0_E_registration_only' if registration_only else (
            "registered_from_E_after_resume" if plan["prototype_registration"] != "none" else "none"
        )),
        "rel_feat": _tensor_hash(inst.rel_feat) if inst.rel_feat is not None else None,
        "rel_feat_first9": _tensor_hash(rel_first9),
        "rel_feat_tensor": rel_first9,
        "relevant_module_names": sorted(set(inst.before.get("model_modes", {}).keys()) | set(inst.before.get("clip_modes", {}).keys())),
        "prototype_parameter_hash": {
            "before": inst.before.get("model", {}).get("parameters", {}).get("predicate_prototypes"),
            "after": inst.after.get("model", {}).get("parameters", {}).get("predicate_prototypes"),
        },
        "downstream": {},
    }
    for key in ("text_logits", "model_logits", "cls_logits", "adaptive_logits", "prior_rows"):
        values = first9.get(key)
        if isinstance(values, list) and values:
            result["downstream"][key] = _tensor_hash(torch.cat(values, dim=0))
    result["checkpoint_load_note"] = (
        "train.py resumes before init_readout_v2; actual R2 construction therefore "
        "must be interpreted using the recorded prototype_source and skipped-key audit"
    )
    result["checkpoint_state_audit"] = {
        **load_audit,
        "checkpoint_contains_predicate_prototypes": checkpoint_p is not None,
        "train_resume_precedes_init_readout_v2": True,
        "explicit_checkpoint_P_control": plan["condition"].startswith("Zp_"),
    }
    if tuple(rel_first9.shape) != (65, 768):
        raise ValueError(f'Unexpected rel_feat population/shape: {rel_first9.shape}')
    native_path = cond_dir / 'rel_feat_native.pt'
    torch.save(rel_first9, native_path)
    result['native_rel_feat_path'] = str(native_path)
    result['rng_changed_during_forward'] = inst.before['rng'] != inst.after['rng']
    result['state_changes'] = {}
    for section in ('model', 'clip'):
        result['state_changes'][section] = {}
        for category in ('parameters', 'buffers', 'plain_module_state'):
            before, after = inst.before[section][category], inst.after[section][category]
            result['state_changes'][section][category] = [
                k for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)]
    return result


def _smoke_wprd(dump: Path, work: Path) -> Dict[str, Any]:
    """Compute final-batch WPRD using existing CPU analysis code."""
    from tools.cprime_mechanism import Mech
    from tools.within_pair_discrimination import Groups, wprd

    b = Mech(str(dump), str(DATA_ROOT / 'frequency_prior_train.json'), 'raw50')
    _assert_population(b.meta['image_id'], b.pairs_per_image, 'final_9')
    if b.n_gt != 65:
        raise ValueError('WPRD must use precisely the original 65 GT rows')
    g = Groups(b)
    out: Dict[str, Any] = {"n_images": b.n_images, "n_gt_rows": int(b.gt_row.numel())}
    channels = {'model': b.model, 'text': b.text_norm51, 'cls': b.cls_norm51}
    if 'adaptive_logits' in b.meta:
        channels['adaptive'] = torch.cat([b._norm(x)[:, b.fg_cols] for x in b.meta['adaptive_logits']])
    for name, score in channels.items():
        value = wprd(g, score[b.gt_row], cap=64)
        out[name] = {k: value.get(k) for k in ('n_cells', '_vals', '_wts', 'n_comparisons')}
        for k in ('wprd_macro', 'wprd_weighted'):
            out[name][k] = value[k] if value['n_cells'] else None
    out['scope'] = 'local 65-row diagnostic; not the full registered WPRD population'
    out['full_endpoint_inertness_established'] = False
    return out


def _preflight() -> Dict[str, Any]:
    try:
        p = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=False, timeout=20)
        smi = {"returncode": p.returncode, "output": p.stdout[-4000:]}
    except Exception as exc:
        smi = {"error": f"{type(exc).__name__}: {exc}"}
    busy = None
    query_ok = False
    if smi.get("returncode") == 0:
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=20,
        )
        query_ok = q.returncode == 0
        busy = [line.strip() for line in q.stdout.splitlines() if line.strip()]
    return {
        "nvidia_smi": smi, "compute_processes": busy,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        'compute_query_ok': query_ok,
    }


def _gpu_ready(preflight):
    return (preflight['nvidia_smi'].get('returncode') == 0
            and preflight.get('compute_query_ok') is True
            and preflight.get('compute_processes') == []
            and preflight['torch_cuda_available']
            and preflight['torch_cuda_device_count'] == 1)


def _launch_condition(plan, args, work):
    # Each call execs a new interpreter, including both repeat conditions.
    # No CUDA allocator/cuDNN autotuner state survives from an earlier row.
    command = [sys.executable, str(Path(__file__).resolve()), '--out', str(args.out),
               '--work-dir', str(work), '--condition', plan['condition']]
    env = dict(os.environ, PYTHONHASHSEED='1234', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with (work / (plan['condition'] + '.log')).open('x') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return torch.load(work / plan['condition'] / 'record.pt', map_location='cpu', weights_only=False)


def _comparisons(results):
    tensors = {r['condition']: r.pop('rel_feat_tensor') for r in results}
    comparisons = {}
    for i, a in enumerate(results):
        for b in results[:i]:
            key = a['condition'] + '__vs__' + b['condition']
            comparisons[key] = tensor_error_metrics(tensors[a['condition']], tensors[b['condition']])
    for r in results:
        r['rel_feat_vs_X_R0_noP_9'] = tensor_error_metrics(tensors[r['condition']], tensors['X_R0_noP_9'])
        original = r['condition'].removesuffix('_repeat')
        if original != r['condition']:
            r['same_condition_repeatability'] = tensor_error_metrics(tensors[r['condition']], tensors[original])
            baseline = next(x for x in results if x['condition'] == original)
            r['repeat_controls_match'] = {
                k: _without_output_paths(r[k]) == _without_output_paths(baseline[k])
                for k in ('input', 'before', 'runtime')}
    return comparisons


def _without_output_paths(value):
    """Only run labels/output destinations may differ in fresh repeats."""
    paths = {'run_name', 'out_dir', 'save_metrics_json', 'eval_sgg_dump_pair_logits_path'}
    if isinstance(value, dict):
        return {k: _without_output_paths(v) for k, v in value.items() if k not in paths}
    if isinstance(value, list):
        return [_without_output_paths(v) for v in value]
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="new, versioned JSON artifact path")
    ap.add_argument("--work-dir", required=True, help="new directory for per-condition dumps")
    ap.add_argument('--condition', choices=[p['condition'] for p in planned_gpu_matrix()], help=argparse.SUPPRESS)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    work = Path(args.work_dir).resolve()
    args.out, args.work_dir = str(out), str(work)
    if args.condition:
        # Private worker path: independently repeat GPU preflight before any
        # train/model import or allocation. No same-process condition reuse.
        if not _gpu_ready(_preflight()):
            raise RuntimeError('GPU is unavailable, busy, or cannot be inspected')
        plan = next(p for p in planned_gpu_matrix() if p['condition'] == args.condition)
        record = _run_one(plan, args, work)
        record['final_batch_wprd'] = _smoke_wprd(Path(record['trimmed_dump_first9']), work)
        torch.save(record, work / args.condition / 'record.pt')
        return 0
    if out.exists() or work.exists():
        raise FileExistsError("refusing to overwrite an existing forensic artifact or work directory")
    work.mkdir(parents=True)

    preflight = _preflight()
    artifact: Dict[str, Any] = {
        "schema": "paper_c_gd_final_batch_gpu_reproduction_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {"final_images_only": True, "full_validation": False,
                   "training": False, "historical_artifacts_modified": False},
        "preflight": preflight,
        "planned_matrix": planned_gpu_matrix(),
        "required_condition_fields": required_condition_fields(),
    }
    if not _gpu_ready(preflight):
        artifact["conditions"] = [dict(row, status="NOT_RUN", reason="GPU preflight failed or device busy")
                                   for row in planned_gpu_matrix()]
        artifact["status"] = "GD-UNRESOLVED"
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out, artifact)
        print(f"[not run] GPU preflight failed or device busy; wrote {out}")
        return 0

    results: List[Dict[str, Any]] = []
    for plan in planned_gpu_matrix():
        try:
            results.append(_launch_condition(plan, args, work))
        except Exception as exc:
            artifact.update(status='GD-UNRESOLVED', failed_condition=plan['condition'],
                            error=f'{type(exc).__name__}: {exc}', human_interpretation_required=True)
            artifact['completed_records'] = [r['work_dir'] for r in results]
            _write_json(out, artifact)
            raise
    artifact['pairwise_rel_feat'] = _comparisons(results)

    # The automatic classifier deliberately refuses to promote a result.  A
    # human must inspect the complete pattern and supply the proven-cause and
    # endpoint flags under the locked rule.
    artifact["conditions"] = results
    artifact["decision_inputs"] = {
        "reproduction_complete": True,
        "causal_explanation_proven": False,
        "endpoint_operationally_irrelevant": False,
        "endpoint_materially_changed": False,
    }
    artifact["status"] = classify_gd(**artifact["decision_inputs"])
    artifact['automatic_promotion_enabled'] = False
    artifact['human_interpretation_required'] = True
    artifact['decision_flags_are_unassessed_placeholders'] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, artifact)
    print(f"[written] {out}\nstatus={artifact['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
