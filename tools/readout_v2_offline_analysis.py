#!/usr/bin/env python
"""Paper C -- Readout v2, offline analysis library.

CPU only. Reads existing checkpoints / `pair_logits.pt` dumps / `result.json`
files read-only. Never touches the GPU, never modifies a checkpoint or a
dump, never launches training or evaluation. Designed to be run the moment
`tools/readout_v2_evaluate.py`'s R2 endpoint finishes, without waiting on any
further GPU work.

This module is a *library* (import and call), plus a thin `--check` CLI for
the checks that don't need two arms to compare. It deliberately does not
duplicate `tools/within_pair_discrimination.py`'s or `tools/c0_c1_compare.py`'s
own CLIs -- it reuses their code (via the same `importlib`-by-path pattern
every `tools/*.py` script in this repo already uses, since these are not a
package) and adds exactly the pieces this repo's existing tooling is missing
for the R0-vs-R2 comparison:

1. `verify_contract_match` -- `tools/c0_c1_compare.py`'s gate D4 hard-codes
   the assumption that the two compared arms' geometry contracts DIFFER
   (correct for C0 pre-repair vs C1 repaired). R0 (`runs/eval_C1/result.json`)
   and R2 (`readout_v2_evaluate.py`'s output) have the SAME contract by
   design -- geometry is a controlled variable for Readout v2, not the
   subject of the experiment. Reusing D4 unmodified against R0/R2 spuriously
   fails a perfectly valid comparison. This function checks the RIGHT
   invariant for either case, selected explicitly by the caller.
2. `verify_paired_dumps` -- `result.json`'s `cell_values` is a flat float
   list with no per-cell identity recorded. `c0_c1_compare.py`'s D1/D2 gates
   check population *counts* match, not that `cell_values[i]` in both arms
   is actually the same (group, predicate-pair) cell. This function
   recomputes both arms' cells from their raw dumps and diffs the cell
   *keys*, not just the counts, before any pairing is trusted.
3. `checkpoint_integrity`, `prototype_drift`, `shuffled_prototype_control`,
   `random_text_control`, `calibration_report`, `prior_correlation` --
   codify checks this session already performed once by hand (during the
   Readout v2 pilot's recovery/verification), as reusable, tested functions.

See `docs/PAPER_C_READOUT_V2_ANALYSIS_PLAN.md` for the full checklist this
module implements pieces of, and for which items remain manual /
result.json-only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent


def _load(name: str):
    """Import a sibling tools/*.py module by path (they are scripts, not a
    package -- this mirrors the exact pattern every existing tools/*.py file
    in this repo already uses, e.g. tools/readout_v2_evaluate.py's `_load`)."""
    spec = importlib.util.spec_from_file_location(name, str(_TOOLS_DIR / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================ 1. checkpoint integrity
def checkpoint_integrity(ckpt_path: str, expected_only_trainable: str = "predicate_prototypes") -> Dict[str, Any]:
    """Static, CPU-only checkpoint checks (no forward pass).

    Mirrors the exact checks performed by hand on `checkpoints/
    readout_v2_pilot_seed1234_v3.pt` during this program's recovery session:
    CLIP present, predicate_prototypes present with the right shape/dtype,
    finite, and -- via the optimizer state -- proof that no OTHER tensor was
    ever actually updated (AdamW only creates a state entry the first time a
    parameter receives a gradient and is stepped, so a state dict with
    exactly one entry is stronger evidence than reading `requires_grad`
    alone, which is a training-time-only property not present in this file).
    """
    out: Dict[str, Any] = {"ckpt": ckpt_path, "checks": {}}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checks = out["checks"]

    checks["has_clip_key"] = "clip" in ckpt
    checks["clip_tensor_count"] = len(ckpt.get("clip", {})) if "clip" in ckpt else 0
    checks["has_model_key"] = "model" in ckpt
    model_sd = ckpt.get("model", {})
    checks["has_predicate_prototypes"] = expected_only_trainable in model_sd

    if expected_only_trainable in model_sd:
        p = model_sd[expected_only_trainable].float()
        checks["predicate_prototypes_shape"] = list(p.shape)
        checks["predicate_prototypes_finite"] = bool(torch.isfinite(p).all())
        row_norms = p.norm(dim=-1)
        checks["row_norm_mean"] = float(row_norms.mean())
        checks["row_norm_min"] = float(row_norms.min())
        checks["row_norm_max"] = float(row_norms.max())

    optim = ckpt.get("optim", {})
    checks["has_optim_key"] = bool(optim)
    if optim:
        n_groups = len(optim.get("param_groups", []))
        checks["optimizer_n_param_groups"] = n_groups
        state = optim.get("state", {})
        checks["optimizer_n_state_entries"] = len(state)
        # The single, strongest gate: if training was genuinely restricted to
        # one tensor, AdamW's state dict has EXACTLY one entry (it is created
        # lazily, only on the first step that tensor actually receives).
        checks["only_one_tensor_ever_stepped"] = len(state) == 1
        if len(state) >= 1 and n_groups >= 3:
            proto_group = optim["param_groups"][-1]
            checks["last_group_lr"] = float(proto_group.get("lr", float("nan")))
            checks["last_group_is_min_lr_stuck"] = bool(abs(proto_group.get("lr", 0.0) - 1e-7) < 1e-9)
            first_id = proto_group["params"][0] if proto_group.get("params") else None
            if first_id is not None and first_id in state:
                st = state[first_id]
                checks["last_group_step"] = float(st.get("step", -1))
                if "exp_avg" in st:
                    checks["last_group_exp_avg_abs_mean"] = float(st["exp_avg"].abs().mean())

    exp = ckpt.get("experiment", {})
    checks["has_experiment_snapshot"] = bool(exp)
    if exp:
        checks["predicate_vocab_hash"] = exp.get("predicate_vocab_hash")
        checks["num_predicates"] = exp.get("num_predicates")

    out["all_pass"] = bool(
        checks.get("has_clip_key")
        and checks.get("has_predicate_prototypes")
        and checks.get("predicate_prototypes_finite")
        and checks.get("only_one_tensor_ever_stepped")
        and not checks.get("last_group_is_min_lr_stuck", True)
    )
    return out


def historical_checkpoint_sha256_check(path: str, expected_sha256: str) -> Dict[str, Any]:
    """Verify a historical checkpoint's byte content is untouched.

    Use this for `checkpoints/demo_best/pure_best_adapt_light_mR50.pt` and
    `checkpoints/C1_seed1234.pt` -- never for the pilot's own output
    checkpoint (which is expected, correctly, to differ from anything)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    return {"path": path, "expected": expected_sha256, "actual": actual,
            "match": actual == expected_sha256}


# ============================================================ 2/3. population + pairing
def verify_contract_match(result_a: Dict[str, Any], result_b: Dict[str, Any],
                           expect_identical: bool) -> Dict[str, Any]:
    """The corrected replacement for `c0_c1_compare.py`'s gate D4.

    `expect_identical=True` for R0-vs-R2 (geometry is a controlled variable
    for Readout v2 -- both arms MUST share the same contract, verified here
    with `==`, not required to differ). `expect_identical=False` reproduces
    the original C0-vs-C1 semantics (the two arms' contracts must differ,
    e.g. C0's `geom_input_pixel_space=False` vs C1's `True`) for anyone who
    later wants to reuse this same function for a geometry-varying pair.
    """
    ca, cb = result_a.get("contract"), result_b.get("contract")
    same = bool(ca == cb)
    passed = same if expect_identical else (not same)
    return {"contract_a": ca, "contract_b": cb, "contracts_identical": same,
            "expected_identical": expect_identical, "pass": passed}


def _mech_with_channel(dump_path: str, prior_path: str, channel: str, scheme: str = "raw50"):
    """Build a Mech-family object exposing `.channel_scores` for `channel` in
    {"text", "classifier", "prior", "adaptive"}, reusing cprime_mechanism.Mech
    (and its Bench base) completely unmodified."""
    MECH = _load("cprime_mechanism")

    class _ChannelMech(MECH.Mech):
        def __init__(self):
            super().__init__(dump_path, prior_path, scheme)
            if channel == "text":
                self.channel_scores = self.fixed_ensemble(0.0)
            elif channel == "classifier":
                self.channel_scores = self.fixed_ensemble(1.0)
            elif channel == "prior":
                self.channel_scores = self.prior
            elif channel == "adaptive":
                d = self.meta
                has = "adaptive_logits" in d and isinstance(d["adaptive_logits"], list)
                if not has:
                    raise ValueError(f"{dump_path} has no adaptive_logits channel")
                av = []
                for i in range(self.n_images):
                    av.append(self._norm(d["adaptive_logits"][i])[:, self.fg_cols])
                self.channel_scores = torch.cat(av, 0)
            else:
                raise ValueError(f"unknown channel {channel!r}")

    return _ChannelMech()


def wprd_with_cell_keys(dump_path: str, prior_path: str, channel: str, cap: int = 64,
                         seed: int = 0, scheme: str = "raw50") -> Dict[str, Any]:
    """Recompute WPRD directly from a raw dump AND return a per-cell identity
    key alongside each value, in the same order as the values -- the piece
    `result.json`'s flat `cell_values` list is missing, and the reason a
    paired bootstrap across two independently-produced dumps needs its own,
    fresh verification rather than trusting positional alignment."""
    WPD = _load("within_pair_discrimination")
    B = _mech_with_channel(dump_path, prior_path, channel, scheme)
    Gs = WPD.Groups(B)
    r = WPD.wprd(Gs, B.channel_scores[B.gt_row], cap=cap, seed=seed)

    keys: List[str] = []
    for p in range(Gs.G):
        cls = sorted(Gs.classes_of[p])
        if len(cls) < 2:
            continue
        for ai in range(len(cls)):
            for bi in range(ai + 1, len(cls)):
                a, b = cls[ai], cls[bi]
                keys.append(f"{Gs.key_of[p]}::{B.classes[a]}|{B.classes[b]}")
    assert len(keys) == len(r["_vals"]), "cell-key / cell-value count mismatch -- bug in this function"

    return {"dump": dump_path, "channel": channel, "cap": cap,
            "population": {"images": B.n_images, "pairs": int(B.channel_scores.shape[0]),
                            "gt_rows": B.n_gt, "cells": r["n_cells"]},
            "wprd_macro": r["wprd_macro"], "wprd_weighted": r["wprd_weighted"],
            "frac_cells_above_half": r["frac_cells_above_half"],
            "cell_keys": keys, "cell_values": r["_vals"], "cell_weights": r["_wts"]}


def verify_paired_dumps(dump_a: str, dump_b: str, prior_path: str,
                         channel_a: str, channel_b: str, cap: int = 64) -> Dict[str, Any]:
    """The real "exact population match" gate: recompute both arms' WPRD
    cells from their raw dumps and diff the cell KEYS, not just the counts.

    Passing this is a precondition for `paired_cell_bootstrap` -- if the key
    lists differ at all, the two `cell_values` arrays are not aligned and
    must not be paired positionally."""
    ra = wprd_with_cell_keys(dump_a, prior_path, channel_a, cap=cap)
    rb = wprd_with_cell_keys(dump_b, prior_path, channel_b, cap=cap)
    pop_match = ra["population"] == rb["population"]
    same_len = len(ra["cell_keys"]) == len(rb["cell_keys"])
    keys_match = same_len and (ra["cell_keys"] == rb["cell_keys"])
    first_mismatch = None
    if same_len and not keys_match:
        for i, (ka, kb) in enumerate(zip(ra["cell_keys"], rb["cell_keys"])):
            if ka != kb:
                first_mismatch = {"index": i, "a": ka, "b": kb}
                break
    return {
        "population_a": ra["population"], "population_b": rb["population"],
        "population_match": pop_match,
        "n_cells_a": len(ra["cell_keys"]), "n_cells_b": len(rb["cell_keys"]),
        "cell_keys_match": keys_match, "first_mismatch": first_mismatch,
        "safe_to_pair": bool(pop_match and keys_match),
        "cell_values_a": ra["cell_values"], "cell_values_b": rb["cell_values"],
        "wprd_macro_a": ra["wprd_macro"], "wprd_macro_b": rb["wprd_macro"],
    }


# ============================================================ 5. paired bootstrap
def paired_cell_bootstrap(values_a: Sequence[float], values_b: Sequence[float],
                           n_boot: int = 2000, seed: int = 11) -> Dict[str, Any]:
    """The exact statistic `tools/c0_c1_compare.py` computes, extracted as a
    standalone, reusable function (same seed convention, same resample-cells-
    with-replacement bootstrap of the mean paired delta)."""
    vA = torch.tensor(values_a, dtype=torch.float64)
    vB = torch.tensor(values_b, dtype=torch.float64)
    if vA.numel() != vB.numel():
        raise ValueError(f"paired arrays must be the same length: {vA.numel()} vs {vB.numel()}")
    d = vB - vA
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(d), (n_boot, len(d)), generator=g)
    bs = d[idx].mean(dim=1)
    lo, hi = torch.quantile(bs, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    delta = float(d.mean())
    p_neg = float((bs < 0).double().mean())
    return {"n_cells": int(vA.numel()), "delta": delta, "ci95": [lo, hi],
            "p_delta_negative": p_neg, "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0)}


# ============================================================ 9/10/11. prototype diagnostics
def prototype_drift(ckpt_path: str, vg150_root: str = "datasets_vg150_clean",
                     clip_name: str = "openai/clip-vit-large-patch14-336") -> Dict[str, Any]:
    """Recompute E fresh from the checkpoint's OWN CLIP weights + the real
    scanned vocabulary, and compare to the checkpoint's stored P.

    This is the exact procedure already run once by hand on
    `checkpoints/readout_v2_pilot_seed1234_v3.pt` during this program's
    recovery session (mean 1-cos != 0.032, max 0.16, no collapse) --
    codified here so it can be re-run identically against the R2 pilot's
    final checkpoint (unaffected by, and independent of, the endpoint
    evaluation run).
    """
    from openvocab_rel.clip_utils import configure_clip, encode_predicate_vocab
    from openvocab_rel.prompts import pred_prompt_roles
    from openvocab_rel.datasets import scan_vg150_predicate_vocab

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    clip_model, processor, _, _ = configure_clip(clip_name, device)
    clip_model.load_state_dict(ckpt["clip"], strict=True)
    clip_model.eval()

    pred_vocab = scan_vg150_predicate_vocab(vg150_root=vg150_root)
    pred_vocab = [str(p).strip().lower() for p in pred_vocab if str(p).strip() != ""]
    pred_vocab = list(dict.fromkeys(pred_vocab))
    if "relation" not in pred_vocab:
        pred_vocab.append("relation")

    _, E = encode_predicate_vocab(clip_model, processor, pred_vocab, device,
                                   prompt_fn=pred_prompt_roles, direction="s2o")
    P = ckpt["model"]["predicate_prototypes"].float()
    E = E.float()
    if P.shape != E.shape:
        raise ValueError(f"P/E shape mismatch: {tuple(P.shape)} vs {tuple(E.shape)}")

    p_n, e_n = F.normalize(P, dim=-1), F.normalize(E, dim=-1)
    cos = (p_n * e_n).sum(-1)
    l2 = (P - E).norm(dim=-1)
    row_norm = P.norm(dim=-1)
    one_minus_cos = 1.0 - cos

    order = torch.argsort(one_minus_cos, descending=True)
    per_predicate = [{"predicate": pred_vocab[i], "one_minus_cos": float(one_minus_cos[i]),
                       "l2": float(l2[i]), "row_norm": float(row_norm[i])}
                      for i in order.tolist()]

    return {
        "ckpt": ckpt_path, "n_predicates": len(pred_vocab),
        "cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "cos_max": float(cos.max()),
        "one_minus_cos_mean": float(one_minus_cos.mean()), "one_minus_cos_max": float(one_minus_cos.max()),
        "l2_mean": float(l2.mean()), "l2_min": float(l2.min()), "l2_max": float(l2.max()),
        "row_norm_mean": float(row_norm.mean()), "row_norm_min": float(row_norm.min()),
        "row_norm_max": float(row_norm.max()),
        "any_nan": bool(torch.isnan(P).any()), "any_inf": bool(torch.isinf(P).any()),
        # Failure criteria from docs/PAPER_C_READOUT_V2_PREREGISTRATION.md sec.15:
        # "1-cos(P_i,E_i) collapses to ~1 for many i" (semantic collapse) and
        # "||P_i-E_i|| diverges" (unbounded growth). Neither is a fixed
        # numeric gate in the preregistration; these flags use a
        # deliberately generous threshold and are reported, not auto-failed.
        "semantic_collapse_flag": bool(one_minus_cos.mean() > 0.5),
        "unbounded_growth_flag": bool(l2.max() > 10.0),
        "per_predicate_by_displacement": per_predicate,
    }


# ============================================================ 15. shuffled-prototype control
def shuffled_prototype_control(dump_path: str, prior_path: str, cap: int = 64,
                                seed: int = 0, channel: str = "adaptive") -> Dict[str, Any]:
    """Permute which predicate each scored column is treated as, and recompute
    WPRD. Purely a column re-indexing of an already-dumped logits matrix --
    no CLIP, no re-scoring, no GPU. If the real result reflects genuine
    per-predicate discrimination (not just "some predicate-shaped output"),
    the shuffled version must collapse toward the prior-control band
    (~0.5), because the model's real per-row score for column k is now
    being read as if it were column shuffle(k)."""
    WPD = _load("within_pair_discrimination")
    B = _mech_with_channel(dump_path, prior_path, channel)
    Gs = WPD.Groups(B)
    real = WPD.wprd(Gs, B.channel_scores[B.gt_row], cap=cap)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(B.channel_scores.shape[1], generator=g)
    shuffled_scores = B.channel_scores[:, perm]
    shuffled = WPD.wprd(Gs, shuffled_scores[B.gt_row], cap=cap)

    return {"dump": dump_path, "channel": channel, "seed": seed,
            "real_wprd_macro": real["wprd_macro"], "shuffled_wprd_macro": shuffled["wprd_macro"],
            "collapse_to_prior_band": bool(0.47 <= shuffled["wprd_macro"] <= 0.53),
            "n_cells": real["n_cells"]}


# ============================================================ 14. random-text control
def random_text_control(dump_path: str, ckpt_path: str, prior_path: str,
                         random_predicate_strings: Sequence[str], cap: int = 64,
                         clip_name: str = "openai/clip-vit-large-patch14-336",
                         encode_fn: Optional[Callable[[Sequence[str]], torch.Tensor]] = None
                         ) -> Dict[str, Any]:
    """Score the dumped `rel_feat` (requires the eval was run with
    `--eval_sgg_dump_rel_feat true`, which both `c0_c1_evaluate.py` and
    `readout_v2_evaluate.py` already pass) against embeddings of predicate
    strings that are NOT any real VG150 predicate, through the SAME frozen
    CLIP text encoder used everywhere else in this program.

    This is a stronger control than `within_pair_discrimination.py`'s own
    "random null" arm (`torch.randn(...)`, structureless Gaussian noise):
    it tests whether CLIP's own embedding-space geometry (norm scale,
    anisotropy) could spuriously produce above-chance WPRD, which pure
    noise cannot test. `encode_fn` is injectable so this can be unit-tested
    without loading real CLIP.

    `random_predicate_strings` must have exactly `B.n_classes` entries (one
    substitute per real predicate slot) -- `Groups`/`wprd` index GT labels
    into a fixed `0..n_classes-1` space, so this control must swap out WHAT
    each column means (a random string instead of a real predicate), not
    how many columns exist.
    """
    MECH = _load("cprime_mechanism")
    WPD = _load("within_pair_discrimination")

    d = torch.load(dump_path, map_location="cpu", weights_only=False)
    if "rel_feat" not in d:
        raise ValueError(f"{dump_path} has no rel_feat channel -- was "
                          "--eval_sgg_dump_rel_feat true passed to the evaluator?")

    if encode_fn is None:
        def encode_fn(strings: Sequence[str]) -> torch.Tensor:
            from openvocab_rel.clip_utils import configure_clip, encode_predicate_vocab
            from openvocab_rel.prompts import pred_prompt_roles
            clip_model, processor, _, _ = configure_clip(clip_name, torch.device("cpu"))
            clip_model.load_state_dict(torch.load(ckpt_path, map_location="cpu",
                                                    weights_only=False)["clip"], strict=True)
            clip_model.eval()
            _, emb = encode_predicate_vocab(clip_model, processor, list(strings),
                                             torch.device("cpu"), prompt_fn=pred_prompt_roles,
                                             direction="s2o")
            return emb.float()

    B = MECH.Mech(dump_path, prior_path, "raw50")
    n_real_classes = B.n_classes
    if len(random_predicate_strings) != n_real_classes:
        raise ValueError(
            f"random_predicate_strings must have exactly one entry per real "
            f"predicate class ({n_real_classes}), got {len(random_predicate_strings)}. "
            "Groups/wprd index into the SAME class-index space as the real GT "
            "labels (0..n_classes-1) -- this control substitutes what each "
            "column MEANS (a random string's embedding instead of a real "
            "predicate's), not how many columns there are.")

    rand_embeds = encode_fn(random_predicate_strings)  # (n_real_classes, d)

    rel_feat = torch.cat([d["rel_feat"][i].float()[:, : rand_embeds.shape[-1]]
                          for i in range(B.n_images)], 0)
    if rel_feat.shape[0] != B.text_norm51.shape[0]:
        raise ValueError("rel_feat row count does not match the dump's pair-row count")

    score = F.normalize(rel_feat, dim=-1) @ F.normalize(rand_embeds, dim=-1).t()
    Gs = WPD.Groups(B)
    r = WPD.wprd(Gs, score[B.gt_row], cap=cap)
    return {"dump": dump_path, "n_real_classes": n_real_classes,
            "wprd_macro": r["wprd_macro"], "wprd_weighted": r["wprd_weighted"],
            "collapse_to_prior_band": bool(0.47 <= r["wprd_macro"] <= 0.53),
            "n_cells": r["n_cells"], "random_strings": list(random_predicate_strings)}


# ============================================================ 12. prior correlation
def prior_correlation(dump_path: str, prior_path: str, train_freq: Dict[str, int],
                       channel: str = "adaptive") -> Dict[str, Any]:
    """Correlate per-class recall DELTA (this channel vs the prior-only
    baseline) against each predicate's real training frequency.

    This is the offline generalization of this session's own manual finding
    on the pilot's prototype-displacement ranking (the 5 predicates whose
    P moved most after one epoch were exactly the 5 highest-frequency ones)
    -- applied here to the ENDPOINT's actual recall numbers, using Pearson
    correlation over the per-class recall-delta vs log-frequency."""
    B = _mech_with_channel(dump_path, prior_path, channel)
    sp = B.prior
    sc = B.channel_scores
    num_p, den = B.per_class_recall_vec(sp)
    num_c, _ = B.per_class_recall_vec(sc)
    present = den > 0
    rec_p = torch.where(present, num_p / den.clamp_min(1), torch.zeros_like(num_p))
    rec_c = torch.where(present, num_c / den.clamp_min(1), torch.zeros_like(num_c))
    delta = rec_c - rec_p

    names, deltas, freqs = [], [], []
    for c in range(B.n_classes):
        if not bool(present[c]):
            continue
        name = B.classes[c]
        f = train_freq.get(name)
        if f is None:
            continue
        names.append(name)
        deltas.append(float(delta[c]))
        freqs.append(float(f))

    if len(deltas) < 3:
        return {"n_classes_matched": len(deltas), "pearson_r": None,
                "note": "too few matched classes for a correlation"}

    x = torch.tensor(freqs, dtype=torch.float64).log()
    y = torch.tensor(deltas, dtype=torch.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = (xc.pow(2).sum().sqrt() * yc.pow(2).sum().sqrt())
    r = float((xc * yc).sum() / denom) if float(denom) > 0 else None

    order = sorted(range(len(names)), key=lambda i: -abs(deltas[i]))
    return {"n_classes_matched": len(deltas), "pearson_r_log_freq_vs_delta_recall": r,
            "top_movers": [{"predicate": names[i], "delta_recall": deltas[i],
                            "train_freq": freqs[i]} for i in order[:10]]}


# ============================================================ 13. calibration
def calibration_report(dump_path: str, channel: str = "adaptive", prior_path: str = "",
                        n_bins: int = 10) -> Dict[str, Any]:
    """Expected-Calibration-Error-style report on the scored channel's
    softmax-max-confidence vs top-1 correctness, on GT rows."""
    B = _mech_with_channel(dump_path, prior_path, channel)
    rows = B.channel_scores[B.gt_row]
    probs = torch.softmax(rows, dim=-1)
    conf, pred_col = probs.max(-1).values, probs.argmax(-1)
    pred = B.col_to_class[pred_col]
    correct = (pred == B.gt_y).float()

    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = conf.numel()
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        m = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "mean_conf": None, "accuracy": None})
            continue
        mean_conf = float(conf[m].mean())
        acc = float(correct[m].mean())
        bins.append({"lo": lo, "hi": hi, "n": cnt, "mean_conf": mean_conf, "accuracy": acc})
        ece += (cnt / n) * abs(acc - mean_conf)

    return {"dump": dump_path, "channel": channel, "n_bins": n_bins, "n_rows": n,
            "ece": float(ece), "overall_accuracy": float(correct.mean()),
            "overall_mean_confidence": float(conf.mean()), "bins": bins}


def calibration_comparison(dump_a: str, dump_b: str, channel_a: str, channel_b: str,
                            prior_path: str, n_bins: int = 10) -> Dict[str, Any]:
    """Calibration side by side for two arms (e.g. R0's `text` channel vs
    R2's `adaptive` channel) -- item 10 of the falsification battery
    (docs/PAPER_C_R2_FALSIFICATION_BATTERY.md). A calibration DIFFERENCE
    alone is not evidence of a semantic mechanism -- it could reflect
    nothing more than a temperature/scale change in the scoring function
    -- but a calibration comparison that is NOT reported at all would let
    exactly that confound go unchecked."""
    a = calibration_report(dump_a, channel=channel_a, prior_path=prior_path, n_bins=n_bins)
    b = calibration_report(dump_b, channel=channel_b, prior_path=prior_path, n_bins=n_bins)
    return {"a": a, "b": b, "delta_ece": b["ece"] - a["ece"],
            "delta_accuracy": b["overall_accuracy"] - a["overall_accuracy"],
            "delta_mean_confidence": b["overall_mean_confidence"] - a["overall_mean_confidence"]}


# ============================================================ 4. geometry-only control
def geometry_only_control(dump_path: str, prior_path: str, cap: int = 64,
                           epochs: int = 200, l2: float = 1e-4,
                           train_jsonl: str = "") -> Dict[str, Any]:
    """Item 4 of the falsification battery. Reuses
    `tools/wprd_geometry_control.py` verbatim (an existing, validated,
    CPU-only cross-fitted linear probe on 19 scale-invariant box-geometry
    numbers -- never pixels, never rel_feat, never a predicate embedding)
    rather than reimplementing it -- this program already has a working,
    previously-used geometry-only control and duplicating its math would
    risk a silent divergence.

    Fits the probe cross-validated over the SAME 5 image-level folds this
    program's other geometry work uses (`candidate_scorer_probe.fold_of_image`),
    and scores it with the SAME `within_pair_discrimination.wprd` estimator
    every other arm in this program uses -- so its number is directly
    comparable to R0's and R2's own WPRD, not a different metric.

    `train_jsonl`, if given, additionally fits the probe on the TRAIN split
    (matching the model's own train/val split) rather than cross-fitting on
    validation -- removing the probe's only structural advantage over the
    model.
    """
    WGC = _load("wprd_geometry_control")
    MECH = _load("cprime_mechanism")
    WPD = _load("within_pair_discrimination")

    B = MECH.Mech(dump_path, prior_path, "raw50")
    Gs = WPD.Groups(B)
    dev = Gs.prior_is_constant(B.prior)
    if dev >= 1e-3:
        raise AssertionError(f"prior not constant within group (dev={dev:.3e}) -- "
                              "geometry control's WPRD would not be prior-free")

    Xr = WGC.geometry_features_raw(B)
    X, _ = WGC._standardise(Xr)
    fold = WGC.fold_of_image(B, 0)
    y = B.gt_y
    geo_logits = WGC.cross_fit_logits(X, y, fold, B.n_classes, epochs, l2)
    y_shuffled = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(7))]
    geo_shuffled_logits = WGC.cross_fit_logits(X, y_shuffled, fold, B.n_classes, epochs, l2)

    out: Dict[str, Any] = {"dump": dump_path, "n_geom_features": X.shape[1] - 1}
    for name, logits in (("geometry_probe", geo_logits),
                          ("geometry_probe_shuffled_label_null", geo_shuffled_logits)):
        r = WPD.wprd(Gs, logits, cap=cap)
        out[name] = {"wprd_macro": r["wprd_macro"], "wprd_weighted": r["wprd_weighted"],
                     "n_cells": r["n_cells"], "_vals": r["_vals"]}

    if train_jsonl:
        Xt_raw, yt = WGC.train_split_geometry(train_jsonl, list(B.classes))
        Xt, tstats = WGC._standardise(Xt_raw)
        Xv, _ = WGC._standardise(Xr, tstats)
        Wt = torch.zeros(Xt.shape[1], B.n_classes, requires_grad=True)
        opt = torch.optim.LBFGS([Wt], max_iter=epochs, history_size=10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xt @ Wt, yt) + l2 * (Wt * Wt).sum()
            loss.backward()
            return loss

        opt.step(closure)
        train_fit_logits = (Xv @ Wt).detach()
        r = WPD.wprd(Gs, train_fit_logits, cap=cap)
        out["geometry_probe_train_fitted"] = {"wprd_macro": r["wprd_macro"],
                                              "wprd_weighted": r["wprd_weighted"],
                                              "n_cells": r["n_cells"], "_vals": r["_vals"]}
    return out


# ============================================================ 12. collapse / anisotropy
def collapse_anisotropy_check(matrix: torch.Tensor) -> Dict[str, Any]:
    """Item 12 of the falsification battery: has the predicate-prototype
    matrix collapsed toward a low-dimensional subspace (or a single shared
    direction), independent of whether it discriminates GT predicates on
    any particular dump? A matrix can pass every WPRD/calibration/drift
    check on the specific cells sampled and still be a degenerate,
    near-rank-1 object that would fail to generalize -- this check is
    orthogonal to (and does not require) any dump or GT label at all,
    only the matrix itself (P from a checkpoint, or E for comparison).

    Reports TWO genuinely different notions of collapse -- a matrix can
    fail either without failing the other, so both are reported, never
    collapsed into one score:
    - `cos_off_diag_mean`/`max` -- computed on RAW row directions. High
      values mean rows point the same way (a scoring function built on
      `cos(x, P_i)` cannot discriminate classes whose prototypes are
      near-parallel, regardless of how good `x` is).
    - `participation_ratio` -- computed on MEAN-CENTERED rows,
      `(sum(s))^2 / sum(s^2)` over the centered matrix's singular values.
      Centering removes exactly the common-direction signal the cosine
      check detects, so this instead measures how many directions the
      rows occupy AROUND their mean -- a matrix can have high off-diagonal
      cosine (all rows near one vector) yet, once centered, its residual
      variation can still be high-dimensional (e.g. "same vector + small
      isotropic noise" -- see the two separate tests in
      tests/test_readout_v2_offline_analysis.py). Conversely a matrix can
      have LOW cosine similarity (rows point in quite different directions)
      while still occupying only a low-rank subspace overall. Use
      `cos_off_diag_mean` to ask "do the SCORES collapse," and
      `participation_ratio` to ask "does the embedding SPACE use its full
      capacity" -- they are not substitutes for each other.
    - the spectrum's own min/max singular value ratio (condition-number
      style anisotropy indicator, same centered spectrum as above)
    """
    M = matrix.float()
    C, d = M.shape
    M_normed = F.normalize(M, dim=-1)
    sim = M_normed @ M_normed.t()
    off_diag_mask = ~torch.eye(C, dtype=torch.bool, device=M.device)
    off_diag = sim[off_diag_mask]

    centered = M - M.mean(0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered)
    except Exception:
        s = torch.linalg.svdvals(centered.double()).float()
    s = s.clamp_min(0.0)
    participation_ratio = float((s.sum() ** 2) / (s.pow(2).sum().clamp_min(1e-12)))
    max_possible_rank = min(C, d)

    return {
        "n_rows": C, "dim": d,
        "cos_off_diag_mean": float(off_diag.mean()),
        "cos_off_diag_max": float(off_diag.max()),
        "cos_off_diag_min": float(off_diag.min()),
        "singular_values_top5": s[:5].tolist(),
        "participation_ratio": participation_ratio,
        "max_possible_rank": max_possible_rank,
        "participation_ratio_frac_of_max": participation_ratio / max(1, max_possible_rank),
        "condition_ratio_min_over_max_sv": float(s[-1] / s[0]) if float(s[0]) > 0 else None,
        # Descriptive flags, not pass/fail gates -- thresholds are a judgment
        # call to be made when comparing against a baseline (E, or an
        # untrained-init P), not hard-coded here.
        "near_rank_one_flag": bool(participation_ratio < 2.0),
        "high_similarity_flag": bool(off_diag.mean() > 0.8),
    }


# ============================================================ CLI (single-arm checks only)
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="run checkpoint_integrity + prototype_drift on this checkpoint")
    ap.add_argument("--vg150_root", default="datasets_vg150_clean")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if not args.ckpt:
        ap.print_help()
        return 2

    res = {"checkpoint_integrity": checkpoint_integrity(args.ckpt),
           "prototype_drift": prototype_drift(args.ckpt, vg150_root=args.vg150_root)}
    text = json.dumps({k: v for k, v in res.items()}, indent=2,
                       default=lambda o: "<non-serializable>")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[written] {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
