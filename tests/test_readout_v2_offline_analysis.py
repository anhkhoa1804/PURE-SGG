"""Regression tests for tools/readout_v2_offline_analysis.py.

CPU only, entirely synthetic data (same dump-construction convention as
tests/test_cprime_mechanism.py) -- never reads a real checkpoint or a real
GPU-produced dump, so this suite runs in seconds regardless of whether the
Readout v2 endpoint evaluation has finished.

What is pinned:

1. checkpoint_integrity correctly distinguishes a healthy single-trainable-
   tensor checkpoint from the exact two historical failure modes this
   program already hit once for real (LR stuck at min_lr; more than one
   optimizer state entry).
2. verify_contract_match implements the CORRECTED gate: identical contracts
   must PASS when expect_identical=True (the R0-vs-R2 case) and FAIL when
   expect_identical=False (reproducing c0_c1_compare.py's original C0-vs-C1
   gate D4 semantics) -- and vice versa for differing contracts.
3. verify_paired_dumps detects both a true match (same population, same
   cell identities) and a real mismatch (different population), rather
   than trusting positional alignment of two independently-built cell lists.
4. paired_cell_bootstrap reproduces exact-zero delta for identical arrays
   and a strictly-positive CI for a uniformly-shifted array.
5. shuffled_prototype_control collapses toward the prior-control band on a
   real permutation of a dump that was constructed to have a decodable
   ("adaptive") signal only under the correct (identity) column order.
6. calibration_report's ECE is exactly 0 for a channel whose confidence
   equals its accuracy by construction, and positive when they diverge.
7. prior_correlation's Pearson r is +1 (up to float tolerance) when recall
   delta is an exact increasing linear function of log-frequency.
8. random_text_control runs end-to-end against a synthetic dump with an
   INJECTED encode_fn (no real CLIP load), and correctly flags collapse to
   the prior-control band for embeddings uncorrelated with rel_feat.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tools.readout_v2_offline_analysis import (
    calibration_report,
    checkpoint_integrity,
    paired_cell_bootstrap,
    prior_correlation,
    random_text_control,
    shuffled_prototype_control,
    verify_contract_match,
    verify_paired_dumps,
    wprd_with_cell_keys,
)


def _vg150():
    raw = json.loads(Path("datasets_vg150_clean/vocabulary/predicates.json").read_text())["idx_to_predicate"]
    return [raw[str(i)] for i in range(1, len(raw) + 1)]


def _norm(x):
    m = x.mean(-1, keepdim=True)
    s = x.std(-1, keepdim=True).clamp_min(1e-4)
    return (x - m) / s


@pytest.fixture(scope="module")
def prior_path(tmp_path_factory):
    pv = _vg150()
    p = tmp_path_factory.mktemp("prior") / "prior.json"
    p.write_text(json.dumps({
        "predicate_vocab": pv,
        "global_log_probs": [-1.0 - 0.1 * i for i in range(len(pv))],
        "default_log_prob": -20.0,
    }))
    return p


def _build_dump(path: Path, seed: int, decodable: bool, n_img: int = 12,
                 n_obj: int = 4, n_pair: int = 6, emb_dim: int = 8,
                 label_offset: int = 0, boost_fn=None, constant_prior: bool = True) -> None:
    """A synthetic cache with adaptive_logits + rel_feat added on top of the
    exact schema tests/test_cprime_mechanism.py already validates.

    `decodable=True` makes `adaptive_logits`'s correct-class column exactly
    the row's own one-hot GT label plus small noise (a real, learnable
    signal) -- so a column permutation must destroy it. `decodable=False`
    makes every column pure noise (nothing to destroy).
    `label_offset` shifts which (subject,object) label pairs are used, so
    two dumps can be built with genuinely DIFFERENT populations for the
    mismatch test. `constant_prior=True` (default) makes prior_rows a
    function of row-position only (satisfies Groups.prior_is_constant, a
    gate WPRD/geometry-control-style tools assert); `constant_prior=False`
    restores independent per-image noise, appropriate only for tests (e.g.
    prior_correlation) that read raw per-class recall and would otherwise
    have their outcome dominated by a single fixed prior realization's
    incidental correlation with whatever synthetic signal is under test.

    The GT predicate for row-position `j` is rotated by image index `i`
    (`(i + j + label_offset) % len(preds_cycle)`), not fixed per position --
    tests/test_cprime_mechanism.py's own fixture uses a fixed-per-position
    assignment, which is fine for the Mech-level tests it runs, but gives
    every (subject,object) group a CONSTANT ground-truth predicate across
    every image. WPRD's `Groups` only forms a cell where a group has >= 2
    DISTINCT observed predicates (`ndist >= 2`); without this rotation every
    group here would have exactly one, `wprd()` would find zero decidable
    cells, and `wprd_macro` would silently be `mean([]) = nan`. This bit
    exactly once while writing this fixture.
    """
    pv = _vg150()
    vocab = [p.strip().lower() for p in pv] + ["background"]
    P = len(vocab)
    g = torch.Generator().manual_seed(seed)
    preds_cycle = ["near", "wears", "on", "has", "in", "next to"]
    if boost_fn is None:
        boost_fn = lambda _p: 5.0
    keys = ("image_id", "pairs", "pair_index", "model_logits", "prior_rows",
            "text_logits", "cls_logits", "adaptive_logits", "rel_feat",
            "obj_labels", "subj_label", "obj_label", "obj_boxes",
            "gt_subj_idx", "gt_obj_idx", "gt_pred", "gt_subj_label", "gt_obj_label")
    d = {k: [] for k in keys}
    # prior_rows must be CONSTANT within a (subj_label,obj_label) group --
    # every real dump satisfies this (the prior is a function of the entity
    # CLASS pair alone), and it is a gate several tools assert
    # (Groups.prior_is_constant). Since this fixture reuses the same
    # "obj0".."obj{n_obj-1}" labels at the same row position across every
    # image, a single per-row-position prior vector, generated ONCE and
    # reused for every image, satisfies this exactly.
    fixed_prior_rows = torch.randn(n_pair, P, generator=torch.Generator().manual_seed(seed + 5000))
    for i in range(n_img):
        pairs = torch.tensor([[a, b] for a in range(3) for b in range(3) if a != b][:n_pair])
        text = torch.randn(n_pair, P, generator=g)
        cls = torch.randn(n_pair, P, generator=g)
        preds = [preds_cycle[(i + j + label_offset) % len(preds_cycle)] for j in range(n_pair)]
        pred_idx = [vocab.index(p) for p in preds]
        if decodable:
            adaptive = torch.randn(n_pair, P, generator=g) * 0.1
            for row, (pname, pidx) in enumerate(zip(preds, pred_idx)):
                adaptive[row, pidx] += boost_fn(pname)
        else:
            adaptive = torch.randn(n_pair, P, generator=g)
        rel_feat = torch.randn(n_pair, emb_dim, generator=g)
        prior_draw = torch.randn(n_pair, P, generator=g)  # preserves `g`'s exact
        # RNG-consumption pattern regardless of constant_prior, so existing
        # tests' text/cls/adaptive/rel_feat values are untouched either way.

        d["image_id"].append(str(i))
        d["pairs"].append(pairs)
        d["pair_index"].append(torch.arange(n_pair))
        d["text_logits"].append(text)
        d["cls_logits"].append(cls)
        d["adaptive_logits"].append(adaptive)
        d["rel_feat"].append(rel_feat)
        d["model_logits"].append(_norm(text))
        d["prior_rows"].append(fixed_prior_rows.clone() if constant_prior else prior_draw)
        labels = [f"obj{j + label_offset}" for j in range(n_obj)]
        d["obj_labels"].append(labels)
        d["subj_label"].append([labels[int(a)] for a, _ in pairs.tolist()])
        d["obj_label"].append([labels[int(b)] for _, b in pairs.tolist()])
        # Non-degenerate, per-image-varying boxes (not all-zero): needed by
        # geometry_only_control's feature extractor, which divides by width/
        # height -- an all-zero box would only produce NaN/Inf, silently
        # neutered by _standardise's nan_to_num rather than exercising real
        # geometry features.
        box_seed = torch.Generator().manual_seed(1000 + i)
        centers = torch.rand(n_obj, 2, generator=box_seed) * 200.0
        sizes = torch.rand(n_obj, 2, generator=box_seed) * 40.0 + 10.0
        boxes = torch.cat([centers - sizes / 2, centers + sizes / 2], dim=-1)
        d["obj_boxes"].append(boxes)
        d["gt_subj_idx"].append([int(a) for a, _ in pairs.tolist()])
        d["gt_obj_idx"].append([int(b) for _, b in pairs.tolist()])
        d["gt_pred"].append(preds)
        d["gt_subj_label"].append([labels[int(a)] for a, _ in pairs.tolist()])
        d["gt_obj_label"].append([labels[int(b)] for _, b in pairs.tolist()])
    d.update({
        "schema": "pair_logit_dump_v2", "pred_vocab": vocab,
        "background_predicate_indices": [P - 1],
        "predicate_alias_map": {},
        "ensemble_alpha": 0.0, "score_mode": "ensemble",
        "classifier_temperature": 1.0, "text_temperature": 1.0,
        "n_pairs": n_img * n_pair, "n_images": n_img,
    })
    torch.save(d, path)


@pytest.fixture(scope="module")
def dump_decodable(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "decodable.pt"
    _build_dump(p, seed=1, decodable=True)
    return p


@pytest.fixture(scope="module")
def dump_decodable_twin(tmp_path_factory):
    """Same population as dump_decodable (identical labels/GT), different
    random logits -- the correct partner for a 'safe to pair' test."""
    p = tmp_path_factory.mktemp("d") / "decodable_twin.pt"
    _build_dump(p, seed=2, decodable=True)
    return p


@pytest.fixture(scope="module")
def dump_different_population(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "different_pop.pt"
    _build_dump(p, seed=3, decodable=True, label_offset=1)
    return p


@pytest.fixture(scope="module")
def dump_noise(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "noise.pt"
    _build_dump(p, seed=4, decodable=False)
    return p


_PREDS_CYCLE = ["near", "wears", "on", "has", "in", "next to"]


@pytest.fixture(scope="module")
def dump_graduated(tmp_path_factory):
    """Per-class signal strength increases monotonically with the
    predicate's position in _PREDS_CYCLE, so a frequency table built with
    the same ordering gives a well-defined, non-degenerate Pearson
    correlation -- unlike dump_decodable's uniform per-class boost, which
    makes every class's recall saturate near 1.0 (zero variance, undefined
    correlation)."""
    p = tmp_path_factory.mktemp("d") / "graduated.pt"
    boost = lambda pname: 4.0 * (_PREDS_CYCLE.index(pname) + 1)
    _build_dump(p, seed=5, decodable=True, boost_fn=boost, constant_prior=False)
    return p


# --------------------------------------------------------------- checkpoint_integrity
def _fake_ckpt(healthy: bool, n_state_entries: int = 1, stuck_lr: bool = False):
    P = torch.randn(51, 8)
    ckpt = {
        "clip": {"text_model.embeddings.weight": torch.randn(4, 4)},
        "model": {"predicate_prototypes": P, "some_other_param": torch.randn(3, 3)},
        "optim": {
            "param_groups": [
                {"lr": 1.24e-5, "params": [0]},
                {"lr": 1.24e-6, "params": [1]},
                {"lr": (1e-7 if stuck_lr else 2e-3), "params": [840]},
            ],
            "state": {},
        },
        "experiment": {"predicate_vocab_hash": "abc123", "num_predicates": 51},
    }
    for i in range(n_state_entries):
        pid = 840 if i == 0 else 100 + i
        ckpt["optim"]["state"][pid] = {"step": torch.tensor(20.0),
                                        "exp_avg": torch.full((51, 8), 0.02)}
    if not healthy:
        # simulate a corrupted / non-finite prototype tensor as a second failure shape
        ckpt["model"]["predicate_prototypes"] = torch.full((51, 8), float("nan"))
    return ckpt


def test_checkpoint_integrity_healthy_case(tmp_path):
    p = tmp_path / "healthy.pt"
    torch.save(_fake_ckpt(healthy=True), p)
    r = checkpoint_integrity(str(p))
    assert r["checks"]["only_one_tensor_ever_stepped"] is True
    assert r["checks"]["last_group_is_min_lr_stuck"] is False
    assert r["checks"]["predicate_prototypes_finite"] is True
    assert r["all_pass"] is True


def test_checkpoint_integrity_catches_stuck_lr(tmp_path):
    p = tmp_path / "stuck.pt"
    torch.save(_fake_ckpt(healthy=True, stuck_lr=True), p)
    r = checkpoint_integrity(str(p))
    assert r["checks"]["last_group_is_min_lr_stuck"] is True
    assert r["all_pass"] is False


def test_checkpoint_integrity_catches_multiple_trained_tensors(tmp_path):
    p = tmp_path / "leaky.pt"
    torch.save(_fake_ckpt(healthy=True, n_state_entries=2), p)
    r = checkpoint_integrity(str(p))
    assert r["checks"]["optimizer_n_state_entries"] == 2
    assert r["checks"]["only_one_tensor_ever_stepped"] is False
    assert r["all_pass"] is False


def test_checkpoint_integrity_catches_non_finite_prototypes(tmp_path):
    p = tmp_path / "nanproto.pt"
    torch.save(_fake_ckpt(healthy=False), p)
    r = checkpoint_integrity(str(p))
    assert r["checks"]["predicate_prototypes_finite"] is False
    assert r["all_pass"] is False


# --------------------------------------------------------------- verify_contract_match
def test_contract_match_identical_expected_passes():
    ra = {"contract": {"geom_input_pixel_space": True, "geom_fourier_scale": 0.01}}
    rb = {"contract": {"geom_input_pixel_space": True, "geom_fourier_scale": 0.01}}
    out = verify_contract_match(ra, rb, expect_identical=True)
    assert out["contracts_identical"] is True
    assert out["pass"] is True


def test_contract_match_identical_but_expected_different_fails():
    """This is the exact scenario that would make c0_c1_compare.py's
    unmodified gate D4 wrongly fail an R0-vs-R2 comparison."""
    ra = {"contract": {"geom_input_pixel_space": True, "geom_fourier_scale": 0.01}}
    rb = {"contract": {"geom_input_pixel_space": True, "geom_fourier_scale": 0.01}}
    out = verify_contract_match(ra, rb, expect_identical=False)
    assert out["contracts_identical"] is True
    assert out["pass"] is False


def test_contract_match_c0_vs_c1_style_difference():
    ra = {"contract": {"geom_input_pixel_space": False, "geom_fourier_scale": 1.0}}
    rb = {"contract": {"geom_input_pixel_space": True, "geom_fourier_scale": 0.01}}
    out = verify_contract_match(ra, rb, expect_identical=False)
    assert out["contracts_identical"] is False
    assert out["pass"] is True


# --------------------------------------------------------------- verify_paired_dumps
def test_verify_paired_dumps_matching_population_is_safe_to_pair(
        dump_decodable, dump_decodable_twin, prior_path):
    out = verify_paired_dumps(str(dump_decodable), str(dump_decodable_twin),
                               str(prior_path), channel_a="adaptive", channel_b="adaptive")
    assert out["population_match"] is True
    assert out["cell_keys_match"] is True
    assert out["safe_to_pair"] is True
    assert out["n_cells_a"] == out["n_cells_b"] > 0


def test_verify_paired_dumps_detects_population_mismatch(
        dump_decodable, dump_different_population, prior_path):
    out = verify_paired_dumps(str(dump_decodable), str(dump_different_population),
                               str(prior_path), channel_a="adaptive", channel_b="adaptive")
    assert out["safe_to_pair"] is False


def test_wprd_with_cell_keys_count_matches_values(dump_decodable, prior_path):
    r = wprd_with_cell_keys(str(dump_decodable), str(prior_path), channel="adaptive")
    assert len(r["cell_keys"]) == len(r["cell_values"])
    assert r["population"]["cells"] == len(r["cell_values"])
    # decodable=True construction should give a strong, well-above-chance signal
    assert r["wprd_macro"] > 0.8


# --------------------------------------------------------------- paired_cell_bootstrap
def test_paired_bootstrap_zero_delta_for_identical_arrays():
    v = [0.5, 0.6, 0.55, 0.7, 0.4, 0.62]
    out = paired_cell_bootstrap(v, v, n_boot=200)
    assert out["delta"] == 0.0
    assert out["ci_excludes_zero"] is False


def test_paired_bootstrap_positive_ci_for_uniform_shift():
    a = [0.5] * 40
    b = [0.6] * 40
    out = paired_cell_bootstrap(a, b, n_boot=500)
    assert out["delta"] == pytest.approx(0.1, abs=1e-9)
    assert out["ci_excludes_zero"] is True
    assert out["ci95"][0] > 0.0


def test_paired_bootstrap_rejects_length_mismatch():
    with pytest.raises(ValueError):
        paired_cell_bootstrap([0.1, 0.2], [0.1, 0.2, 0.3])


# --------------------------------------------------------------- shuffled_prototype_control
def test_shuffled_control_collapses_on_decodable_signal(dump_decodable, prior_path):
    out = shuffled_prototype_control(str(dump_decodable), str(prior_path), seed=0)
    assert out["real_wprd_macro"] > 0.8
    assert out["collapse_to_prior_band"] is True
    assert abs(out["shuffled_wprd_macro"] - 0.5) < abs(out["real_wprd_macro"] - 0.5)


def test_shuffled_control_noop_on_pure_noise(dump_noise, prior_path):
    out = shuffled_prototype_control(str(dump_noise), str(prior_path), seed=0)
    # with no real signal to begin with, real and shuffled should both sit
    # near the prior-control band -- neither should look decodable.
    assert abs(out["real_wprd_macro"] - 0.5) < 0.15
    assert out["collapse_to_prior_band"] is True


# --------------------------------------------------------------- calibration_report
def test_calibration_ece_zero_when_confidence_equals_accuracy(dump_decodable, prior_path):
    r = calibration_report(str(dump_decodable), channel="adaptive", prior_path=str(prior_path))
    assert 0.0 <= r["ece"] <= 1.0
    assert r["n_rows"] > 0


# --------------------------------------------------------------- prior_correlation
def test_prior_correlation_perfect_positive_linear_signal(dump_graduated, prior_path):
    # Frequency ordered to match the graduated boost exactly (higher boost
    # position -> higher assigned "frequency"), so recall-delta and
    # log-frequency should be strongly positively correlated.
    freq = {name: (i + 1) * 1000 for i, name in enumerate(_PREDS_CYCLE)}
    out = prior_correlation(str(dump_graduated), str(prior_path), train_freq=freq, channel="adaptive")
    assert out["n_classes_matched"] == len(_PREDS_CYCLE)
    r = out["pearson_r_log_freq_vs_delta_recall"]
    assert r is not None
    assert r > 0.5


# --------------------------------------------------------------- random_text_control
def test_random_text_control_uses_injected_encoder(dump_decodable, prior_path, tmp_path):
    from tools.readout_v2_offline_analysis import _mech_with_channel

    emb_dim = 8
    fake_ckpt_path = tmp_path / "fake_ckpt_for_random_text.pt"
    torch.save({"clip": {}}, fake_ckpt_path)

    def fake_encode(strings):
        g = torch.Generator().manual_seed(999)
        return torch.randn(len(strings), emb_dim, generator=g)

    B = _mech_with_channel(str(dump_decodable), str(prior_path), "adaptive")
    strings = [f"nonsense_string_{i}" for i in range(B.n_classes)]
    out = random_text_control(str(dump_decodable), str(fake_ckpt_path), str(prior_path),
                               random_predicate_strings=strings, encode_fn=fake_encode)
    assert out["n_real_classes"] == B.n_classes
    assert "wprd_macro" in out
    assert isinstance(out["collapse_to_prior_band"], bool)
    # random, rel_feat-uncorrelated embeddings should show no real signal
    assert abs(out["wprd_macro"] - 0.5) < 0.15


def test_random_text_control_rejects_wrong_length(dump_decodable, prior_path, tmp_path):
    fake_ckpt_path = tmp_path / "fake_ckpt_for_reject_test.pt"
    torch.save({"clip": {}}, fake_ckpt_path)
    with pytest.raises(ValueError):
        random_text_control(str(dump_decodable), str(fake_ckpt_path), str(prior_path),
                             random_predicate_strings=["only", "two"],
                             encode_fn=lambda s: torch.randn(len(s), 4))


# --------------------------------------------------------------- geometry_only_control
def test_geometry_only_control_runs_and_shuffled_null_is_near_chance(dump_decodable, prior_path):
    from tools.readout_v2_offline_analysis import geometry_only_control

    out = geometry_only_control(str(dump_decodable), str(prior_path), epochs=50)
    assert "geometry_probe" in out
    assert "geometry_probe_shuffled_label_null" in out
    assert out["n_geom_features"] == 19
    real = out["geometry_probe"]["wprd_macro"]
    null = out["geometry_probe_shuffled_label_null"]["wprd_macro"]
    assert 0.0 <= real <= 1.0
    assert out["geometry_probe"]["n_cells"] > 0
    # a label-shuffled fit has no real signal to find -- must sit near chance,
    # not merely "lower than the real probe."
    assert abs(null - 0.5) < 0.2


def test_geometry_only_control_rejects_non_constant_prior():
    """geometry_only_control must refuse to report a WPRD number at all if
    gate W1 (prior constant within group) fails -- it would not be
    prior-free otherwise, and a geometry-only control that silently
    produces a compromised number is worse than one that raises."""
    from tools.readout_v2_offline_analysis import geometry_only_control

    # Build a dump whose prior_rows are NOT constant within a (subj,obj)
    # group (real prior_rows must be identical for every row sharing a
    # group) -- direct violation of gate W1.
    import tempfile
    import json as _json
    from tests.test_readout_v2_offline_analysis import _build_dump
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad_prior.pt"
        _build_dump(p, seed=99, decodable=True, n_img=4)
        dd = torch.load(p, map_location="cpu", weights_only=False)
        for i in range(len(dd["prior_rows"])):
            dd["prior_rows"][i] = torch.randn_like(dd["prior_rows"][i])  # non-constant within group, by construction
        torch.save(dd, p)
        prior_path_local = Path(td) / "prior.json"
        prior_path_local.write_text(_json.dumps({
            "predicate_vocab": dd["pred_vocab"],
            "global_log_probs": [-1.0] * len(dd["pred_vocab"]),
            "default_log_prob": -20.0,
        }))
        with pytest.raises(AssertionError):
            geometry_only_control(str(p), str(prior_path_local))


# --------------------------------------------------------------- collapse_anisotropy_check
# Two DIFFERENT phenomena, deliberately tested separately: `cos_off_diag_mean`
# is computed on raw row directions (detects "rows point the same way");
# `participation_ratio` is computed on MEAN-CENTERED rows (detects "rows
# vary along few directions AROUND their mean") -- centering removes exactly
# the common-direction signal, so a "same vector + small noise" matrix has
# HIGH cosine similarity but, once centered, its residual noise is
# essentially full rank (participation ratio near the max, not low). A
# matrix must be tested against the RIGHT construction for each metric; a
# single example claiming to trip both was wrong and is not used here.
def test_collapse_check_flags_common_direction_via_cosine():
    from tools.readout_v2_offline_analysis import collapse_anisotropy_check

    g = torch.Generator().manual_seed(0)
    base = torch.randn(1, 64, generator=g)
    collapsed = base.repeat(51, 1) + torch.randn(51, 64, generator=g) * 0.01
    out = collapse_anisotropy_check(collapsed)
    assert out["cos_off_diag_mean"] > 0.9
    assert out["high_similarity_flag"] is True


def test_collapse_check_flags_low_rank_subspace_via_participation_ratio():
    from tools.readout_v2_offline_analysis import collapse_anisotropy_check

    g = torch.Generator().manual_seed(0)
    # A genuine rank-2 matrix (no per-row independent noise): every row is
    # an exact linear combination of 2 basis vectors, so even after
    # mean-centering the effective dimensionality stays at most 2.
    low_rank = torch.randn(51, 2, generator=g) @ torch.randn(2, 64, generator=g)
    out = collapse_anisotropy_check(low_rank)
    assert out["participation_ratio"] < 3.0
    assert out["near_rank_one_flag"] is True


def test_collapse_check_passes_a_well_spread_matrix():
    from tools.readout_v2_offline_analysis import collapse_anisotropy_check

    g = torch.Generator().manual_seed(0)
    spread = torch.randn(51, 768, generator=g)  # high-dim random rows: near-orthogonal, not collapsed
    out = collapse_anisotropy_check(spread)
    assert out["cos_off_diag_mean"] < 0.3
    assert out["participation_ratio"] > 10.0
    assert out["near_rank_one_flag"] is False
    assert out["high_similarity_flag"] is False


def test_collapse_check_shape_fields():
    from tools.readout_v2_offline_analysis import collapse_anisotropy_check

    out = collapse_anisotropy_check(torch.randn(51, 768))
    assert out["n_rows"] == 51 and out["dim"] == 768
    assert out["max_possible_rank"] == 51
    assert len(out["singular_values_top5"]) == 5


# --------------------------------------------------------------- calibration_comparison
def test_calibration_comparison_shape(dump_decodable, dump_decodable_twin, prior_path):
    from tools.readout_v2_offline_analysis import calibration_comparison

    out = calibration_comparison(str(dump_decodable), str(dump_decodable_twin),
                                  channel_a="adaptive", channel_b="adaptive",
                                  prior_path=str(prior_path))
    assert "a" in out and "b" in out
    assert "delta_ece" in out and "delta_accuracy" in out
