"""Paper C -- Readout v2 treatment-fidelity tests (strategy S1).

Registered in docs/PAPER_C_R2_TREATMENT_FIDELITY_AMENDMENT_2026-09-10.md
section H.3 BEFORE any corrected run executes. Synthetic tensors only: no
CLIP model, no checkpoint on disk, no CUDA, no real data -- same convention
as tests/test_readout_v2.py.

The governing principle under test (amendment section F.3): treatment
fidelity is established from PARAMETER PROVENANCE, never from whether the
resulting logits happen to differ from the baseline.
"""
import copy

import pytest
import torch

from openvocab_rel.config import TrainConfig, apply_stage_config
from openvocab_rel.models.relational_model import RelationalModel
from openvocab_rel.train import build_argparser
from openvocab_rel import readout_v2_provenance as prov


# --------------------------------------------------------------- fixtures
def _tiny_model(classes: int = 6, dim: int = 8) -> RelationalModel:
    cfg = apply_stage_config(TrainConfig())
    cfg = copy.deepcopy(cfg)
    cfg.emb_dim = dim
    cfg.predicate_classifier_classes = classes
    cfg.readout_v2_enabled = True
    return RelationalModel(cfg, clip_vision_dim=48, text_dim=None)


def _E(classes: int = 6, dim: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.nn.functional.normalize(torch.randn(classes, dim), dim=-1)


def _trained_P(E: torch.Tensor) -> torch.Tensor:
    """A P that has moved from E on every row, boundedly -- the shape of the
    real pilot checkpoint's prototypes (1-cos mean 3.2e-02, max 1.6e-01)."""
    torch.manual_seed(1)
    return (E + 0.05 * torch.randn_like(E)).contiguous()


# ------------------------------------------------------------------ Test 1
def test_checkpoint_mode_installs_non_E_prototypes_exactly():
    """A synthetic checkpoint whose P != E must round-trip in exactly."""
    model, E = _tiny_model(), _E()
    P = _trained_P(E)
    assert not torch.equal(P, E)

    state = {prov.PROTOTYPE_KEY: P.clone()}
    ckpt_p, ckpt_hash = prov.extract_checkpoint_prototypes(
        state, expected_sha256=None, expected_shape=None)

    model.init_readout_v2(E)
    assert torch.equal(model.predicate_prototypes.detach(), E)   # P := E first

    record = prov.restore_checkpoint_prototypes(
        model, ckpt_p, checkpoint_hash=ckpt_hash, anchor_E=E)

    assert torch.equal(model.predicate_prototypes.detach(), P)
    assert record["prototype_source"] == prov.PROTOTYPE_SOURCE_LABEL_CHECKPOINT
    assert record["installed_P_matches_checkpoint"] is True
    assert record["installed_P_equals_E"] is False
    assert record["P_vs_E"]["rows_moved_gt_1e-6"] == E.shape[0]


def test_restoration_preserves_parameter_identity_and_optimizer_reference():
    """copy_ under no_grad, not .data rebinding: the optimizer's reference to
    the Parameter object must survive restoration (amendment F.2 step 5)."""
    model, E = _tiny_model(), _E()
    P = _trained_P(E)
    model.init_readout_v2(E)
    param_obj = model.predicate_prototypes
    optim = torch.optim.AdamW([param_obj], lr=1e-3)

    ckpt_p, ckpt_hash = prov.extract_checkpoint_prototypes(
        {prov.PROTOTYPE_KEY: P.clone()}, expected_sha256=None, expected_shape=None)
    prov.restore_checkpoint_prototypes(
        model, ckpt_p, checkpoint_hash=ckpt_hash, anchor_E=E)

    assert model.predicate_prototypes is param_obj
    assert optim.param_groups[0]["params"][0] is model.predicate_prototypes
    assert model.predicate_prototypes.requires_grad is True
    assert isinstance(model.predicate_prototypes, torch.nn.Parameter)


# ------------------------------------------------------------------ Test 2
def test_hash_mismatch_fails_closed():
    """A wrong registered hash must RAISE, not warn-and-continue."""
    E = _E()
    P = _trained_P(E)
    with pytest.raises(prov.ReadoutV2ProvenanceError, match="sha256"):
        prov.extract_checkpoint_prototypes(
            {prov.PROTOTYPE_KEY: P},
            expected_sha256="0" * 64,
            expected_shape=None)


def test_missing_prototype_key_fails_closed():
    with pytest.raises(prov.ReadoutV2ProvenanceError, match="no 'predicate_prototypes'"):
        prov.extract_checkpoint_prototypes({"other.weight": torch.zeros(2, 2)},
                                           expected_sha256=None, expected_shape=None)


def test_absent_checkpoint_fails_closed():
    with pytest.raises(prov.ReadoutV2ProvenanceError, match="requires a resumed checkpoint"):
        prov.extract_checkpoint_prototypes(None, expected_sha256=None, expected_shape=None)


def test_non_finite_prototypes_fail_closed():
    E = _E()
    bad = _trained_P(E)
    bad[0, 0] = float("nan")
    with pytest.raises(prov.ReadoutV2ProvenanceError, match="non-finite"):
        prov.extract_checkpoint_prototypes({prov.PROTOTYPE_KEY: bad},
                                           expected_sha256=None, expected_shape=None)


def test_wrong_shape_and_dtype_fail_closed():
    with pytest.raises(prov.ReadoutV2ProvenanceError, match="shape"):
        prov.extract_checkpoint_prototypes(
            {prov.PROTOTYPE_KEY: torch.zeros(7, 8)},
            expected_sha256=None, expected_shape=(6, 8))
    with pytest.raises(prov.ReadoutV2ProvenanceError, match="dtype"):
        prov.extract_checkpoint_prototypes(
            {prov.PROTOTYPE_KEY: torch.zeros(6, 8, dtype=torch.float64)},
            expected_sha256=None, expected_shape=(6, 8),
            expected_dtype=torch.float32)


def test_p_equal_to_E_fails_closed():
    """A checkpoint whose trained P never moved cannot instantiate R2c."""
    model, E = _tiny_model(), _E()
    model.init_readout_v2(E)
    ckpt_p, ckpt_hash = prov.extract_checkpoint_prototypes(
        {prov.PROTOTYPE_KEY: E.clone()}, expected_sha256=None, expected_shape=None)
    with pytest.raises(prov.ReadoutV2ProvenanceError, match="exactly equal to E"):
        prov.restore_checkpoint_prototypes(
            model, ckpt_p, checkpoint_hash=ckpt_hash, anchor_E=E)


def test_restore_before_init_fails_closed():
    model, E = _tiny_model(), _E()
    ckpt_p, ckpt_hash = prov.extract_checkpoint_prototypes(
        {prov.PROTOTYPE_KEY: _trained_P(E)}, expected_sha256=None, expected_shape=None)
    with pytest.raises(prov.ReadoutV2ProvenanceError, match="init_readout_v2"):
        prov.restore_checkpoint_prototypes(
            model, ckpt_p, checkpoint_hash=ckpt_hash, anchor_E=E)


# ------------------------------------------------------------------ Test 3
def test_installed_equals_checkpoint_and_anchor_still_equals_E():
    """After restoration: installed_P == checkpoint_P AND _readout_v2_anchor == E.

    The anchor must stay E because the preregistration's L_anchor (section 6)
    is defined against E; letting it become P would silently redefine the
    registered loss.
    """
    model, E = _tiny_model(), _E()
    P = _trained_P(E)
    model.init_readout_v2(E)
    ckpt_p, ckpt_hash = prov.extract_checkpoint_prototypes(
        {prov.PROTOTYPE_KEY: P.clone()}, expected_sha256=None, expected_shape=None)
    record = prov.restore_checkpoint_prototypes(
        model, ckpt_p, checkpoint_hash=ckpt_hash, anchor_E=E)

    assert torch.equal(model.predicate_prototypes.detach(), P)
    assert prov.tensor_hash(model.predicate_prototypes)["sha256"] == ckpt_hash["sha256"]
    assert record["installed_P_hash"]["sha256"] == record["checkpoint_P_hash"]["sha256"]

    assert torch.equal(model._readout_v2_anchor, E)
    assert record["anchor_equals_E"] is True
    # and the anchor loss is still measured against E, so it is now non-zero
    assert float(model.readout_v2_anchor_loss()) > 0.0


# ------------------------------------------------------------------ Test 4
def test_default_reinit_E_preserves_todays_behaviour_bitwise():
    """Default flag value must be reinit_E, and that path must be P := E."""
    assert TrainConfig.readout_v2_prototype_source == prov.PROTOTYPE_SOURCE_REINIT_E
    args = build_argparser().parse_args(["--stage", "3"])
    assert args.readout_v2_prototype_source == prov.PROTOTYPE_SOURCE_REINIT_E
    assert args.readout_v2_expected_p_sha256 == ""

    model, E = _tiny_model(), _E()
    model.init_readout_v2(E)
    # No restoration is invoked under reinit_E: P stays bit-identical to E,
    # a distinct tensor (the preregistration section 7 copy-not-alias trap).
    assert torch.equal(model.predicate_prototypes.detach(), E)
    assert model.predicate_prototypes.data_ptr() != E.data_ptr()
    assert float(model.readout_v2_anchor_loss()) == pytest.approx(0.0, abs=1e-7)


def test_argparser_rejects_unknown_prototype_source():
    parser = build_argparser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--stage", "3", "--readout_v2_prototype_source", "sneaky"])


# ------------------------------------------------------------------ Test 5
def test_unexpected_skipped_key_fails_closed():
    audit = prov.audit_skipped_keys(["predicate_prototypes", "decoder.rel_seed.weight"])
    assert audit["skipped_keys_unexpected"] == ["decoder.rel_seed.weight"]
    with pytest.raises(prov.ReadoutV2ProvenanceError, match="beyond the expected"):
        prov.assert_no_unexpected_skips(audit)


# ------------------------------------------------------------------ Test 6
def test_expected_prototype_skip_is_accepted_and_disclosed():
    """The predicate_prototypes skip is EXPECTED under S1 (amendment F.3.4):
    accepted, and surfaced in the record rather than filtered out."""
    audit = prov.audit_skipped_keys(["predicate_prototypes"])
    assert audit["skipped_keys_by_name"] == ["predicate_prototypes"]
    assert audit["skipped_keys_expected"] == ["predicate_prototypes"]
    assert audit["skipped_keys_unexpected"] == []
    assert audit["prototype_skip_was_expected"] is True
    prov.assert_no_unexpected_skips(audit)          # must NOT raise

    # ... and an empty skip list is also fine (nothing to restore from a
    # checkpoint that never carried the key is caught separately, by
    # extract_checkpoint_prototypes).
    empty = prov.audit_skipped_keys([])
    assert empty["prototype_skip_was_expected"] is False
    prov.assert_no_unexpected_skips(empty)


# ------------------------------------------------------------------ Test 7
def test_adaptive_equals_text_is_not_a_treatment_fidelity_failure():
    """Endpoint equality must NOT invalidate a provenance-certified run.

    Construct a genuinely restored trained P that nonetheless yields
    adaptive_logits bit-identical to text_logits (achieved here by scoring
    against P as the text argument too). Provenance must still certify the
    run: nothing in the provenance path consults the logits.
    """
    model, E = _tiny_model(), _E()
    P = _trained_P(E)
    model.init_readout_v2(E)
    ckpt_p, ckpt_hash = prov.extract_checkpoint_prototypes(
        {prov.PROTOTYPE_KEY: P.clone()}, expected_sha256=None, expected_shape=None)
    record = prov.restore_checkpoint_prototypes(
        model, ckpt_p, checkpoint_hash=ckpt_hash, anchor_E=E)

    rel = torch.randn(5, model.dim)
    adaptive = model.adaptive_predicate_logits(rel)
    text_against_P = model.text_predicate_logits(rel, model.predicate_prototypes)
    assert torch.equal(adaptive, text_against_P)     # endpoint coincidence

    # Provenance is unaffected by that coincidence -- all five section F.3.1
    # assertions still hold, so the run stays VALID and the null (if any) is a
    # scientific result rather than a provenance failure.
    assert record["installed_P_matches_checkpoint"] is True
    assert record["installed_P_equals_E"] is False
    assert record["prototype_source"] == prov.PROTOTYPE_SOURCE_LABEL_CHECKPOINT
    # No provenance function in the module takes logits as an argument at all.
    import inspect
    for fn in (prov.extract_checkpoint_prototypes, prov.restore_checkpoint_prototypes,
               prov.audit_skipped_keys, prov.assert_no_unexpected_skips):
        params = " ".join(inspect.signature(fn).parameters)
        assert "logit" not in params


# ------------------------------------------------- registered constants lock
def test_registered_constants_match_the_amendment():
    assert prov.REGISTERED_CHECKPOINT_P_SHA256 == (
        "294d01cccb7bc78233d52de2c5b532e261b1782f92a77cf72f9b768954943e5c")
    assert prov.EXPECTED_P_SHAPE == (51, 768)
    assert prov.EXPECTED_P_DTYPE is torch.float32
    assert prov.EXPECTED_SKIPPED_KEYS == ("predicate_prototypes",)
    assert prov.PROTOTYPE_SOURCES == ("reinit_E", "checkpoint")


def test_tensor_hash_matches_gd_runner_scheme():
    """Same digest scheme as tools/run_final_batch_gd_cuda.py::_tensor_hash,
    so hashes recorded there and here are directly comparable."""
    import hashlib
    import json
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expect = hashlib.sha256()
    expect.update(str(t.dtype).encode())
    expect.update(json.dumps(list(t.shape)).encode())
    expect.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    assert prov.tensor_hash(t)["sha256"] == expect.hexdigest()
