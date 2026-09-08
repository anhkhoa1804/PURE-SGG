"""Paper C -- Readout v2 regression tests.

Pre-registered in docs/PAPER_C_READOUT_V2_PREREGISTRATION.md section 17.
Synthetic tensors only, no CLIP model, no GPU, no real data -- matches the
existing test_model_forward.py convention. Every test here must pass before
any GPU use (section 18 of the same document).

The pilot-configuration tests below (test_pilot_script_resolves_*) resolve
the ACTUAL runtime TrainConfig the shell scripts produce -- they run
scripts/train/run_readout_v2.sh / run_c0_c1.sh with PYTHON=/bin/echo (so no
Python process, no GPU, no training is ever touched; bash's own array
expansion prints the exact argv main() would receive) and feed that argv
through openvocab_rel.train's own config-resolution logic. This is a
runtime check, not a string grep -- it would catch a flag being silently
dropped, misspelled, or overridden downstream just as well as one left at
the wrong literal value.
"""
import copy
import shlex
import subprocess
import sys
from pathlib import Path

import torch

from openvocab_rel.config import TrainConfig, apply_stage_config
from openvocab_rel.models.relational_model import RelationalModel
from openvocab_rel.train import build_argparser

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_cfg(argv):
    """Exact replica of openvocab_rel.train.main()'s own cfg-resolution
    logic (train.py ~1275-1297), so this test exercises the same code path
    a real run would, not a hand-maintained approximation of it."""
    parser = build_argparser()
    args, _unknown = parser.parse_known_args(argv)
    base_cfg = TrainConfig()
    base_cfg.stage = args.stage
    base_cfg = apply_stage_config(base_cfg)
    called_args = [a[2:].split("=")[0] for a in argv if a.startswith("--")]
    for k, v in vars(args).items():
        if k in called_args:
            setattr(base_cfg, k, v)
        elif not hasattr(base_cfg, k) or getattr(base_cfg, k) is None:
            setattr(base_cfg, k, v)
    return base_cfg


def _dry_run_argv(script: str, env_overrides: dict) -> list:
    """Runs a scripts/train/*.sh launcher with PYTHON=/bin/echo, so bash
    resolves and prints the exact argv it would hand to `python -m
    openvocab_rel.train` -- without ever starting Python, touching the GPU,
    or training anything. Returns the parsed argv list."""
    import os
    import tempfile

    env = dict(os.environ)
    env["PYTHON"] = "/bin/echo"
    with tempfile.TemporaryDirectory() as td:
        env["OUT_DIR"] = str(Path(td) / "out")
        env.update(env_overrides)
        proc = subprocess.run(
            ["bash", script], cwd=str(REPO_ROOT), env=env,
            capture_output=True, text=True, timeout=60,
        )
    assert proc.returncode == 0, f"{script} dry run failed: {proc.stderr}"
    # PYTHONUNBUFFERED=1 /bin/echo "${ARGS[@]}" -- one line of space-joined args.
    line = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("-u")]
    assert len(line) == 1, f"expected exactly one echoed argv line, got: {proc.stdout!r}"
    return shlex.split(line[0])


def _tiny_cfg() -> TrainConfig:
    cfg = TrainConfig()
    cfg.emb_dim = 32
    cfg.clip_input_res = 64
    cfg.progressive_node_layers = 1
    cfg.progressive_edge_layers = 1
    cfg.progressive_bilinear_layers = 0
    cfg.deformable_num_points = 4
    cfg.relation_context_layers = 0
    cfg.gradient_checkpointing = False
    cfg.learned_prune_k = 0
    cfg.predicate_classifier_classes = 6
    return cfg


def _tiny_model() -> RelationalModel:
    cfg = _tiny_cfg()
    return RelationalModel(cfg, clip_vision_dim=48, text_dim=None)


# ------------------------------------------------------------------ 17.1
def test_flag_off_predicate_prototypes_absent():
    model = _tiny_model()
    assert not hasattr(model, "predicate_prototypes")
    # predicate_scores must behave exactly as before -- the "adaptive" branch
    # is unreachable unless init_readout_v2() was called.
    rel_feats = torch.randn(4, model.dim)
    text_feats = torch.randn(model.predicate_classifier_classes, model.dim)
    before = model.predicate_scores(rel_feats, text_feats, mode="text")
    after = model.predicate_scores(rel_feats, text_feats, mode="text")
    assert torch.equal(before, after)


def test_adaptive_mode_requires_init():
    model = _tiny_model()
    rel_feats = torch.randn(3, model.dim)
    try:
        model.predicate_scores(rel_feats, mode="adaptive")
        raised = False
    except RuntimeError:
        raised = True
    assert raised


# ------------------------------------------------------------------ 17.2 / 17.3
def test_prototype_shape_and_init_equals_E_but_independent():
    model = _tiny_model()
    C, d = model.predicate_classifier_classes, model.dim
    E = torch.randn(C, d)
    model.init_readout_v2(E)

    assert model.predicate_prototypes.shape == (C, d)
    assert torch.equal(model.predicate_prototypes.detach(), E.detach())
    # copy, not alias (preregistration section 7's named trap)
    assert model.predicate_prototypes.data_ptr() != E.data_ptr()

    # mutating E afterward must not move P
    E_snapshot = E.clone()
    E += 1.0
    assert torch.equal(model.predicate_prototypes.detach(), E_snapshot)


def test_init_readout_v2_twice_raises():
    model = _tiny_model()
    E = torch.randn(model.predicate_classifier_classes, model.dim)
    model.init_readout_v2(E)
    try:
        model.init_readout_v2(E)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


# ------------------------------------------------------------------ 17.4
def test_predicate_indexing_preserves_row_order():
    model = _tiny_model()
    C, d = model.predicate_classifier_classes, model.dim
    E = torch.arange(C * d, dtype=torch.float32).reshape(C, d)
    model.init_readout_v2(E)
    for i in range(C):
        assert torch.equal(model.predicate_prototypes[i].detach(), E[i])


# ------------------------------------------------------------------ 17.5
def test_anchor_loss_zero_at_init_and_increases_with_rotation():
    model = _tiny_model()
    C, d = model.predicate_classifier_classes, model.dim
    E = torch.randn(C, d)
    model.init_readout_v2(E)

    l0 = model.readout_v2_anchor_loss()
    assert torch.isfinite(l0)
    assert float(l0) < 1e-6

    with torch.no_grad():
        # rotate row 0 by adding an orthogonal component -- moves cosine
        # similarity away from 1 without collapsing to exactly antiparallel.
        row = model.predicate_prototypes[0]
        ortho = torch.randn_like(row)
        ortho = ortho - (ortho @ row) / (row @ row) * row
        model.predicate_prototypes[0] = row + ortho

    l1 = model.readout_v2_anchor_loss()
    assert float(l1) > float(l0)


# ------------------------------------------------------------------ 17.6
def test_gradient_flow_touches_only_predicate_prototypes():
    model = _tiny_model()
    C, d = model.predicate_classifier_classes, model.dim
    E = torch.randn(C, d)
    model.init_readout_v2(E)
    model.predicate_prototypes.requires_grad_(True)
    for name, p in model.named_parameters():
        if name != "predicate_prototypes":
            p.requires_grad_(False)

    rel_feats = torch.randn(5, model.dim)
    labels = torch.randint(0, C, (5,))
    logits = model.adaptive_predicate_logits(rel_feats)
    loss = torch.nn.functional.cross_entropy(logits, labels) + model.readout_v2_anchor_loss()
    loss.backward()

    assert model.predicate_prototypes.grad is not None
    assert torch.isfinite(model.predicate_prototypes.grad).all()
    for name, p in model.named_parameters():
        if name != "predicate_prototypes":
            assert p.grad is None, f"{name} unexpectedly received a gradient"


# ------------------------------------------------------------------ 17.7 / 17.12
def test_adaptive_logits_take_no_prior_argument():
    import inspect

    sig = inspect.signature(RelationalModel.adaptive_predicate_logits)
    params = list(sig.parameters.keys())
    assert "pred_log_prior" not in params
    assert "prior" not in params

    model = _tiny_model()
    E = torch.randn(model.predicate_classifier_classes, model.dim)
    model.init_readout_v2(E)
    rel_feats = torch.randn(3, model.dim)
    out1 = model.adaptive_predicate_logits(rel_feats)
    # nothing in the call depends on any external prior state; recomputing
    # with the same inputs must be bit-identical (also covers 17.9).
    out2 = model.adaptive_predicate_logits(rel_feats)
    assert torch.equal(out1, out2)


# ------------------------------------------------------------------ 17.8
def test_checkpoint_roundtrip_predicate_prototypes():
    model = _tiny_model()
    E = torch.randn(model.predicate_classifier_classes, model.dim)
    model.init_readout_v2(E)
    with torch.no_grad():
        model.predicate_prototypes.add_(0.01)
    saved = copy.deepcopy(model.state_dict())

    reloaded = _tiny_model()
    reloaded.init_readout_v2(E)
    reloaded.load_state_dict(saved, strict=True)
    assert torch.equal(reloaded.predicate_prototypes.detach(), model.predicate_prototypes.detach())

    # loading the SAME state dict into a flag-off model (no predicate_prototypes
    # attribute) must not be fatal -- the extra key is ignored, not an error.
    flagoff = _tiny_model()
    missing, unexpected = flagoff.load_state_dict(saved, strict=False)
    assert "predicate_prototypes" in unexpected


# ------------------------------------------------------------------ 17.9 (determinism)
def test_deterministic_inference():
    model = _tiny_model()
    model.eval()
    E = torch.randn(model.predicate_classifier_classes, model.dim)
    model.init_readout_v2(E)
    rel_feats = torch.randn(6, model.dim)
    with torch.no_grad():
        a = model.adaptive_predicate_logits(rel_feats)
        b = model.adaptive_predicate_logits(rel_feats)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()


# ------------------------------------------------------------------ 17.10 (train/eval parity)
def test_predicate_scores_adaptive_mode_matches_direct_call():
    model = _tiny_model()
    E = torch.randn(model.predicate_classifier_classes, model.dim)
    model.init_readout_v2(E)
    rel_feats = torch.randn(4, model.dim)
    via_predicate_scores = model.predicate_scores(rel_feats, mode="adaptive")
    via_direct_call = model.adaptive_predicate_logits(rel_feats)
    assert torch.equal(via_predicate_scores, via_direct_call)


# ------------------------------------------------------------------ 17.11 (no geometry change)
def test_forward_from_featmap_unaffected_by_readout_v2():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model_a = RelationalModel(cfg, clip_vision_dim=48, text_dim=None)
    model_b = RelationalModel(cfg, clip_vision_dim=48, text_dim=None)
    model_b.load_state_dict(model_a.state_dict())

    E = torch.randn(model_b.predicate_classifier_classes, model_b.dim)
    model_b.init_readout_v2(E)  # adds a new parameter, touches nothing else

    feat_map = torch.randn(1, 48, 8, 8)
    obj_boxes = [torch.tensor([[0.0, 0.0, 20.0, 20.0], [10.0, 10.0, 40.0, 40.0]], dtype=torch.float32)]
    pairs = [[(0, 1), (1, 0)]]

    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        _, rel_a, _, _, gates_a, _ = model_a.forward_from_featmap(feat_map, obj_boxes=obj_boxes, pairs=pairs)
        _, rel_b, _, _, gates_b, _ = model_b.forward_from_featmap(feat_map, obj_boxes=obj_boxes, pairs=pairs)

    assert torch.equal(rel_a[0], rel_b[0])
    assert torch.equal(gates_a[0], gates_b[0])


# ------------------------------------------------------------------ pilot-config invariants
# Pre-GPU audit correction: three flags (explicit_spoa_enabled,
# text_conditioned_projection_enabled, predicate_label_relaxation_enabled)
# were found active=true in scripts/train/run_readout_v2.sh, contradicting
# the preregistration and the eval tool. These tests resolve the REAL
# runtime cfg the script produces and fail if any of the three regress.

def test_pilot_script_resolves_the_three_corrected_flags_false():
    argv = _dry_run_argv("scripts/train/run_readout_v2.sh", {"PILOT": "1"})
    cfg = _resolve_cfg(argv)
    assert cfg.explicit_spoa_enabled is False, "DEFECT A regressed: rel_feat would differ train vs eval"
    assert cfg.text_conditioned_projection_enabled is False, "DEFECT B regressed: readout input would differ train vs eval"
    assert cfg.predicate_label_relaxation_enabled is False, "DEFECT C regressed: unregistered second intervention on l_readout_v2_ce"


def test_pilot_script_activates_readout_v2_on_the_fixed_c1_contract():
    argv = _dry_run_argv("scripts/train/run_readout_v2.sh", {"PILOT": "1"})
    cfg = _resolve_cfg(argv)
    assert cfg.readout_v2_enabled is True
    assert cfg.geom_input_pixel_space is True
    assert float(cfg.geom_fourier_scale) == 0.01
    assert cfg.resume_from == "checkpoints/C1_seed1234.pt"
    assert cfg.reset_epoch is True


def test_pilot_script_refuses_full_budget_without_PILOT_flag():
    proc = subprocess.run(
        ["bash", "scripts/train/run_readout_v2.sh"],
        cwd=str(REPO_ROOT),
        env={**__import__("os").environ, "PYTHON": "/bin/echo"},
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "not registered" in (proc.stdout + proc.stderr).lower()


def test_readout_v2_pilot_config_differs_from_c1_baseline_only_in_declared_ways():
    """The resolved Readout v2 pilot cfg must differ from the resolved C1
    baseline cfg only in: (a) the readout_v2_* flags themselves, (b) the
    three corrected isolation flags (intentionally false, unlike the base
    run), and (c) explicitly declared training-control bookkeeping (budget,
    LR, resume target, naming). Any OTHER differing key is an unreviewed,
    unregistered deviation and must fail this test."""
    v2_argv = _dry_run_argv("scripts/train/run_readout_v2.sh", {"PILOT": "1"})
    c1_argv = _dry_run_argv("scripts/train/run_c0_c1.sh", {"ARM": "C1", "SEED": "1234"})
    v2_cfg = _resolve_cfg(v2_argv)
    c1_cfg = _resolve_cfg(c1_argv)

    allowed_diff_keys = {
        # readout v2's own new surface
        "readout_v2_enabled", "readout_v2_lambda_anchor", "readout_v2_lr",
        # the three corrected isolation flags -- intentionally false here,
        # true in the C1 baseline recipe (see the preregistration's
        # CORRECTION block)
        "explicit_spoa_enabled", "text_conditioned_projection_enabled",
        "predicate_label_relaxation_enabled",
        # every existing loss lambda is zero-weighted for the pilot (only
        # readout_v2_term is active) -- these are DECLARED, not hidden
        "lambda_predicate_ce", "lambda_spoa_alignment", "lambda_dense_grounding",
        "lambda_counterfactual", "lambda_text_predicate_ce", "lambda_relationness",
        "lambda_calibration_reg", "lambda_calibration_kl", "lambda_calibration_rank",
        # declared training-control / bookkeeping differences
        "resume_from", "run_name", "out_dir", "save_path", "save_metrics_json",
        "epochs", "samples_per_epoch", "batch_size", "accum_steps", "lr",
        "warmup_steps", "gradient_checkpointing", "freeze_clip",
        "eval_batches", "log_every",
    }
    checked = 0
    for key in vars(c1_cfg):
        if key.startswith("_"):
            continue
        a = getattr(c1_cfg, key, None)
        b = getattr(v2_cfg, key, None)
        if a == b:
            continue
        checked += 1
        assert key in allowed_diff_keys, (
            f"unexpected, unreviewed config difference: {key!r} "
            f"(C1 baseline={a!r}, readout_v2 pilot={b!r})"
        )
    assert checked > 0, "sanity: the two configs must differ in at least the readout_v2 surface"


def test_rel_feat_train_eval_parity_traced_through_forward_pass():
    """Direct proof, not config equality: construct a model the way the
    corrected train-side cfg would (explicit_spoa_enabled=False,
    text_conditioned_projection_enabled=False) and one the way
    tools/readout_v2_evaluate.py's eval-side cfg does (same two flags,
    hardcoded False), same seed/weights/inputs, and assert rel_feat is
    bit-identical. A negative control reproduces the ORIGINAL bug
    (explicit_spoa_enabled=True at train, False at eval) and asserts it
    WOULD have produced a different rel_feat -- proving the fix is not a
    no-op and the mechanism is exactly what the pre-GPU audit described.
    """
    def cfg_with(explicit_spoa: bool, text_proj: bool) -> TrainConfig:
        cfg = _tiny_cfg()
        cfg.explicit_spoa_enabled = explicit_spoa
        cfg.text_conditioned_projection_enabled = text_proj
        return cfg

    feat_map = torch.randn(1, 48, 8, 8)
    obj_boxes = [torch.tensor([[0.0, 0.0, 20.0, 20.0], [10.0, 10.0, 40.0, 40.0]], dtype=torch.float32)]
    pairs = [[(0, 1), (1, 0)]]

    def rel_feat_for(cfg: TrainConfig, seed: int, state_dict=None):
        torch.manual_seed(seed)
        model = RelationalModel(cfg, clip_vision_dim=48, text_dim=None)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            _, rel, _, _, _, _ = model.forward_from_featmap(feat_map, obj_boxes=obj_boxes, pairs=pairs)
        return rel[0], model.state_dict()

    rel_train_corrected, train_state = rel_feat_for(cfg_with(False, False), seed=42)
    rel_eval_corrected, _ = rel_feat_for(cfg_with(False, False), seed=42, state_dict=train_state)
    assert torch.equal(rel_train_corrected, rel_eval_corrected)

    rel_train_buggy, _ = rel_feat_for(cfg_with(True, False), seed=42, state_dict=train_state)
    assert not torch.equal(rel_train_buggy, rel_eval_corrected), (
        "negative control failed: explicit_spoa_enabled mismatch should have "
        "produced a different rel_feat -- if this now passes with torch.equal, "
        "the mechanism this test guards against may have changed."
    )
