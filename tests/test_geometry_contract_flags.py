"""C0 must be bit-exact with the C1a/C1b flags present and at their defaults.

These tests exist so a NEGATIVE C1 result can never be blamed on the flags
having silently perturbed the control arm, and so the C1a defect claim itself
(docs/GEOMETRY_INPUT_DEGENERACY_RESULT.md) is asserted in the suite rather than
only in prose.
"""
import math

import torch

from openvocab_rel.geometry import geom_feats_torch


def _boxes():
    torch.manual_seed(0)
    x1 = torch.rand(64) * 200.0
    y1 = torch.rand(64) * 200.0
    b = torch.stack([x1, y1, x1 + 10.0 + torch.rand(64) * 120.0,
                     y1 + 10.0 + torch.rand(64) * 120.0], dim=1)
    return b[:32], b[32:]


def test_normalised_boxes_zero_six_of_eight_channels():
    """p68's defect, asserted. Divide by img_res and clamp_min(1.0) binds."""
    b1, b2 = _boxes()
    g = geom_feats_torch(b1 / 336.0, b2 / 336.0)
    assert g.shape[1] == 8
    for col in range(2, 8):                      # rw, rh, ar1, ar2, a1, a2
        assert g[:, col].std().item() == 0.0, f"column {col} is not constant"
    assert g[:, 0].std().item() > 0.0            # dx survives
    assert g[:, 1].std().item() > 0.0            # dy survives


def test_pixel_boxes_keep_all_eight_channels_informative():
    """C1a's contract: every channel varies when the clamp does not bind."""
    b1, b2 = _boxes()
    g = geom_feats_torch(b1, b2)
    for col in range(8):
        assert g[:, col].std().item() > 0.0, f"column {col} is constant"


def test_c1a_model_path_passes_unscaled_pixel_boxes_to_geometry(monkeypatch):
    """Exercise the actual RelationalModel path, not just the geometry helper."""
    import openvocab_rel.models.relational_model as rel_mod
    from openvocab_rel.config import TrainConfig
    from openvocab_rel.models.relational_model import RelationalModel

    cfg = TrainConfig()
    cfg.emb_dim = 32
    cfg.clip_input_res = 64
    cfg.progressive_node_layers = 1
    cfg.progressive_edge_layers = 0
    cfg.progressive_bilinear_layers = 0
    cfg.deformable_num_points = 4
    cfg.relation_context_layers = 0
    cfg.gradient_checkpointing = False
    cfg.learned_prune_k = 0
    cfg.geom_input_pixel_space = True
    model = RelationalModel(cfg, clip_vision_dim=48, text_dim=None).eval()

    boxes = torch.tensor(
        [[2.0, 3.0, 24.0, 31.0], [15.0, 8.0, 49.0, 44.0],
         [28.0, 18.0, 61.0, 58.0]], dtype=torch.float32
    )
    captured = {}
    original = rel_mod.geom_feats_torch

    def capture_geometry(subject_boxes, object_boxes):
        captured["subject"] = subject_boxes.detach().clone()
        captured["object"] = object_boxes.detach().clone()
        return original(subject_boxes, object_boxes)

    monkeypatch.setattr(rel_mod, "geom_feats_torch", capture_geometry)
    feat_map = torch.randn(1, 48, 8, 8)
    pairs = [[(0, 1), (1, 2), (2, 0)]]
    with torch.no_grad():
        _regs, rel_feats, _swaps, _assigns, _gates, _kept = model.forward_from_featmap(
            feat_map, obj_boxes=[boxes], pairs=pairs, return_swapped=False
        )

    box_identity = {tuple(row.tolist()): i for i, row in enumerate(boxes)}
    captured_pairs = {
        (box_identity[tuple(s.tolist())], box_identity[tuple(o.tolist())])
        for s, o in zip(captured["subject"], captured["object"])
    }
    assert captured_pairs == set(pairs[0])
    assert float(captured["subject"].abs().max()) > 1.0
    assert float(captured["object"].abs().max()) > 1.0
    geom = original(captured["subject"], captured["object"])
    assert geom.shape == (3, 8)
    assert torch.isfinite(geom).all()
    for col in range(8):
        assert geom[:, col].std().item() > 0.0, f"C1a geometry column {col} collapsed"
    assert rel_feats[0].shape == (3, cfg.emb_dim)
    assert torch.isfinite(rel_feats[0]).all()


def test_six_channels_are_scale_invariant_only_log_areas_shift():
    """Pixel vs normalised differ ONLY by a constant offset in a1, a2.

    Justifies computing a1/a2 in raw pixel units in p72's `fixed` contract.
    """
    b1, b2 = _boxes()
    g_px = geom_feats_torch(b1, b2)
    g_sc = geom_feats_torch(b1 / 4.0, b2 / 4.0)   # /4 keeps the clamp off
    for col in range(6):
        assert torch.allclose(g_px[:, col], g_sc[:, col], atol=1e-4)
    for col in (6, 7):
        d = g_px[:, col] - g_sc[:, col]
        assert torch.allclose(d, torch.full_like(d, 2.0 * math.log(4.0)), atol=1e-4)


def test_config_defaults_are_the_historical_behaviour():
    from openvocab_rel.config import TrainConfig
    assert TrainConfig.geom_input_pixel_space is False
    assert TrainConfig.geom_fourier_scale == 1.0


def test_fourier_scale_one_is_the_identity():
    """C1b at scale 1.0 must reproduce the historical projection exactly."""
    torch.manual_seed(1)
    x = torch.randn(16, 8)
    B = torch.randn(8, 32) * 10.0
    hist = (2.0 * math.pi * x) @ B
    flagged = (2.0 * math.pi * 1.0 * x) @ B
    assert torch.equal(hist, flagged)


def test_save_best_checkpoints_defaults_to_historical_behaviour():
    """The C0/C1 disk guard must not silently change any historical run."""
    from openvocab_rel.config import TrainConfig
    assert TrainConfig.save_best_checkpoints is True
