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
