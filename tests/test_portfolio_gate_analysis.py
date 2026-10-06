"""Small deterministic checks for the CPU-only portfolio-gate helpers."""
import numpy as np

from tools.portfolio_gate_analysis import image_cluster_bootstrap, spearman


def test_spearman_is_monotonic_and_tie_aware():
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert np.isclose(spearman([1, 1, 2, 3], [2, 2, 4, 6]), 1.0)


def test_image_cluster_bootstrap_is_reproducible_and_keeps_image_clusters():
    delta = np.asarray([-1.0, -3.0, 2.0, 4.0])
    image_ids = np.asarray(["a", "a", "b", "b"])
    first = image_cluster_bootstrap(delta, image_ids, seed=17, reps=100)
    second = image_cluster_bootstrap(delta, image_ids, seed=17, reps=100)
    assert first == second
    assert first["unit"] == "image_id"
    assert first["estimate"] == 0.5
    assert len(first["replicate_values"]) == 100
