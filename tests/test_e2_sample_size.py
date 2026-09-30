from tools.e2_sample_size import simulate_required_n


def test_sample_size_is_multiple_of_25_and_reproducible():
    a = simulate_required_n(.6, .05, .2, replicates=2000)
    b = simulate_required_n(.6, .05, .2, replicates=2000)
    assert a == b
    assert a["required_n"] % 25 == 0


def test_incompatible_paired_parameters_are_rejected():
    result = simulate_required_n(.8, .05, .4, replicates=100)
    assert result["status"] == "invalid_parameters"
