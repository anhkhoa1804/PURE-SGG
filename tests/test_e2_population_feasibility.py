from tools.e2_population_feasibility import candidate_id, select_blocks


def _candidate(a, b, relation_a="holding", relation_b="looking at"):
    value = {
        "image_a": str(a),
        "image_b": str(b),
        "subject_label": "person",
        "object_label": "object",
        "relation_a": relation_a,
        "relation_b": relation_b,
    }
    value["candidate_id"] = candidate_id(value)
    return value


def test_candidate_id_is_deterministic():
    assert candidate_id(_candidate("1", "2")) == candidate_id(_candidate("1", "2"))
    assert candidate_id(_candidate("1", "2")) != candidate_id(_candidate("2", "1"))


def test_select_blocks_is_deterministic_and_image_disjoint():
    candidates = [
        _candidate("1", "2"),
        _candidate("3", "4"),
        _candidate("2", "5"),
        _candidate("6", "7", "holding", "using"),
    ]
    quotas = {"holding||looking at": 2, "holding||using": 1}
    a = select_blocks(candidates, quotas, seed=11)
    b = select_blocks(candidates, quotas, seed=11)
    assert a == b
    images = [image for c in a for image in (c["image_a"], c["image_b"])]
    assert len(images) == len(set(images))


def test_select_blocks_honors_forbidden_images_and_quotas():
    candidates = [_candidate(str(i), str(i + 100)) for i in range(5)]
    selected = select_blocks(
        candidates,
        {"holding||looking at": 10},
        seed=0,
        forbidden_images={"0", "100"},
    )
    assert all("0" not in (c["image_a"], c["image_b"]) for c in selected)
    assert len(selected) <= 4
