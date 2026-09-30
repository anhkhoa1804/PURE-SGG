import inspect

from tools import e2_score_models


def test_nuisance_fit_has_explicit_train_source():
    source = inspect.getsource(e2_score_models)
    assert "training_rows_from_jsonl" in source
    assert "--train" in source
    assert "fit_geometry(train_path" in source
    assert "fit_count_baseline(train_path" in source


def test_training_iterator_rejects_evaluation_image(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"image_id": "eval", "obj_boxes": [[0,0,1,1],[0,0,1,1]], "objects": [{"object_id":0,"names":["a"]},{"object_id":1,"names":["b"]}], "relationships": [{"subject_id":0,"object_id":1,"predicate":"holding"}]}\n')
    import pytest
    with pytest.raises(ValueError):
        next(e2_score_models.training_rows_from_jsonl(train, {"eval"}))
