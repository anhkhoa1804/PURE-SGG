from pathlib import Path

from openvocab_rel import paths


def test_default_roots_are_repository_relative(monkeypatch):
    for name in (
        "RESEARCH_NO1_DATA_ROOT",
        "RESEARCH_NO1_CHECKPOINT_ROOT",
        "RESEARCH_NO1_RUN_ROOT",
        "RESEARCH_NO1_CACHE_ROOT",
        "DATA_ROOT",
        "HF_HOME",
    ):
        monkeypatch.delenv(name, raising=False)

    assert paths.data_root() == paths.REPO_ROOT / "datasets_vg150_clean"
    assert paths.checkpoint_root() == paths.REPO_ROOT / "checkpoints"
    assert paths.run_root() == paths.REPO_ROOT / "runs"
    assert paths.cache_root() == paths.REPO_ROOT / "caches"


def test_new_environment_variables_win_over_legacy(monkeypatch, tmp_path):
    new = tmp_path / "new-data"
    legacy = tmp_path / "legacy-data"
    monkeypatch.setenv("RESEARCH_NO1_DATA_ROOT", str(new))
    monkeypatch.setenv("DATA_ROOT", str(legacy))
    assert paths.data_root() == new


def test_legacy_data_root_remains_supported(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy-data"
    monkeypatch.delenv("RESEARCH_NO1_DATA_ROOT", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(legacy))
    assert paths.data_root() == legacy


def test_relative_external_root_resolves_from_process_directory(monkeypatch):
    monkeypatch.setenv("RESEARCH_NO1_RUN_ROOT", "external-runs")
    assert paths.run_root() == Path("external-runs").resolve()
