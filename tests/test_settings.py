"""CES-76: every environment read in the package resolves through this one module."""

from __future__ import annotations

from ai_experiments.settings import Settings, get_settings


def test_defaults_do_not_require_any_environment(monkeypatch):
    monkeypatch.delenv("IAX_RUNS_DIR", raising=False)
    assert get_settings().runs_dir == "outputs/experiments/runs"


def test_environment_overrides_the_default(monkeypatch):
    monkeypatch.setenv("IAX_RUNS_DIR", "/elsewhere/runs")
    assert get_settings().runs_dir == "/elsewhere/runs"


def test_binding_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("iax_runs_dir", "/lower/runs")
    assert get_settings().runs_dir == "/lower/runs"


def test_unprefixed_third_party_variables_bind_too(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "http://ray:8265")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///mlruns")
    settings = get_settings()
    assert settings.ray_address == "http://ray:8265"
    assert settings.mlflow_tracking_uri == "file:///mlruns"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_unknown_keys_are_ignored_not_fatal(monkeypatch):
    monkeypatch.setenv("IAX_NOT_A_SETTING", "1")
    Settings()  # asserts no raise: model_config sets extra="ignore"
