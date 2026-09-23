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


def test_every_field_binds_the_variable_it_claims(monkeypatch):
    """Pin all seven aliases at once, so a typo in any `validation_alias` fails loudly.

    Three fields -- `artifacts_dir`, `notify_webhook`, `notify_command` -- have no other test
    reaching them: `tests/test_report.py` never calls `artifacts_dir()`, and `tests/test_notify.py`
    always passes `webhook_url=`/`command=` explicitly, so the env fallback never runs. Without
    this test a mistyped alias would bind silently wrong (the field would just keep its default)
    with nothing in the suite to notice.
    """
    expected = {
        "IAX_RUNS_DIR": ("runs_dir", "/pinned/runs"),
        "IAX_ARTIFACTS_DIR": ("artifacts_dir", "/pinned/artifacts"),
        "IAX_CLUSTERS": ("clusters_config", "/pinned/clusters.yaml"),
        "IAX_NOTIFY_WEBHOOK": ("notify_webhook", "https://pinned.example/hook"),
        "IAX_NOTIFY_COMMAND": ("notify_command", "/pinned/notify.sh"),
        "MLFLOW_TRACKING_URI": ("mlflow_tracking_uri", "file:///pinned/mlruns"),
        "RAY_ADDRESS": ("ray_address", "http://pinned:8265"),
    }
    for variable, (_, value) in expected.items():
        monkeypatch.setenv(variable, value)

    settings = get_settings()
    bound = {variable: getattr(settings, field) for variable, (field, _) in expected.items()}
    assert bound == {variable: value for variable, (_, value) in expected.items()}


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_unknown_keys_are_ignored_not_fatal(monkeypatch):
    monkeypatch.setenv("IAX_NOT_A_SETTING", "1")
    Settings()  # asserts no raise: model_config sets extra="ignore"
