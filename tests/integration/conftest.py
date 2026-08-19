"""Fixtures for the integration suite: real Ray, real MLflow, no fakes.

The unit suite stubs the Ray client and the mlflow module, which is fast and
covers states a live cluster will not produce on demand (resource starvation,
a stuck PENDING job). It cannot tell you whether the harness actually works
against real services -- the MLflow linkage bug in issue #6 passed the entire
stubbed suite.

These tests skip when the services are not up, so `pytest tests` stays green
without Docker. Bring them up with:

    docker compose -f tests/integration/compose.yaml up -d
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

RAY_ADDRESS = "http://127.0.0.1:8265"
MLFLOW_URI = "http://127.0.0.1:5000"


def _reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


@pytest.fixture(scope="session")
def ray_address() -> str:
    pytest.importorskip("ray", reason="needs the ray extra: uv sync --extra ray")
    if not _reachable(f"{RAY_ADDRESS}/api/version"):
        pytest.skip(
            f"no Ray cluster at {RAY_ADDRESS} -- "
            "docker compose -f tests/integration/compose.yaml up -d"
        )
    return RAY_ADDRESS


@pytest.fixture(scope="session")
def mlflow_uri() -> str:
    # The client library matters as much as the server: without it
    # begin_tracking degrades to a warning event and records no linkage, so
    # every assertion here would fail rather than skip.
    pytest.importorskip(
        "mlflow", reason="needs the mlflow extra: uv sync --extra mlflow"
    )
    if not _reachable(f"{MLFLOW_URI}/health"):
        pytest.skip(
            f"no MLflow server at {MLFLOW_URI} -- "
            "docker compose -f tests/integration/compose.yaml up -d"
        )
    return MLFLOW_URI


@pytest.fixture(scope="session")
def mlflow_api(mlflow_uri: str):
    """Query the real tracking server over its REST API.

    Deliberately not the mlflow client: asserting through the same library the
    harness writes with would hide a broken write.
    """

    def _get(path: str, **params: str) -> dict:
        url = f"{mlflow_uri}/api/2.0/mlflow/{path}"
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read())

    return _get
