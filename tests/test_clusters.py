from __future__ import annotations

import urllib.error

from ai_experiments.clusters import ClusterProfile, ClusterStatus, cluster_status


def _profile(address: str | None) -> ClusterProfile:
    return ClusterProfile(name="vader", provider="local", address=address)


def test_cluster_status_no_address_configured():
    status = cluster_status(_profile(address=None))

    assert isinstance(status, ClusterStatus)
    assert status.reachable is False
    assert status.error == "no address configured"
    assert status.address is None
    assert status.ray_version is None

    # Pin the serialization: model_dump(mode="json") now emits address/
    # ray_version as explicit None where they were previously absent keys.
    dumped = status.model_dump(mode="json")
    assert set(dumped) == {"name", "reachable", "address", "ray_version", "error"}


def test_cluster_status_successful_probe(monkeypatch):
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return b'{"ray_version": "2.9.0"}'

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: _FakeResponse())

    status = cluster_status(_profile(address="http://vader:8265"))

    assert status.reachable is True
    assert status.address == "http://vader:8265"
    assert status.ray_version == "2.9.0"
    assert status.error is None


def test_cluster_status_unreachable(monkeypatch):
    def _raise(url, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    status = cluster_status(_profile(address="http://vader:8265"))

    assert status.reachable is False
    assert status.address == "http://vader:8265"
    assert status.error
