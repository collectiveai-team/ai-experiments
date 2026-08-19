# Integration tests — real Ray, real MLflow

These exercise the harness against live services. No stubbed Ray client, no
fake mlflow module.

## Why they exist

The unit suite stubs `JobSubmissionClient` and the `mlflow` module. That is
fast and it covers states a live cluster will not produce on demand (resource
starvation, a job stuck PENDING), so it stays. But it cannot tell you whether
the harness works against real services.

Issue #6 is the proof: `RayBackend.submit` destroyed the MLflow linkage it had
just created, so no Ray run was ever mirrored and every one sat `RUNNING` in
MLflow forever. **The entire stubbed suite passed.** Against a real server the
failure is immediate and obvious.

## Running them

```bash
docker compose -f tests/integration/compose.yaml up -d
uv sync --python 3.12 --extra ray --extra mlflow --extra dev
uv run --python 3.12 --extra ray --extra mlflow --extra dev \
    python -m pytest tests/integration -v
docker compose -f tests/integration/compose.yaml down -v
```

They skip when the services are unreachable or `ray` is not installed, so
plain `pytest tests` stays green without Docker.

## Version pinning matters

The Ray Jobs API is versioned, so the client and the cluster must agree. The
compose file pins `rayproject/ray:2.37.0-py312`; install the matching client:

```bash
uv pip install "ray[default]==2.37.0" "mlflow-skinny==2.17.2"
```

A mismatched client can fail in ways that look like harness bugs.

## Why host networking

The harness injects `MLFLOW_TRACKING_URI` into the Ray job's `runtime_env`, so
the workload resolves that URI *inside the cluster* while the harness uses it
from outside. Both sides must agree on `127.0.0.1:5000`. A bridge network would
need split-horizon DNS to make one URI work in both places.

## Python version

`ray` publishes no wheels for 3.13+ or free-threaded builds, so the
integration venv is pinned to 3.12. The default dev environment can be newer —
these tests skip there.

## Verifying a test actually catches its bug

A regression test that passes against the broken code is worthless. Check it:

```bash
git checkout <bad-commit> -- ai_experiments/
uv run --python 3.12 --extra ray --extra mlflow --extra dev \
    python -m pytest tests/integration -v      # expect failures
git checkout HEAD -- ai_experiments/
```

Note `git stash` does **not** work for this once the fix is committed — it only
stashes uncommitted changes, and the tests will silently pass against the fix
while appearing to test the old code.
