"""A minimal instrumented workload.

The harness reads two things from a workload:

1. Stdout lines. ``IAX_METRIC {json}`` is one point on the progress curve;
   ``step`` is optional but makes progress and plateau detection work.
   ``IAX_RESULT {json}`` is the one declared result — only this is scored.
2. The exit code. Zero means completed, anything else means failed.

It hands the workload three environment variables:

* ``IAX_PARAMS``        the trial's params as JSON (a campaign sets this)
* ``IAX_TRIAL_ID``      the trial id, empty for a one-off run
* ``IAX_ARTIFACTS_DIR`` where to write checkpoints and outputs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def report(step: int, **metrics: float) -> None:
    """Report one metric point to the harness."""
    print("IAX_METRIC " + json.dumps({"step": step, **metrics}), flush=True)


def report_result(**values: float) -> None:
    """Declare the one result the harness will score."""
    print("IAX_RESULT " + json.dumps(values), flush=True)


def main() -> None:
    params = json.loads(os.environ.get("IAX_PARAMS") or "{}")
    lr = float(params.get("lr", 0.01))

    # A stand-in for training: the loss falls by `lr` each step. Replace the
    # body of this loop with a real step; keep the report() call.
    loss = 1.0
    for step in range(100):
        loss = max(loss - lr, 0.0)
        report(step, loss=loss)

    report_result(loss=loss)

    artifacts = os.environ.get("IAX_ARTIFACTS_DIR")
    if artifacts:
        with (Path(artifacts) / "result.json").open("w") as fh:
            json.dump({"loss": loss, "params": params}, fh)

    sys.exit(0)


if __name__ == "__main__":
    main()
