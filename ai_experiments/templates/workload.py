"""A minimal instrumented workload.

The harness reads two things from a workload:

1. ``IAX_METRIC {json}`` lines on stdout. Each line is one point; ``step`` is
   optional but makes progress and plateau detection work.
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


def report(step: int, **metrics: float) -> None:
    """Report one metric point to the harness."""
    print("IAX_METRIC " + json.dumps({"step": step, **metrics}), flush=True)


def main() -> None:
    params = json.loads(os.environ.get("IAX_PARAMS") or "{}")
    lr = float(params.get("lr", 0.01))

    loss = 1.0
    for step in range(20):
        loss = max(loss - lr, 0.0)  # replace with a real training step
        report(step, loss=loss)

    artifacts = os.environ.get("IAX_ARTIFACTS_DIR")
    if artifacts:
        with open(os.path.join(artifacts, "result.json"), "w") as fh:
            json.dump({"loss": loss, "params": params}, fh)

    sys.exit(0)


if __name__ == "__main__":
    main()
