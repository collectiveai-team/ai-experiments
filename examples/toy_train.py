"""Toy training workload: minimizes (x - 2)^2 with noisy gradient steps.

The trainer reports progress via ``IAX_METRIC {json}`` lines (or
``ai_experiments.report.report_metric``), hands the trained model off through
``$IAX_WORK_DIR``, but does not declare a result. Declaring the result (the
final loss that actually scores the trial) is the evaluator's job
(``examples/toy_evaluate.py``). Works on the local backend and on any Ray
cluster — the harness extracts metrics and results from both phases' stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--x0", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    rng = random.Random(0)
    x = args.x0
    for step in range(args.steps):
        grad = 2 * (x - 2.0) + rng.gauss(0, 0.1)
        x -= args.lr * grad
        loss = (x - 2.0) ** 2
        print("IAX_METRIC " + json.dumps({"step": step, "loss": loss, "x": x}))
        sys.stdout.flush()
        time.sleep(args.sleep)

    # The trainer produces an artifact. It does not declare a result: the
    # number it would be judged by is not its to report.
    loss = (x - 2.0) ** 2
    work = Path(os.environ.get("IAX_WORK_DIR", "."))
    with (work / "model.json").open("w") as fh:
        json.dump({"x": x}, fh)

    # Anything written to $IAX_ARTIFACTS_DIR is listed by `iax artifacts
    # <run_id>` and downloadable from the dashboard.
    artifacts = os.environ.get("IAX_ARTIFACTS_DIR")
    if artifacts:
        with open(os.path.join(artifacts, "model.json"), "w") as fh:
            json.dump({"x": x, "loss": loss}, fh)

    print(f"final x={x:.4f} loss={loss:.6f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
