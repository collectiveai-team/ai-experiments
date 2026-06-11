"""Toy training workload: minimizes (x - 2)^2 with noisy gradient steps.

Demonstrates the metric contract: print ``IAX_METRIC {json}`` lines (or use
ai_experiments.report.report_metric). Works on the local backend and on any
Ray cluster — the harness extracts metrics from stdout either way.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time


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

    print(f"final x={x:.4f} loss={(x - 2.0) ** 2:.6f}")


if __name__ == "__main__":
    main()
