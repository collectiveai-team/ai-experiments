"""A workload whose failures are informative, for `strategy: agent`.

A random or adaptive search learns only from trials that produced a number.
This one refuses configurations that would not fit in memory, and says so:

    RuntimeError: out of memory: needs 5.6 GB, the device has 4.0 GB

The failure names the constraint. An agent planner reads it in the trial
history and stops proposing configurations on the wrong side of it, which is
the whole reason to hand round planning to an agent.

Nothing here trains anything: the loss is an analytic surface, so the example
runs in seconds on a laptop and still exercises the full loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

#: Pretend device. The interesting configurations sit near this boundary.
DEVICE_MEMORY_GB = 4.0


def memory_gb(width: int, depth: int, batch: int) -> float:
    """Activations plus parameters, in the shape a real model would scale."""
    return (width * width * depth * 4 + width * depth * batch * 32) / 1e6


def loss_surface(lr: float, width: int, depth: int) -> float:
    """A capacity term and a learning-rate term. The optimum needs both."""
    capacity = 1.0 / (1.0 + math.log1p(width * depth) / 4.0)
    schedule = (math.log10(lr) + 2.6) ** 2 / 6.0
    return capacity + schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()

    needed = memory_gb(args.width, args.depth, args.batch)
    if needed > DEVICE_MEMORY_GB:
        # A trial that cannot run must fail loudly. The harness records this
        # message in the campaign state, and the planner gets to read it.
        reason = (
            f"out of memory: needs {needed:.1f} GB, "
            f"the device has {DEVICE_MEMORY_GB:.1f} GB "
            f"(width={args.width}, depth={args.depth}, batch={args.batch})"
        )
        raise RuntimeError(reason)

    target = loss_surface(args.lr, args.width, args.depth)
    rng = random.Random(args.width * 1000 + args.depth)
    loss = target + 2.0

    for step in range(args.steps):
        loss = target + (loss - target) * 0.7 + rng.gauss(0, 0.01)
        print("IAX_METRIC " + json.dumps({"step": step, "loss": round(loss, 6)}))
        sys.stdout.flush()
        time.sleep(args.sleep)

    artifacts = os.environ.get("IAX_ARTIFACTS_DIR")
    if artifacts:
        with open(os.path.join(artifacts, "summary.json"), "w") as fh:
            json.dump({"loss": loss, "memory_gb": round(needed, 3)}, fh)

    print(f"final loss={loss:.6f} memory={needed:.2f}GB")


if __name__ == "__main__":
    main()
