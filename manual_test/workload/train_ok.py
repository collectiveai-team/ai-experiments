"""Small OK training workload: runs a few epochs, logs metrics, exits 0."""

import sys
import time


def main() -> int:
    print("starting OK training run", flush=True)
    best = None
    for epoch in range(1, 6):
        loss = 1.0 / epoch
        best = loss if best is None else min(best, loss)
        print(f"epoch {epoch}/5 loss={loss:.4f}", flush=True)
        time.sleep(2)
    print(f"training complete: best_loss={best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
