"""Stuck training workload: requests a GPU on a CPU-only cluster.

The inner Ray task is infeasible, so Ray keeps it pending forever and the
driver blocks in ray.get(). The Ray driver logs an unschedulable-resource
warning, which the iax Ray monitor should classify as resource_starved.
"""

import time

import ray


def main() -> None:
    ray.init(address="auto")
    print(f"cluster_resources={ray.cluster_resources()}", flush=True)
    print(f"available_resources={ray.available_resources()}", flush=True)

    @ray.remote(num_gpus=1)
    def needs_gpu() -> str:
        return "ran"

    print("submitting a GPU task on a CPU-only cluster; it will never schedule", flush=True)
    future = needs_gpu.remote()

    # Block forever waiting on an infeasible task. This is the "stuck" condition.
    while True:
        ready, _ = ray.wait([future], timeout=10)
        if ready:
            print(ray.get(ready[0]), flush=True)
            return
        print("still waiting for GPU resources...", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
