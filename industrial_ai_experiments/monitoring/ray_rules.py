from __future__ import annotations

RESOURCE_STARVATION_PATTERNS = (
    "cannot be scheduled",
    "insufficient resources",
    "no available node",
    "no available nodes",
    "resource request",
    "resources are not available",
    "waiting for resources",
)

STUCK_PATTERNS = (
    "actor died",
    "deadlock",
    "heartbeat timeout",
    "node failure",
    "worker crashed",
)


def classify_ray_condition(details: dict[str, object]) -> str:
    """Classify Ray-specific state without changing the generic run state."""
    status = str(details.get("ray_status", "")).lower()
    text = "\n".join(
        str(value)
        for key, value in details.items()
        if key in {"ray_message", "ray_error_type", "ray_log_tail", "ray_job_info"}
        and value is not None
    ).lower()

    if any(pattern in text for pattern in RESOURCE_STARVATION_PATTERNS):
        return "resource_starved"
    if any(pattern in text for pattern in STUCK_PATTERNS):
        return "stuck_suspected"
    if status == "pending":
        return "queued"
    if status == "running":
        return "running"
    if status == "succeeded":
        return "completed"
    if status in {"failed", "stopped"}:
        return status
    return "unknown"
