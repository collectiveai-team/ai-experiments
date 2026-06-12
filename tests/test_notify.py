from __future__ import annotations

import json
import sys

from ai_experiments.notify import Notifier, read_notifications


def test_notifications_always_logged(tmp_path):
    notifier = Notifier(tmp_path / "runs")

    notifier.send("campaign completed", "toy: target_reached", campaign_id="cmp_1")

    logged = read_notifications(tmp_path / "runs")
    assert len(logged) == 1
    assert logged[0]["title"] == "campaign completed"
    assert logged[0]["text"] == "campaign completed: toy: target_reached"
    assert logged[0]["campaign_id"] == "cmp_1"


def test_command_sink_receives_payload(tmp_path):
    sink_file = tmp_path / "received.json"
    script = tmp_path / "sink.py"
    script.write_text(
        "import sys, pathlib\n"
        f"pathlib.Path({str(sink_file)!r}).write_text(sys.stdin.read())\n"
    )
    notifier = Notifier(tmp_path / "runs", command=f"{sys.executable} {script}")

    notifier.send("run auto_killed", "run_x: timeout_exceeded", run_id="run_x")

    received = json.loads(sink_file.read_text())
    assert received["run_id"] == "run_x"


def test_failing_sinks_never_raise(tmp_path):
    notifier = Notifier(
        tmp_path / "runs",
        webhook_url="http://127.0.0.1:1/nope",  # refused
        command="/nonexistent-binary-xyz",
    )

    payload = notifier.send("t", "m")

    assert payload["title"] == "t"
    assert len(read_notifications(tmp_path / "runs")) == 1
