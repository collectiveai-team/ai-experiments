"""Evaluator for the toy workload: scores the artifact the trainer produced.

This is the protected phase. It runs from a tree the agent cannot edit, and it
is the only one whose ``IAX_RESULT`` line the harness will score.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ai_experiments.report import report_result


def main() -> None:
    work = Path(os.environ.get("IAX_WORK_DIR", "."))
    with (work / "model.json").open() as fh:
        model = json.load(fh)

    loss = (model["x"] - 2.0) ** 2
    report_result(loss=loss)


if __name__ == "__main__":
    main()
