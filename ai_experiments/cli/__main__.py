"""Allow `python -m ai_experiments.cli`, matching how `ai_experiments.worker` is launched."""

from ai_experiments.cli import app

if __name__ == "__main__":
    app()
