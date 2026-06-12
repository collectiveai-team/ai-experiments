from __future__ import annotations

import subprocess

from ai_experiments.repro import capture_repro, current_git_sha, read_repro
from ai_experiments.schemas import ExperimentManifest, WorkloadSpec
from ai_experiments.store import FilesystemRunStore


def _manifest(working_dir: str) -> ExperimentManifest:
    return ExperimentManifest(
        experiment="artifacts-repro",
        workload=WorkloadSpec(entrypoint="python train.py", working_dir=working_dir),
    )


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "train.py").write_text("print('hi')\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def test_artifacts_listing(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    run_id, run_dir = store.create_run(_manifest(str(tmp_path)))

    artifacts = run_dir / "artifacts"
    (artifacts / "checkpoints").mkdir()
    (artifacts / "checkpoints" / "best.pt").write_bytes(b"x" * 100)
    (artifacts / "loss.png").write_bytes(b"y" * 5)

    entries = store.list_artifacts(run_id)
    paths = [e["path"] for e in entries]
    assert paths == ["checkpoints/best.pt", "loss.png"]
    assert entries[0]["size_bytes"] == 100


def test_artifacts_empty_for_fresh_run(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs", capture_repro=False)
    run_id, _ = store.create_run(_manifest(str(tmp_path)))

    assert store.list_artifacts(run_id) == []


def test_repro_capture_in_git_repo(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "train.py").write_text("print('changed')\n")  # dirty tree

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    context = capture_repro(run_dir, repo)

    assert context["git_sha"] == current_git_sha(repo)
    assert context["git_dirty"] is True
    assert "changed" in (run_dir / "repro" / "diff.patch").read_text()
    assert (run_dir / "repro" / "environment.txt").exists()
    assert read_repro(run_dir) == context


def test_repro_capture_outside_git_repo(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    context = capture_repro(run_dir, plain)

    assert context["git_sha"] is None
    assert context["python"]
    assert not (run_dir / "repro" / "diff.patch").exists()


def test_create_run_captures_repro_bundle(tmp_path):
    repo = _git_repo(tmp_path)
    store = FilesystemRunStore(tmp_path / "runs")  # capture on by default

    _run_id, run_dir = store.create_run(_manifest(str(repo)))

    context = read_repro(run_dir)
    assert context is not None
    assert context["git_sha"] is not None
    assert context["git_dirty"] is False
