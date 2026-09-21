import json
import os
import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "maintenance_events.train", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _metrics(stdout):
    return [
        json.loads(line[len("IAX_METRIC ") :])
        for line in stdout.splitlines()
        if line.startswith("IAX_METRIC ")
    ]


def test_self_test_emits_the_objective_exactly_once():
    result = _run("--self-test", "--window-days", "60", "--n-folds", "3", "--model", "logreg")
    assert result.returncode == 0, result.stderr
    points = _metrics(result.stdout)
    objective = [p for p in points if "pr_auc" in p]
    assert len(objective) == 1


def test_per_fold_metrics_never_use_the_objective_key():
    """extract_objective se queda con el MEJOR valor: compartir clave haría que
    la campaña registrara el fold más afortunado en vez del promedio."""
    result = _run("--self-test", "--window-days", "60", "--n-folds", "3", "--model", "logreg")
    points = _metrics(result.stdout)
    per_fold = [p for p in points if "fold_pr_auc" in p]
    assert per_fold
    assert all("pr_auc" not in p for p in per_fold)


def test_final_point_carries_the_baseline_and_valid_folds():
    result = _run("--self-test", "--window-days", "60", "--n-folds", "3", "--model", "logreg")
    final = [p for p in _metrics(result.stdout) if "pr_auc" in p][0]
    assert {
        "pr_auc", "pr_auc_std", "auroc", "recall_at_p50", "baseline_pr_auc", "valid_folds",
    } <= set(final)


def test_search_space_flag_names_are_accepted():
    result = _run(
        "--self-test",
        "--window-days", "30",
        "--resample-freq", "6h",
        "--label-source", "union",
        "--model", "hist_gb",
        "--learning-rate", "0.05",
        "--max-leaf-nodes", "15",
        "--min-samples-leaf", "5",
        "--n-folds", "3",
    )
    assert result.returncode == 0, result.stderr


def test_missing_dataset_fails_with_instructions(tmp_path, monkeypatch):
    """Corre con PROJECT_DIR apuntando a un directorio vacío: si el test
    dependiera de que no exista dataset.local.toml, pasaría en CI y se caería en
    cuanto alguien configure el link en su máquina."""
    empty_project = tmp_path / "project"
    empty_project.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "maintenance_events.train", "--n-folds", "2"],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "MAINTENANCE_EVENTS_CACHE": str(tmp_path / "cache"),
            "MAINTENANCE_EVENTS_URL": "",
            "MAINTENANCE_EVENTS_PROJECT_DIR": str(empty_project),
        },
    )
    assert result.returncode != 0
    assert "dataset.local.toml" in result.stderr or "dataset.local.toml" in result.stdout


def test_max_bins_is_searchable_and_reaches_the_model():
    """``max_bins`` es la palanca de costo dominante del binner de HistGB: con
    sample_weight (class_weight=balanced) sklearn calcula un percentil ponderado
    por feature y por bin, así que el tiempo del fit es casi lineal en max_bins."""
    from maintenance_events.models import make_model
    from maintenance_events.train import parse_args

    assert parse_args(["--max-bins", "32"]).max_bins == 32
    model = make_model("hist_gb", max_bins=32)()
    assert model.max_bins == 32

    result = _run("--self-test", "--window-days", "30", "--n-folds", "2",
                  "--model", "hist_gb", "--max-bins", "16")
    assert result.returncode == 0, result.stderr


def test_thread_limit_is_applied_and_overridable(monkeypatch):
    """iax corre varios trials a la vez; con 16 hilos por trial el fit tarda el
    doble que con uno, porque el dataset es chico y solo paga sincronización."""
    from maintenance_events.train import resolve_thread_limit

    monkeypatch.delenv("MAINTENANCE_EVENTS_THREADS", raising=False)
    assert resolve_thread_limit() == 1
    monkeypatch.setenv("MAINTENANCE_EVENTS_THREADS", "4")
    assert resolve_thread_limit() == 4
    monkeypatch.setenv("MAINTENANCE_EVENTS_THREADS", "0")
    assert resolve_thread_limit() is None


def test_every_search_space_key_is_a_valid_flag():
    """Fija el contrato contra goal.yaml: si alguien agrega una clave al espacio
    de búsqueda sin la flag, la campaña falla en el primer trial, no acá.

    Las claves de ``search_space`` son identificadores de Python; iax las manda
    con la grafía que argparse declara por convención (``flag_style: hyphen``,
    el default). argparse rechaza cualquier opción larga que no declaró, así que
    esta traducción es el contrato entero: una sola grafía, la de guiones.
    """
    import pathlib

    import yaml

    from maintenance_events.train import parse_args

    goal = yaml.safe_load((pathlib.Path(__file__).parents[1] / "goal.yaml").read_text())
    defaults = {"choice": lambda s: str(s["values"][0]), "loguniform": lambda s: str(s["low"])}
    argv = []
    for key, spec in goal["search_space"].items():
        argv += [f"--{key.replace('_', '-')}", defaults[spec["type"]](spec)]
    parse_args(argv)  # no levanta SystemExit


def test_the_objective_and_its_baseline_are_both_reported(capsys):
    """goal.yaml puntúa cada trial por ``pr_auc - baseline_pr_auc``. Si el
    workload dejara de emitir la baseline, el trial no tendría objetivo."""
    import json
    import pathlib

    import yaml

    from maintenance_events.train import main

    goal = yaml.safe_load((pathlib.Path(__file__).parents[1] / "goal.yaml").read_text())
    assert main(["--self-test", "--n-folds", "3"]) == 0

    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("IAX_METRIC ")]
    final = json.loads(lines[-1].removeprefix("IAX_METRIC "))
    assert goal["objective"]["metric"] in final
    assert goal["objective"]["baseline_metric"] in final
