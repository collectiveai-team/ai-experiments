"""A key iax does not know is an error, not a comment (#14).

The loop's whole input is a YAML file an agent wrote. When the agent invents
`monitor:` for `monitoring:`, the old schema accepted the file and ran under
defaults, so the agent believed it had configured something it had not. These
tests hold the two halves of the fix: unknown keys are refused by name, and
files iax itself wrote still load after a field is removed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from ai_experiments import scaffold
from ai_experiments.api import goal_from_dict
from ai_experiments.cli_support import IaxError
from ai_experiments.schema_errors import _model_in
from ai_experiments.schemas import (
    ExperimentManifest,
    GoalSpec,
    MonitorPolicy,
    load_stored,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

GOAL: dict[str, object] = {
    "goal": "minimize loss",
    "name": "strict",
    "objective": {"metric": "loss", "mode": "min", "target": 0.1},
    "search_space": {"lr": {"type": "uniform", "low": 0.0, "high": 1.0}},
    "workload": {"entrypoint": "true"},
}


def _write(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _manifest(**extra: object) -> dict[str, object]:
    return {"experiment": "strict", "workload": {"entrypoint": "true"}, **extra}


def test_the_readme_shaped_typo_is_refused_and_named(tmp_path):
    """The exact file the README used to teach: `monitor:`/`stuck_after_seconds`."""
    path = _write(tmp_path, _manifest(monitor={"stuck_after_seconds": 60}))

    with pytest.raises(ValueError) as caught:
        ExperimentManifest.from_yaml(path)

    message = str(caught.value)
    assert "monitor: unknown field" in message
    assert "did you mean 'monitoring'?" in message


def test_a_nested_typo_names_the_nested_field(tmp_path):
    path = _write(tmp_path, _manifest(workload={"entrypoint": "true", "workdir": "."}))

    with pytest.raises(ValueError, match=r"workload.workdir: unknown field"):
        ExperimentManifest.from_yaml(path)


def test_an_unrecognizable_key_lists_the_fields_that_do_exist(tmp_path):
    """No near match means the author needs the menu, not a guess."""
    path = _write(tmp_path, _manifest(gpu_quota=4))

    with pytest.raises(ValueError) as caught:
        ExperimentManifest.from_yaml(path)

    message = str(caught.value)
    assert "did you mean" not in message
    assert "known fields:" in message
    assert "resources" in message


def test_a_goal_reports_every_bad_key_at_once(tmp_path):
    """An agent that fixes one key per round burns a round per typo."""
    bad = dict(GOAL, budget={"max_trails": 4}, stratagem={"name": "grid"})
    path = _write(tmp_path, bad)

    with pytest.raises(ValueError) as caught:
        GoalSpec.from_yaml(path)

    message = str(caught.value)
    assert "budget.max_trails: unknown field; did you mean 'max_trials'?" in message
    assert "stratagem: unknown field; did you mean 'strategy'?" in message


def test_a_goal_an_agent_composed_in_memory_gets_the_same_message():
    with pytest.raises(IaxError) as caught:
        goal_from_dict(dict(GOAL, objective={"metric": "loss", "targt": 0.1}))

    assert caught.value.code == "invalid_input"
    assert "objective.targt: unknown field; did you mean 'target'?" in (
        caught.value.message
    )


@pytest.mark.parametrize("dead_key", ["checks", "no_event_after_minutes"])
def test_a_removed_monitoring_key_says_it_was_removed(tmp_path, dead_key):
    """ "Unknown field" would send the author looking for the right spelling."""
    path = _write(tmp_path, _manifest(monitoring={dead_key: ["anything"]}))

    with pytest.raises(ValueError) as caught:
        ExperimentManifest.from_yaml(path)

    message = str(caught.value)
    assert f"{dead_key}: removed" in message
    assert "never read" in message


def test_the_monitoring_rules_have_no_configuration_left_to_ignore():
    """The fields exist because something reads them. Nothing reads a check name."""
    assert "checks" not in MonitorPolicy.model_fields
    assert "no_event_after_minutes" not in MonitorPolicy.model_fields


def test_a_run_written_by_an_older_iax_still_replays(tmp_path):
    """Every stored manifest carried `checks:`; refusing them would strand runs."""
    from ai_experiments.store import FilesystemRunStore

    store = FilesystemRunStore(tmp_path / "runs")
    run_id, run_dir = store.create_run(
        ExperimentManifest(**_manifest(monitoring={"stuck_after_minutes": 7}))
    )
    stored = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    stored["monitoring"]["checks"] = ["no_status_update"]
    stored["monitoring"]["no_event_after_minutes"] = 15
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump(stored))

    manifest = store.read_manifest(run_id)

    assert manifest is not None
    assert manifest.monitoring.stuck_after_minutes == 7


def test_a_campaign_written_by_an_older_iax_still_loads(tmp_path):
    from ai_experiments.store.campaign import CampaignStore

    store = CampaignStore(tmp_path / "campaigns")
    campaign_id = store.create_campaign(GoalSpec(**GOAL)).campaign_id
    goal_file = store.campaign_dir(campaign_id) / "goal.yaml"
    stored = yaml.safe_load(goal_file.read_text())
    stored["monitoring"]["checks"] = ["process_exit"]
    goal_file.write_text(yaml.safe_dump(stored))

    assert store.read_goal(campaign_id).name == "strict"


def test_a_stored_file_is_still_checked_for_real_mistakes(tmp_path):
    """Tolerating removed keys must not tolerate a corrupt file."""
    path = _write(tmp_path, _manifest(experiment=""))

    with pytest.raises(ValueError, match="experiment"):
        load_stored(ExperimentManifest, path)


#: A commented-out field in a template: `  # gpus: 1`.
_COMMENTED_KEY = re.compile(r"^\s*#\s*([a-z_][a-z0-9_]*):")
#: Top-level sections whose keys the author names: parameters, env vars, tags.
_FREE_FORM = {"search_space", "env", "metadata"}


def _every_field(model: type[BaseModel], seen: set[str] | None = None) -> set[str]:
    """Every field name anywhere in a model tree."""
    names = seen if seen is not None else set()
    for name, field in model.model_fields.items():
        names.add(name)
        nested = _model_in(field.annotation)
        if nested is not None and nested is not model:
            _every_field(nested, names)
    return names


@pytest.mark.parametrize("kind", ["manifest", "goal"])
def test_no_template_comment_offers_a_field_that_does_not_exist(tmp_path, kind):
    """`iax new` writes these. A commented `cluster:` is now a file that fails."""
    path = tmp_path / f"{kind}.yaml"
    scaffold.write(kind, path)
    model = ExperimentManifest if kind == "manifest" else GoalSpec

    offered = set()
    section = ""
    for line in path.read_text().splitlines():
        if line[:1].isalpha():
            section = line.split(":", 1)[0]
        match = _COMMENTED_KEY.match(line)
        if match and section not in _FREE_FORM:
            offered.add(match.group(1))

    assert offered
    assert offered <= _every_field(model)


@pytest.mark.parametrize(
    "path", sorted(EXAMPLES.glob("goal_*.yaml")), ids=lambda p: p.name
)
def test_every_shipped_example_validates(path):
    assert GoalSpec.from_yaml(path).search_space


@pytest.mark.parametrize("kind", ["manifest", "goal"])
def test_what_iax_writes_is_what_iax_reads(tmp_path, kind):
    """`extra=forbid` breaks the tool itself if a dump names a field the model lost."""
    model = ExperimentManifest if kind == "manifest" else GoalSpec
    original = model(**(_manifest() if kind == "manifest" else GOAL))
    path = tmp_path / "dumped.yaml"
    path.write_text(original.to_yaml())

    assert model.from_yaml(path) == original


def test_a_typo_in_a_cluster_profile_is_refused_by_name(tmp_path):
    """A dropped `address:` shows up much later, as an unreachable backend."""
    from ai_experiments.clusters import ClusterConfigError, load_clusters

    path = tmp_path / "clusters.yaml"
    path.write_text("clusters:\n  vader:\n    adress: http://vader:8265\n")

    with pytest.raises(ClusterConfigError, match=r"did you mean 'address'\?"):
        load_clusters(path)


def test_the_shipped_cluster_example_validates():
    from ai_experiments.clusters import load_clusters

    assert load_clusters(EXAMPLES / "clusters.yaml")
