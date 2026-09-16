# Circuito honesto — Implementation Plan (fases 0–1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hacer que "el agente mejoró la métrica" sea una afirmación creíble: un candidato no puede fabricar su propio resultado ni leakear el test set, y una campaña autónoma supervisa lo que ejecuta.

**Architecture:** Se separa el canal que puntúa (`IAX_RESULT`) del canal de progreso (`IAX_METRIC`), y la ejecución de un trial pasa de un proceso a dos fases —`train` y `evaluate`— corridas por un lanzador común. Solo la fase `evaluate` puede producir un resultado, y solo ella recibe una referencia al test set. El mismo lanzador sirve al backend local y a Ray, y el loop autónomo pasa a supervisar con el mismo tick que el daemon.

**Tech Stack:** Python ≥3.10, pydantic v2, pytest, Ray Job Submission API, subprocess.

**Spec:** [docs/superpowers/specs/2026-09-15-campanas-de-mejora-verificables-design.md](../specs/2026-09-15-campanas-de-mejora-verificables-design.md)

## Global Constraints

- Python `>=3.10` (`pyproject.toml:6`). `shlex.join` está disponible.
- Los workloads no dependen del paquete: imprimir la línea de stdout directamente es contrato soportado (`report.py` docstring). Toda decisión debe funcionar para un workload que solo hace `print`.
- Prompts y salida de campaña son **no confiables**: nunca se interpolan en un string de shell.
- El prefijo de resultado es exactamente `IAX_RESULT ` (con espacio final), en paralelo a `METRIC_PREFIX = "IAX_METRIC "`.
- **Ruptura dura:** un workload sin `report_result` no puntúa. Prohibido el fallback al último `IAX_METRIC`.
- Los tests se nombran como frases descriptivas (`test_a_failed_trial_never_wins`), siguiendo el estilo de `tests/test_variants.py`.
- Marca `integration` reservada para servicios reales (`pyproject.toml:45`). Nada de este plan la usa.
- Suite de referencia: `.venv/bin/python -m pytest -q -m 'not integration' --ignore=tests/test_server.py` debe quedar verde al cerrar cada tarea.

## File Structure

| Archivo | Responsabilidad |
|---|---|
| `ai_experiments/report.py` (modificar) | Contrato workload→arnés: progreso y resultado, emisión y parseo |
| `ai_experiments/phases.py` (crear) | Lanzador de dos fases: corre `train`, luego `evaluate`, con entornos distintos |
| `ai_experiments/worker.py` (modificar) | Supervisa **una** fase; enruta `IAX_RESULT` según cuál sea |
| `ai_experiments/schemas.py` (modificar) | `ResultRecord`, `WorkloadSpec` con fases, `DataSpec` |
| `ai_experiments/store/filesystem.py` (modificar) | Canal `results.jsonl`, separado de `metrics.jsonl` |
| `ai_experiments/planner/analysis.py` (modificar) | Puntúa solo resultados; `no_result` como causa |
| `ai_experiments/backends/ray.py` (modificar) | argv exacto; entrypoint al lanzador |
| `ai_experiments/backends/local.py` (modificar) | Invoca el lanzador en vez de un worker único |
| `ai_experiments/loop.py` (modificar) | Supervisa con el tick común; cierra cohorte antes de admitir |

---

### Task 1: Base canónica (fase 0)

No es trabajo TDD: es verificación e integración de ramas existentes. El resultado es una rama sobre la que el resto del plan se apoya.

> **Hecho (2026-09-15).** El inventario por archivo que describen los pasos 1–3 no hizo falta: `c4f0453` es el merge-base más un commit, y `fae2c00` es un superconjunto estricto del resto. La base canónica es `fae2c00` + cherry-pick de `c4f0453`, en la rama `feat/honest-circuit`. Detalle en [docs/integration-matrix-2026-09-15.md](../../integration-matrix-2026-09-15.md).

**Files:**
- Create: `docs/integration-matrix-2026-09-15.md`
- Branch: `feat/honest-circuit` desde la base canónica resultante

**Interfaces:**
- Produces: una rama donde `planner/analysis.py` expone `ObjectiveReading` (con `miss_reason`) y `best_of(trials, mode)` filtrando `status == "completed"`; `improve/variants.py` y `agents/` presentes. Las tareas 5 en adelante lo asumen.

- [x] **Step 1: Inventariar capacidades por commit**

Las ramas no son versiones sucesivas: `c4f0453` no contiene todo lo de `fae2c00`. Para cada archivo, registrar qué commit tiene la versión buena.

```bash
git diff --stat fae2c00 c4f0453 -- ai_experiments
git show fae2c00:ai_experiments/store/filesystem.py | grep -n "lock\|flock"
git show fae2c00:ai_experiments/backends/ray.py | grep -n "preflight\|resources"
git show c4f0453:ai_experiments/planner/analysis.py | grep -n "def best_of\|miss_reason"
git show c4f0453:ai_experiments/loop.py | grep -n "reconcile"
```

- [x] **Step 2: Escribir la matriz**

Crear `docs/integration-matrix-2026-09-15.md` con una fila por capacidad y una columna por commit. Mínimo a cubrir: locks por run, preflight de submit, fixes de worker, `ObjectiveReading`/`miss_reason`, `best_of` elegible, validación de params, agotamiento de espacio discreto, reconciliación del último lote, `improve/variants.py`, `agents/`. Cada fila dice qué commit gana y por qué.

- [x] **Step 3: Construir la rama canónica**

```bash
git checkout -b feat/honest-circuit c4f0453
git checkout fae2c00 -- <los archivos que la matriz asigna a fae2c00>
```

No fusionar a ciegas: traer archivo por archivo según la matriz.

- [x] **Step 4: Verificar que la base pasa**

```bash
.venv/bin/python -m pytest -q -m 'not integration' --ignore=tests/test_server.py
```
Expected: PASS. Si un test falla, la matriz asignó mal un archivo — corregirla antes de seguir.

- [x] **Step 5: Commit**

```bash
git add docs/integration-matrix-2026-09-15.md
git commit -m "docs: integration matrix for the canonical base"
```

---

### Task 2: El canal de resultado

**Files:**
- Modify: `ai_experiments/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Produces: `RESULT_PREFIX: str`, `report_result(**values: float) -> None`, `parse_result_line(line: str) -> dict[str, float] | None`. `parse_result_line` devuelve el dict de valores (sin `step`: un resultado no tiene paso) o `None` si la línea no es un resultado o no trae valores usables.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py — agregar al final
from ai_experiments.report import parse_result_line, report_result


def test_parses_a_result_line():
    parsed = parse_result_line('IAX_RESULT {"test_acc": 0.91}')

    assert parsed == {"test_acc": 0.91}


def test_a_metric_line_is_not_a_result():
    assert parse_result_line('IAX_METRIC {"loss": 0.5}') is None


def test_a_result_line_is_not_a_metric():
    from ai_experiments.report import parse_metric_line

    assert parse_metric_line('IAX_RESULT {"test_acc": 0.91}') is None


def test_a_result_ignores_step_because_it_has_no_progress():
    parsed = parse_result_line('IAX_RESULT {"step": 7, "test_acc": 0.91}')

    assert parsed == {"test_acc": 0.91}


def test_a_result_with_no_numeric_values_is_not_a_result():
    assert parse_result_line('IAX_RESULT {"note": "done"}') is None


def test_report_result_prints_the_contract_line(capsys):
    report_result(test_acc=0.91)

    assert capsys.readouterr().out.strip() == 'IAX_RESULT {"test_acc": 0.91}'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report.py -q`
Expected: FAIL con `ImportError: cannot import name 'parse_result_line'`.

- [ ] **Step 3: Write minimal implementation**

`parse_metric_line` hoy usa `stripped.find(...)`, que encontraría `IAX_METRIC` dentro de otra línea. Los dos prefijos son distintos, así que no colisionan, pero el resultado debe descartar `step` explícitamente.

```python
# ai_experiments/report.py
RESULT_PREFIX = "IAX_RESULT "


def report_result(**values: float) -> None:
    """Print the one evaluation result the harness will score.

    Only the ``evaluate`` phase may call this: a result reported from the
    training phase is discarded. Progress belongs in :func:`report_metric`,
    which never scores.
    """
    sys.stdout.write(RESULT_PREFIX + json.dumps(dict(values)) + "\n")
    sys.stdout.flush()


def parse_result_line(line: str) -> dict[str, float] | None:
    """Parse an ``IAX_RESULT {...}`` line into numeric values, or None.

    A result has no step: it is the answer, not a point on a curve.
    """
    stripped = line.strip()
    idx = stripped.find(RESULT_PREFIX.strip())
    if idx == -1:
        return None
    raw = stripped[idx + len(RESULT_PREFIX.strip()) :].strip()
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    payload.pop("step", None)

    values: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values[str(key)] = float(value)
        elif isinstance(value, str):
            lowered = value.lower()
            if lowered in {"nan", "inf", "-inf", "infinity", "-infinity"}:
                values[str(key)] = float(lowered.replace("infinity", "inf"))
    return values or None
```

Actualizar el docstring del módulo para nombrar los dos canales y decir cuál puntúa.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_report.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/report.py tests/test_report.py
git commit -m "feat: add IAX_RESULT, the only channel that scores"
```

---

### Task 3: El store guarda resultados aparte

**Files:**
- Modify: `ai_experiments/schemas.py`, `ai_experiments/store/filesystem.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: nada de tareas previas.
- Produces: `ResultRecord(timestamp: datetime, values: dict[str, float])` en `schemas.py`; `FilesystemRunStore.results_path(run_id) -> Path`, `.append_result(run_id, record: ResultRecord) -> None`, `.read_results(run_id) -> list[ResultRecord]`. El archivo es `results.jsonl`, hermano de `metrics.jsonl`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py — agregar al final
from ai_experiments.schemas import ResultRecord


def test_results_live_in_their_own_channel(tmp_path):
    store = FilesystemRunStore(tmp_path)
    run_id, _ = store.create_run(_manifest())

    store.append_metric(run_id, MetricPoint(step=1, values={"loss": 0.5}))
    store.append_result(run_id, ResultRecord(values={"test_acc": 0.9}))

    assert [p.values for p in store.read_metrics(run_id)] == [{"loss": 0.5}]
    assert [r.values for r in store.read_results(run_id)] == [{"test_acc": 0.9}]


def test_reading_results_of_a_run_that_reported_none(tmp_path):
    store = FilesystemRunStore(tmp_path)
    run_id, _ = store.create_run(_manifest())

    assert store.read_results(run_id) == []
```

`_manifest()` es el helper que ya usa `tests/test_store.py`; reusarlo tal cual está en ese archivo.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_store.py -q`
Expected: FAIL con `ImportError: cannot import name 'ResultRecord'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ai_experiments/schemas.py — junto a MetricPoint
class ResultRecord(BaseModel):
    """The evaluation result of a run: the number that may be scored.

    Separate from :class:`MetricPoint` on purpose. A metric is a point on a
    curve and picking its best value is a biased estimator; a result is what
    the protected evaluator declared, and there is at most one that counts.
    """

    timestamp: datetime = Field(default_factory=utc_now)
    values: dict[str, float] = Field(default_factory=dict)
```

```python
# ai_experiments/store/filesystem.py — junto a los métodos de métricas
    def results_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "results.jsonl"

    def append_result(self, run_id: str, record: ResultRecord) -> None:
        with self.results_path(run_id).open("a") as fh:
            fh.write(json.dumps(record.model_dump(mode="json")) + "\n")

    def read_results(self, run_id: str) -> list[ResultRecord]:
        path = self.results_path(run_id)
        if not path.exists():
            return []
        return [
            ResultRecord(**json.loads(line))
            for line in path.read_text().splitlines()
            if line.strip()
        ]
```

Agregar `ResultRecord` al import de `schemas` en `filesystem.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/schemas.py ai_experiments/store/filesystem.py tests/test_store.py
git commit -m "feat: store evaluation results in their own channel"
```

---

### Task 4: El worker enruta el resultado según la fase

**Files:**
- Modify: `ai_experiments/worker.py`
- Test: `tests/test_worker_phases.py` (crear)

**Interfaces:**
- Consumes: `parse_result_line` (Task 2), `ResultRecord` + `store.append_result` (Task 3).
- Produces: `_Supervisor(store, run_id, phase: str)` y el flag CLI `--phase` con valores `train` y `evaluate` (default `evaluate`, para que un manifest de una sola fase siga puntuando). En fase `train` un `IAX_RESULT` **no** se guarda: se registra un `RunEvent` de nivel `warning` con mensaje `result reported from the train phase; discarded`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worker_phases.py
"""The training phase cannot declare a result. Only the evaluator can."""

from __future__ import annotations

import sys
import textwrap

from ai_experiments.schemas import ExperimentManifest, WorkloadSpec
from ai_experiments.store import FilesystemRunStore
from ai_experiments.worker import _Supervisor


def _run(tmp_path, body: str, phase: str):
    script = tmp_path / "workload.py"
    script.write_text(textwrap.dedent(body))
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="e",
        workload=WorkloadSpec(
            entrypoint=sys.executable,
            args=[str(script)],
            working_dir=str(tmp_path),
        ),
    )
    run_id, _ = store.create_run(manifest)
    _Supervisor(store, run_id, phase=phase).run()
    return store, run_id


CHEATS = """
    print('IAX_METRIC {"step": 0, "loss": 0.5}')
    print('IAX_RESULT {"test_acc": 0.99}')
    """


def test_the_train_phase_cannot_declare_a_result(tmp_path):
    store, run_id = _run(tmp_path, CHEATS, phase="train")

    assert store.read_results(run_id) == []
    assert [p.values["loss"] for p in store.read_metrics(run_id)] == [0.5]


def test_a_result_from_the_train_phase_is_recorded_as_a_warning(tmp_path):
    store, run_id = _run(tmp_path, CHEATS, phase="train")

    warnings = [e for e in store.read_events(run_id) if e.level == "warning"]
    assert any("discarded" in e.message for e in warnings)


def test_the_evaluate_phase_declares_the_result(tmp_path):
    store, run_id = _run(tmp_path, CHEATS, phase="evaluate")

    assert [r.values for r in store.read_results(run_id)] == [{"test_acc": 0.99}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_worker_phases.py -q`
Expected: FAIL con `TypeError: _Supervisor.__init__() got an unexpected keyword argument 'phase'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ai_experiments/worker.py
from ai_experiments.report import parse_metric_line, parse_result_line
from ai_experiments.schemas import ResultRecord


class _Supervisor:
    def __init__(
        self, store: FilesystemRunStore, run_id: str, phase: str = "evaluate"
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.phase = phase
        ...
```

En el bucle de stdout, antes de intentar la métrica:

```python
        for line in self.process.stdout:
            result = parse_result_line(line)
            if result is not None:
                if self.phase == "train":
                    # The trainer is the code an agent may rewrite. Letting it
                    # declare the number it is judged by would make the metric
                    # the cheapest thing in the search space to optimize.
                    self.store.append_event(
                        self.run_id,
                        RunEvent(
                            level="warning",
                            message="result reported from the train phase; discarded",
                            details={"values": result},
                        ),
                    )
                else:
                    self.store.append_result(
                        self.run_id, ResultRecord(values=result)
                    )
                    self._update_status(details={"result": result})
                continue

            metric = parse_metric_line(line)
            ...
```

En `main()`, agregar el flag y pasarlo:

```python
    parser.add_argument("--phase", default="evaluate", choices=["train", "evaluate"])
    ...
    _Supervisor(store, args.run_id, phase=args.phase).run()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_worker_phases.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/worker.py tests/test_worker_phases.py
git commit -m "feat: only the evaluate phase may declare a result"
```

---

### Task 5: La puntuación lee resultados, no curvas

**Files:**
- Modify: `ai_experiments/planner/analysis.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- Consumes: `store.read_results` (Task 3); `ObjectiveReading` y `best_of` de la base canónica (Task 1).
- Produces: `extract_objective(store, run_id, objective) -> ObjectiveReading` que lee **solo** resultados. `ObjectiveReading.miss_reason` gana el valor `"no_result"`. `observed_metrics` pasa a listar los nombres de resultado observados.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_planner.py — agregar al final
from ai_experiments.planner.analysis import extract_objective
from ai_experiments.schemas import MetricPoint, ObjectiveSpec, ResultRecord


def _run_with(store, metrics, results):
    run_id, _ = store.create_run(_manifest())
    for point in metrics:
        store.append_metric(run_id, point)
    for record in results:
        store.append_result(run_id, record)
    return run_id


def test_a_progress_curve_alone_scores_nothing(tmp_path):
    store = FilesystemRunStore(tmp_path)
    run_id = _run_with(
        store,
        [MetricPoint(step=i, values={"loss": v}) for i, v in enumerate([0.9, 0.001, 0.8])],
        [],
    )

    reading = extract_objective(store, run_id, ObjectiveSpec(metric="loss", mode="min"))

    assert reading.value is None
    assert reading.miss_reason == "no_result"


def test_the_declared_result_is_the_score(tmp_path):
    store = FilesystemRunStore(tmp_path)
    run_id = _run_with(
        store,
        [MetricPoint(step=0, values={"loss": 0.001})],
        [ResultRecord(values={"loss": 0.42})],
    )

    reading = extract_objective(store, run_id, ObjectiveSpec(metric="loss", mode="min"))

    assert reading.value == 0.42


def test_a_result_without_the_objective_metric_says_which_ones_it_had(tmp_path):
    store = FilesystemRunStore(tmp_path)
    run_id = _run_with(store, [], [ResultRecord(values={"test_acc": 0.9})])

    reading = extract_objective(store, run_id, ObjectiveSpec(metric="auroc", mode="max"))

    assert reading.miss_reason == "metric_absent"
    assert reading.observed_metrics == ["test_acc"]


def test_a_non_finite_result_is_not_a_score(tmp_path):
    store = FilesystemRunStore(tmp_path)
    run_id = _run_with(store, [], [ResultRecord(values={"loss": float("nan")})])

    reading = extract_objective(store, run_id, ObjectiveSpec(metric="loss", mode="min"))

    assert reading.value is None
    assert reading.miss_reason == "not_finite"


def test_a_failed_trial_never_wins():
    """Regression guard for the eligibility fix Task 1 brought in.

    A run that reported a result and then crashed used to be the best trial
    in the campaign, because `best_trial` ranked on the value alone.
    """
    from ai_experiments.planner.analysis import best_of
    from ai_experiments.schemas import TrialRecord

    trials = [
        TrialRecord(trial_id="t000", params={}, status="failed", objective_value=0.01),
        TrialRecord(
            trial_id="t001", params={}, status="completed", objective_value=0.50
        ),
    ]

    assert best_of(trials, "min").trial_id == "t001"
```

`_manifest()` es el helper local del archivo; si `tests/test_planner.py` no lo tiene, copiar el de `tests/test_store.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_planner.py -q`
Expected: FAIL — `extract_objective` sigue leyendo métricas y devuelve `0.001`.

- [ ] **Step 3: Write minimal implementation**

```python
# ai_experiments/planner/analysis.py
def extract_objective(
    store: FilesystemRunStore, run_id: str, objective: ObjectiveSpec
) -> ObjectiveReading:
    """The result the run declared, or why there is none.

    Only ``IAX_RESULT`` scores. Aggregating a progress curve — the old
    ``min(values)`` — rewarded whichever trial rolled the dice most times,
    and rewarded a crashed run for one lucky step before it died.
    """
    results = store.read_results(run_id)
    if not results:
        return ObjectiveReading(miss_reason="no_result")

    values: dict[str, float] = {}
    for record in results:
        values.update(record.values)
    observed = sorted(values)

    if objective.metric not in values:
        return ObjectiveReading(
            final_metrics=values,
            observed_metrics=observed,
            miss_reason="metric_absent",
        )
    value = values[objective.metric]
    if not math.isfinite(value):
        return ObjectiveReading(
            final_metrics=values,
            observed_metrics=observed,
            miss_reason="not_finite",
        )
    return ObjectiveReading(
        value=value, final_metrics=values, observed_metrics=observed
    )
```

Agregar `"no_result"` al `Literal` de `ObjectiveReading.miss_reason` y su rama en `miss_message`:

```python
        if self.miss_reason == "no_result":
            return (
                "no result reported: the workload printed no IAX_RESULT line, "
                f"so objective '{metric}' could not be scored"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_planner.py tests/test_orchestrator.py -q`
Expected: PASS. Si `tests/test_orchestrator.py` o `tests/test_e2e_campaign.py` fallan, es porque sus fixtures emiten solo `IAX_METRIC`: actualizarlos para que además emitan `IAX_RESULT` — esa migración es parte de esta tarea, no un daño colateral.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/planner/analysis.py tests/
git commit -m "feat: score the declared result, never the progress curve"
```

---

### Task 6: argv exacto en Ray

**Files:**
- Modify: `ai_experiments/backends/ray.py:69`
- Test: `tests/test_ray_backend.py`

**Interfaces:**
- Produces: el entrypoint enviado a `submit_job` preserva los límites de cada argumento.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ray_backend.py — agregar al final
def test_an_argument_with_spaces_stays_one_argument(tmp_path):
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_1"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(
            entrypoint="python train.py",
            args=["--label", "hello world"],
            working_dir=str(tmp_path),
        ),
    )

    backend.submit(manifest)

    assert captured["entrypoint"] == "python train.py --label 'hello world'"
```

Reusar el `client_factory` falso que `tests/test_ray_backend.py` ya define si su forma coincide; si difiere, quedarse con el de arriba.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ray_backend.py -q`
Expected: FAIL — el entrypoint sale `python train.py --label hello world`, que el shell parte en dos argumentos.

- [ ] **Step 3: Write minimal implementation**

```python
# ai_experiments/backends/ray.py
import shlex

        # Ray takes a shell string, so every argument has to survive the
        # shell's own word splitting. `" ".join` did not: an argument with a
        # space arrived at the workload as two.
        entrypoint = " ".join(
            [manifest.workload.entrypoint, shlex.join(manifest.workload.args)]
        ).strip()
```

`manifest.workload.entrypoint` se deja tal cual: es configuración del operador y puede traer flags propios (`python -m mod`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ray_backend.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/backends/ray.py tests/test_ray_backend.py
git commit -m "fix: keep argument boundaries when building the Ray entrypoint"
```

---

### Task 7: El workload declara sus fases y sus datos

**Files:**
- Modify: `ai_experiments/schemas.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces:
  - `WorkloadSpec.train: str | None` y `WorkloadSpec.evaluate: str | None`, junto al `entrypoint` existente.
  - `WorkloadSpec.phases() -> list[tuple[str, str]]`: `[("train", cmd), ("evaluate", cmd)]` cuando ambos están declarados; `[("evaluate", entrypoint)]` cuando solo hay `entrypoint`.
  - `DataSpec(train: str | None, val: str | None, test: str | None)` y `WorkloadSpec.data: DataSpec`.
  - `DataSpec.env_for(phase: str) -> dict[str, str]`: en `train` exporta `IAX_DATA_TRAIN`/`IAX_DATA_VAL` y **nunca** `IAX_DATA_TEST`; en `evaluate` exporta los tres.
- Validación: declarar `evaluate` sin `train` es un error; declarar `data.test` sin `evaluate` es un error (no habría fase protegida que lo reciba).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py — agregar al final
import pytest

from ai_experiments.schemas import DataSpec, WorkloadSpec


def test_a_single_entrypoint_workload_runs_as_the_evaluate_phase():
    workload = WorkloadSpec(entrypoint="python toy.py")

    assert workload.phases() == [("evaluate", "python toy.py")]


def test_a_two_phase_workload_runs_train_then_evaluate():
    workload = WorkloadSpec(
        entrypoint="python train.py",
        train="python train.py",
        evaluate="python evaluate.py",
    )

    assert workload.phases() == [
        ("train", "python train.py"),
        ("evaluate", "python evaluate.py"),
    ]


def test_the_train_phase_never_receives_the_test_reference():
    data = DataSpec(train="data/train", val="data/val", test="data/test")

    env = data.env_for("train")

    assert env == {"IAX_DATA_TRAIN": "data/train", "IAX_DATA_VAL": "data/val"}
    assert "IAX_DATA_TEST" not in env


def test_the_evaluate_phase_receives_the_test_reference():
    data = DataSpec(train="data/train", val="data/val", test="data/test")

    assert data.env_for("evaluate")["IAX_DATA_TEST"] == "data/test"


def test_declaring_evaluate_without_train_is_rejected():
    with pytest.raises(ValueError, match="train"):
        WorkloadSpec(entrypoint="python x.py", evaluate="python evaluate.py")


def test_declaring_test_data_without_an_evaluate_phase_is_rejected():
    with pytest.raises(ValueError, match="evaluate"):
        WorkloadSpec(entrypoint="python x.py", data=DataSpec(test="data/test"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -q`
Expected: FAIL con `ImportError: cannot import name 'DataSpec'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ai_experiments/schemas.py — reemplazar WorkloadSpec
class DataSpec(BaseModel):
    """Where the workload's data lives, by reference.

    The boundary this draws is the only thing that stops a generated feature
    from leaking the test set: a training process that has no reference to
    test cannot read it, however the feature code is written.
    """

    train: str | None = None
    val: str | None = None
    test: str | None = None

    def env_for(self, phase: str) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.train:
            env["IAX_DATA_TRAIN"] = self.train
        if self.val:
            env["IAX_DATA_VAL"] = self.val
        if phase == "evaluate" and self.test:
            env["IAX_DATA_TEST"] = self.test
        return env


class WorkloadSpec(BaseModel):
    """Executable workload for a training experiment.

    A workload may declare one command (``entrypoint``) or two (``train`` and
    ``evaluate``). Two is what lets the harness protect the evaluator: only
    the evaluate phase may declare a result, and only it sees the test data.
    """

    entrypoint: str
    args: list[str] = Field(default_factory=list)
    working_dir: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    train: str | None = None
    evaluate: str | None = None
    data: DataSpec = Field(default_factory=DataSpec)

    @model_validator(mode="after")
    def phases_are_coherent(self) -> WorkloadSpec:
        if self.evaluate and not self.train:
            raise ValueError(
                "declaring 'evaluate' requires 'train': with only an evaluate "
                "phase there is nothing for it to score"
            )
        if self.data.test and not self.evaluate:
            raise ValueError(
                "data.test requires an 'evaluate' phase: without one there is "
                "no protected phase to receive the test reference"
            )
        return self

    def phases(self) -> list[tuple[str, str]]:
        if self.train and self.evaluate:
            return [("train", self.train), ("evaluate", self.evaluate)]
        return [("evaluate", self.entrypoint)]
```

`model_validator` ya se importa en `schemas.py` (lo usa `BudgetSpec`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_manifest.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/schemas.py tests/test_manifest.py
git commit -m "feat: let a workload declare train/evaluate phases and its data"
```

---

### Task 8: El lanzador de fases

**Files:**
- Create: `ai_experiments/phases.py`
- Modify: `ai_experiments/backends/local.py:53-79`
- Test: `tests/test_phases.py` (crear)

**Interfaces:**
- Consumes: `WorkloadSpec.phases()` y `DataSpec.env_for` (Task 7); `_Supervisor(store, run_id, phase=...)` (Task 4).
- Produces: `run_phases(store: FilesystemRunStore, run_id: str) -> int`, que corre cada fase en orden y devuelve el exit code de la primera que falla (0 si todas pasan). Si `train` falla, `evaluate` **no** corre. El módulo es ejecutable: `python -m ai_experiments.phases --run-id <id> --runs-dir <dir>`.
- Produces: `IAX_WORK_DIR`, un directorio `<run_dir>/work/` que ambas fases reciben. Es el único canal de handoff entre train y evaluate: train deja ahí su modelo, evaluate lo lee de ahí. **No** se usa el cwd, porque en la fase 2 del spec las dos fases dejan de compartirlo.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phases.py
"""Two phases, run in order, with the evaluator's result the only score."""

from __future__ import annotations

import sys
import textwrap

from ai_experiments.phases import run_phases
from ai_experiments.schemas import ExperimentManifest, WorkloadSpec
from ai_experiments.store import FilesystemRunStore


def _store_with(tmp_path, train_body: str, evaluate_body: str):
    (tmp_path / "train.py").write_text(textwrap.dedent(train_body))
    (tmp_path / "evaluate.py").write_text(textwrap.dedent(evaluate_body))
    # NOTE: the bodies below hand off through IAX_WORK_DIR, never the cwd.
    store = FilesystemRunStore(tmp_path / "runs")
    manifest = ExperimentManifest(
        experiment="e",
        workload=WorkloadSpec(
            entrypoint=f"{sys.executable} train.py",
            train=f"{sys.executable} train.py",
            evaluate=f"{sys.executable} evaluate.py",
            working_dir=str(tmp_path),
        ),
    )
    run_id, _ = store.create_run(manifest)
    return store, run_id


def test_the_evaluator_result_is_the_one_that_counts(tmp_path):
    store, run_id = _store_with(
        tmp_path,
        """
        print('IAX_METRIC {"step": 0, "loss": 0.1}')
        print('IAX_RESULT {"test_acc": 0.99}')
        """,
        """
        print('IAX_RESULT {"test_acc": 0.61}')
        """,
    )

    assert run_phases(store, run_id) == 0
    assert [r.values for r in store.read_results(run_id)] == [{"test_acc": 0.61}]


def test_a_failing_train_phase_stops_the_evaluation(tmp_path):
    store, run_id = _store_with(
        tmp_path,
        """
        import sys
        sys.exit(3)
        """,
        """
        print('IAX_RESULT {"test_acc": 0.61}')
        """,
    )

    assert run_phases(store, run_id) == 3
    assert store.read_results(run_id) == []


def test_both_phases_share_one_handoff_directory(tmp_path):
    store, run_id = _store_with(
        tmp_path,
        """
        import os, pathlib
        pathlib.Path(os.environ["IAX_WORK_DIR"], "model.txt").write_text("7")
        """,
        """
        import os, pathlib
        x = pathlib.Path(os.environ["IAX_WORK_DIR"], "model.txt").read_text()
        print('IAX_RESULT {"loss": %s}' % x)
        """,
    )

    assert run_phases(store, run_id) == 0
    assert [r.values for r in store.read_results(run_id)] == [{"loss": 7.0}]


def test_a_single_entrypoint_workload_still_runs(tmp_path):
    store = FilesystemRunStore(tmp_path / "runs")
    script = tmp_path / "toy.py"
    script.write_text('print(\'IAX_RESULT {"loss": 0.25}\')\n')
    manifest = ExperimentManifest(
        experiment="e",
        workload=WorkloadSpec(
            entrypoint=f"{sys.executable} toy.py", working_dir=str(tmp_path)
        ),
    )
    run_id, _ = store.create_run(manifest)

    assert run_phases(store, run_id) == 0
    assert [r.values for r in store.read_results(run_id)] == [{"loss": 0.25}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_phases.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'ai_experiments.phases'`.

- [ ] **Step 3: Write minimal implementation**

`_Supervisor` hoy lee el manifest y fija el estado terminal del run. Con dos fases, solo la última puede declarar el estado terminal. Se agrega `final: bool` al supervisor: cuando es `False`, no escribe `completed` al terminar bien (sí escribe `failed`/`cancelled`, porque una fase rota termina el run).

```python
# ai_experiments/phases.py
"""Run a workload's phases in order, with the evaluator kept separate.

A trial is not one process. The trainer is the code an agent may rewrite; the
evaluator is the code that judges it. Running them as one process would put
the number and the thing being measured in the same hands, so they run as two,
and only the second one may declare a result.
"""

from __future__ import annotations

import argparse

from ai_experiments.schemas import ExperimentManifest, RunEvent
from ai_experiments.store import FilesystemRunStore
from ai_experiments.worker import _Supervisor


def run_phases(store: FilesystemRunStore, run_id: str) -> int:
    """Run every declared phase in order; stop at the first failure."""
    manifest = ExperimentManifest.from_yaml(store.run_dir(run_id) / "manifest.yaml")
    # The two phases hand off through this directory and not through the cwd:
    # once the trainer runs from a variant copy, they no longer share one.
    work_dir = store.run_dir(run_id) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    phases = manifest.workload.phases()
    for index, (phase, command) in enumerate(phases):
        store.append_event(
            run_id, RunEvent(message="phase started", details={"phase": phase})
        )
        supervisor = _Supervisor(
            store, run_id, phase=phase, final=index == len(phases) - 1
        )
        exit_code = supervisor.run(command=command, work_dir=work_dir)
        if exit_code != 0:
            return exit_code
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", required=True)
    args = parser.parse_args()
    raise SystemExit(run_phases(FilesystemRunStore(args.runs_dir), args.run_id))


if __name__ == "__main__":
    main()
```

En `worker.py`, `_Supervisor.run` acepta el comando de la fase y su entorno, y devuelve el exit code:

```python
    def __init__(self, store, run_id, phase="evaluate", final=True) -> None:
        ...
        self.final = final

    def run(self, command: str | None = None, work_dir: Path | None = None) -> int:
        ...
        entrypoint = command or manifest.workload.entrypoint
        cmd = [*shlex.split(entrypoint), *manifest.workload.args]
        ...
        env.update(manifest.workload.data.env_for(self.phase))
        env["IAX_PHASE"] = self.phase
        if work_dir is not None:
            env["IAX_WORK_DIR"] = str(work_dir)
        ...
        if exit_code == 0:
            if self.final:
                self._update_status(
                    status="completed", exit_code=exit_code, completed_at=utc_now()
                )
            ...
        return exit_code
```

En `backends/local.py`, el subprocess pasa a lanzar el módulo de fases:

```python
        cmd = [
            sys.executable,
            "-m",
            "ai_experiments.phases",
            "--run-id",
            run_id,
            "--runs-dir",
            str(self.store.root),
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_phases.py tests/test_worker_phases.py tests/test_e2e_campaign.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/phases.py ai_experiments/worker.py ai_experiments/backends/local.py tests/test_phases.py
git commit -m "feat: run a trial as train then evaluate, not one process"
```

---

### Task 9: Ray corre las mismas fases

**Files:**
- Modify: `ai_experiments/backends/ray.py`, `ai_experiments/worker.py`
- Test: `tests/test_ray_backend.py`

**Interfaces:**
- Consumes: `run_phases` (Task 8), `shlex.join` del entrypoint (Task 6).
- Produces: cuando el workload declara dos fases, el entrypoint enviado a Ray encadena ambas en orden con `&&`, cada una con su `IAX_PHASE` y su entorno de datos; `runtime_env["env_vars"]` incluye las variables `IAX_*` que el worker local ya inyecta.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ray_backend.py — agregar al final
def test_a_two_phase_workload_chains_both_phases_remotely(tmp_path):
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_2"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(
            entrypoint="python train.py",
            train="python train.py",
            evaluate="python evaluate.py",
            working_dir=str(tmp_path),
            data=DataSpec(train="data/train", test="data/test"),
        ),
    )

    backend.submit(manifest)

    entrypoint = captured["entrypoint"]
    assert "python train.py" in entrypoint
    assert "python evaluate.py" in entrypoint
    assert entrypoint.index("train.py") < entrypoint.index("evaluate.py")
    assert "&&" in entrypoint


def test_the_remote_run_carries_the_harness_variables(tmp_path):
    captured = {}

    class _Client:
        def submit_job(self, **kwargs):
            captured.update(kwargs)
            return "raysubmit_3"

    backend = RayBackend(
        address="http://fake:8265",
        store=FilesystemRunStore(tmp_path),
        client_factory=lambda _address: _Client(),
    )
    manifest = ExperimentManifest(
        experiment="e",
        backend="ray",
        workload=WorkloadSpec(entrypoint="python toy.py", working_dir=str(tmp_path)),
    )

    handle = backend.submit(manifest)

    assert captured["runtime_env"]["env_vars"]["IAX_RUN_ID"] == handle.run_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ray_backend.py -q`
Expected: FAIL — el entrypoint trae una sola fase y `env_vars` no tiene `IAX_RUN_ID`.

- [ ] **Step 3: Write minimal implementation**

```python
# ai_experiments/backends/ray.py — reemplazar la construcción del entrypoint
        args = shlex.join(manifest.workload.args)
        # The handoff directory is created once, inside the job's working dir,
        # and both phases see the same absolute path.
        commands = ["mkdir -p iax_work", "export IAX_WORK_DIR=$PWD/iax_work"]
        for phase, command in manifest.workload.phases():
            data_env = manifest.workload.data.env_for(phase)
            prefix = " ".join(
                f"{name}={shlex.quote(value)}"
                for name, value in sorted({**data_env, "IAX_PHASE": phase}.items())
            )
            commands.append(f"{prefix} {command} {args}".strip())
        # `&&` and not `;`: a training phase that failed must not be followed
        # by an evaluation that would score whatever was left behind.
        entrypoint = " && ".join(commands)
```

Y en `runtime_env`:

```python
        runtime_env = {
            "working_dir": str(Path(manifest.workload.working_dir).resolve()),
            "env_vars": {
                **manifest.workload.env,
                **tracking_env,
                "IAX_RUN_ID": run_id,
            },
        }
```

`IAX_ARTIFACTS_DIR` queda fuera a propósito: en Ray los artefactos viven en el almacenamiento del cluster, y el transporte es fase 4 del spec. El handoff train→evaluate usa el working dir del job, que es el mismo para ambas fases por correr en el mismo job.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ray_backend.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/backends/ray.py tests/test_ray_backend.py
git commit -m "feat: run both phases on Ray, with the harness variables present"
```

---

### Task 10: Los ejemplos declaran resultado y fases

**Files:**
- Modify: `examples/toy_train.py`, `examples/goal_toy.yaml`
- Create: `examples/toy_evaluate.py`
- Test: `tests/test_e2e_campaign.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: un ejemplo de dos fases que sirve de fixture al contrato. `toy_train.py` deja de imprimir el resultado y escribe `model.json` en el working dir; `toy_evaluate.py` lo lee y declara `loss`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e2e_campaign.py — agregar al final
def test_the_shipped_example_scores_only_from_its_evaluator(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    examples = Path(__file__).resolve().parents[1] / "examples"
    work = tmp_path / "work"
    work.mkdir()
    env = {**os.environ, "IAX_WORK_DIR": str(work)}

    train = subprocess.run(
        [sys.executable, str(examples / "toy_train.py"), "--steps", "3", "--sleep", "0"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert train.returncode == 0
    assert "IAX_RESULT" not in train.stdout
    assert "IAX_METRIC" in train.stdout

    evaluate = subprocess.run(
        [sys.executable, str(examples / "toy_evaluate.py")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert evaluate.returncode == 0
    assert "IAX_RESULT" in evaluate.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_e2e_campaign.py -q`
Expected: FAIL — `toy_evaluate.py` no existe.

- [ ] **Step 3: Write minimal implementation**

En `examples/toy_train.py`, reemplazar el bloque de artefactos por la escritura del modelo en el directorio de handoff, y dejar el `print` final sin resultado:

```python
    # The trainer produces an artifact. It does not declare a result: the
    # number it would be judged by is not its to report.
    work = Path(os.environ.get("IAX_WORK_DIR", "."))
    with (work / "model.json").open("w") as fh:
        json.dump({"x": x}, fh)

    print(f"final x={x:.4f}")
```

```python
# examples/toy_evaluate.py
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
```

En `examples/goal_toy.yaml`, el workload declara ambas fases:

```yaml
workload:
  entrypoint: python examples/toy_train.py
  train: python examples/toy_train.py
  evaluate: python examples/toy_evaluate.py
  working_dir: .
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -q -m 'not integration' --ignore=tests/test_server.py`
Expected: PASS, suite completa.

- [ ] **Step 5: Commit**

```bash
git add examples/ tests/test_e2e_campaign.py
git commit -m "feat: ship a two-phase example whose trainer cannot score itself"
```

---

### Task 11: El loop supervisa lo que ejecuta

**Files:**
- Modify: `ai_experiments/loop.py`, `ai_experiments/daemon.py`
- Test: `tests/test_loop_supervision.py` (crear)

**Interfaces:**
- Consumes: el orquestador de la base canónica.
- Produces: `ai_experiments.daemon.supervise_once(store, campaign_store, campaign_id) -> list[MonitorDecision]`, extraído del recorrido del daemon, que `run_loop` llama en cada iteración. Una decisión de matar un run se ejecuta en los dos caminos por igual.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loop_supervision.py
"""The unattended path is the one that most needs supervision."""

from __future__ import annotations

from ai_experiments.daemon import supervise_once
from ai_experiments.loop import run_loop


def test_the_loop_supervises_every_iteration(tmp_path, monkeypatch):
    calls: list[str] = []

    def _spy(store, campaign_store, campaign_id):
        calls.append(campaign_id)
        return []

    monkeypatch.setattr("ai_experiments.loop.supervise_once", _spy)

    report = _run_a_short_loop(tmp_path)   # helper abajo

    assert calls, "run_loop never supervised the campaign it was driving"
    assert set(calls) == {report.campaign_id}
```

El helper `_run_a_short_loop` arma un goal de dos trials contra `examples/toy_train.py` sobre el backend local con `max_rounds=2`, copiando el armado que ya usa `tests/test_e2e_campaign.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_supervision.py -q`
Expected: FAIL con `AttributeError: module 'ai_experiments.loop' has no attribute 'supervise_once'`.

- [ ] **Step 3: Write minimal implementation**

Extraer de `daemon.py` el cuerpo que hoy recorre una campaña, sin el bucle ni los sleeps:

```python
# ai_experiments/daemon.py
def supervise_once(
    store: FilesystemRunStore, campaign_store: CampaignStore, campaign_id: str
) -> list[MonitorDecision]:
    """One supervision pass over a campaign's active runs.

    Kept short and free of agent calls on purpose: `iax loop` runs this on
    every iteration, and an overnight loop that pauses for a slow agent is a
    loop that is not watching anything.
    """
```

El recorrido del daemon pasa a llamar a esta función por campaña, de modo que ambos caminos comparten el código.

En `loop.py`, dentro del `while`, después de `advance`:

```python
from ai_experiments.daemon import supervise_once

        state = orchestrator.advance(state.campaign_id)
        supervise_once(store, orchestrator.campaign_store, state.campaign_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_loop_supervision.py tests/test_daemon.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/loop.py ai_experiments/daemon.py tests/test_loop_supervision.py
git commit -m "feat: supervise the campaign from the loop, not only the daemon"
```

---

### Task 12: Una cohorte se cierra antes de admitir la siguiente

**Files:**
- Modify: `ai_experiments/orchestrator.py`, `ai_experiments/loop.py`
- Test: `tests/test_loop_supervision.py`

**Interfaces:**
- Consumes: `CampaignOrchestrator.advance` de la base canónica.
- Produces: `CampaignOrchestrator.advance(campaign_id, admit: bool = True)`. Con `admit=False` refresca, evalúa la condición de parada y actualiza el mejor, pero **no** llama `_fill_capacity`. `run_loop` pasa `admit=False` cuando hay una revisión pendiente, corre la revisión, y recién entonces admite.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loop_supervision.py — agregar al final
def test_the_review_sees_the_cohort_before_the_next_one_is_submitted(tmp_path):
    """An agent that says 'stop' must be able to stop something.

    Reviewing after `advance` had already filled capacity meant the verdict
    arrived when the next cohort was submitted and paid for.
    """
    submitted_at_review: list[int] = []

    class _Reviewer:
        def run(self, prompt, *, role="planner"):
            submitted_at_review.append(prompt.count("trial_id"))
            return AgentResult(ok=True, payload={"verdict": "stop", "reason": "done"})

    report = _run_a_reviewed_loop(tmp_path, _Reviewer())   # helper abajo

    assert report.loop_stop == "agent_review_stop"
    assert report.trials == submitted_at_review[0], (
        "the review ran after a new cohort had already been submitted"
    )
```

El helper `_run_a_reviewed_loop` arma un goal con `analysis.review_between_rounds: True` y `budget.max_parallel: 1`, e inyecta el reviewer vía `orchestrator.agent_runner`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_supervision.py -q`
Expected: FAIL — `report.trials` es mayor que los trials que la revisión vio.

- [ ] **Step 3: Write minimal implementation**

```python
# ai_experiments/orchestrator.py
    def advance(self, campaign_id: str, admit: bool = True) -> CampaignState:
        """Move the campaign one step.

        With ``admit=False`` the campaign collects and scores what it has, but
        submits nothing: the caller wants to review a closed cohort before
        paying for the next one.
        """
        ...
        if admit:
            submitted = self._fill_capacity(state, goal, backend)
            if submitted:
                state.rounds += 1
        ...
```

```python
# ai_experiments/loop.py — dentro del while
        state = orchestrator.advance(state.campaign_id, admit=False)
        supervise_once(store, orchestrator.campaign_store, state.campaign_id)

        if state.status in TERMINAL_STATUSES:
            break

        verdict = _review(orchestrator, state, reviews)
        if verdict == "stop":
            state = orchestrator.stop(state.campaign_id, "agent_review_stop")
            loop_stop = "agent_review_stop"
            break

        state = orchestrator.advance(state.campaign_id)
```

La condición `state.rounds > rounds_before` desaparece: la revisión ya no depende de que un lote se haya enviado, sino de que una cohorte se haya cerrado.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -q -m 'not integration' --ignore=tests/test_server.py`
Expected: PASS, suite completa.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/orchestrator.py ai_experiments/loop.py tests/test_loop_supervision.py
git commit -m "feat: close a cohort and review it before admitting the next"
```

---

### Task 13: El agente no mueve su propio techo

**Files:**
- Modify: `ai_experiments/loop.py` (`_apply_changes`)
- Test: `tests/test_loop_supervision.py`

**Interfaces:**
- Consumes: `_apply_changes` de la base canónica, que hoy mergea las claves `("search_space", "budget")`.
- Produces: `_apply_changes` acepta `search_space` y rechaza `budget`, registrando un evento `warning` con la petición íntegra. Ampliar el techo requiere al usuario, no una revisión aprobada.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loop_supervision.py — agregar al final
def test_the_agent_can_widen_the_search_space(tmp_path):
    goal = _goal(tmp_path)

    changed = _apply_changes(goal, {"search_space": {"lr": {"type": "loguniform",
                                                           "low": 1e-5, "high": 1.0}}})

    assert "lr" in changed.search_space


def test_the_agent_cannot_widen_its_own_budget(tmp_path):
    """An optimizer asked to stay under a ceiling will ask to raise it."""
    goal = _goal(tmp_path)
    before = goal.budget.max_trials

    changed = _apply_changes(goal, {"budget": {"max_trials": before * 100}})

    assert changed.budget.max_trials == before


def test_a_rejected_budget_change_is_recorded(tmp_path, caplog):
    goal = _goal(tmp_path)

    _apply_changes(goal, {"budget": {"max_gpu_hours": 10_000}})

    assert "budget" in caplog.text
```

`_goal` construye un `GoalSpec` mínimo contra `examples/toy_train.py`, como el resto del archivo.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_loop_supervision.py -q`
Expected: FAIL — `max_trials` sube a `before * 100`.

- [ ] **Step 3: Write minimal implementation**

```python
# ai_experiments/loop.py
#: What an accepted review may change on its own. `budget` is deliberately
#: absent: a loop is an optimizer, and a ceiling it can move is not a ceiling.
#: Redistributing within the budget is fine; raising it needs the user.
APPLICABLE_KEYS = ("search_space",)


def _apply_changes(goal: GoalSpec, changes: dict[str, Any]) -> GoalSpec:
    refused = [key for key in changes if key not in APPLICABLE_KEYS]
    if refused:
        logger.warning(
            "refusing agent changes to %s; only %s may be changed by a review",
            ", ".join(sorted(refused)),
            ", ".join(APPLICABLE_KEYS),
        )
    ...
```

El resto del cuerpo sigue igual, iterando solo sobre `APPLICABLE_KEYS`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -q -m 'not integration' --ignore=tests/test_server.py`
Expected: PASS, suite completa.

- [ ] **Step 5: Commit**

```bash
git add ai_experiments/loop.py tests/test_loop_supervision.py
git commit -m "fix: a review may widen the search space, never the budget"
```

---

## Puerta de salida de la fase 1

Antes de pasar al plan de fase 2, estos criterios del spec deben verificarse a mano sobre `examples/`:

- [ ] Un candidato que escribe `report_result(...)` en `toy_train.py` **no puntúa** (Task 4, Task 8).
- [ ] Un trial que falla no gana, aunque haya reportado (Task 1 + Task 5).
- [ ] Un workload sin `report_result` falla ruidosamente con `no_result` (Task 5).
- [ ] El proceso de train no recibe `IAX_DATA_TEST` (Task 7).
- [ ] El mismo comando que corre la campaña supervisa (Task 11).
- [ ] Una cohorte se cierra y se revisa antes de admitir la siguiente (Task 12).
- [ ] Ninguna revisión aprobada amplía el techo de presupuesto (Task 13).
- [ ] `.venv/bin/python -m pytest -q -m 'not integration' --ignore=tests/test_server.py` verde.
