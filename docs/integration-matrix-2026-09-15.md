# Matriz de integración — base canónica para el circuito honesto

Fecha: 2026-09-15. Rama producida: `feat/honest-circuit` (`d8273f3`).

## Lo que se esperaba encontrar

El plan de fases 0–1 asumía dos ramas con capacidades cruzadas: `integration/all`
(`fae2c00`) con la higiene de ejecución, y `fix/loop-collects-last-round` (`c4f0453`)
con el trabajo de agente —`ObjectiveReading`, `best_of`, `improve/`, `agents/`—, de
modo que la base canónica hubiera que armarla archivo por archivo.

## Lo que hay

No es así. Las dos ramas salen del mismo commit:

```
137315f  Say who chose each trial        <- merge-base
   |
   +-- fae2c00  integration/all                 (+22 commits)
   +-- c4f0453  fix/loop-collects-last-round     (+1 commit)
```

El trabajo de agente ya estaba en el merge-base. Estos archivos son byte-idénticos
en las dos ramas:

| Archivo | `fae2c00` vs `c4f0453` |
|---|---|
| `ai_experiments/planner/analysis.py` | idéntico |
| `ai_experiments/improve/variants.py` | idéntico |
| `ai_experiments/improve/rounds.py` | idéntico |
| `ai_experiments/agents/runner.py` | idéntico |
| `ai_experiments/agents/strategy.py` | idéntico |

Y `fae2c00` es un superconjunto estricto en todo lo demás: `preflight.py` (73 líneas)
y `procs.py` (221 líneas) existen solo ahí, junto con los locks por run, los fixes de
worker, el reaping de huérfanos portable y el ruido de logs.

## La matriz

Una fila:

| Capacidad | Dónde está | Acción |
|---|---|---|
| Todo lo de higiene, locks, preflight, procs, worker | `fae2c00` | rama base |
| `reconcile` del último lote (`iax loop --max-rounds N` descartaba la ronda que ya había pagado) | `c4f0453` | cherry-pick |

```bash
git checkout -b feat/honest-circuit fae2c00
git cherry-pick c4f0453      # auto-merge limpio en cli.py y orchestrator.py
```

## Verificación

```
399 passed, 16 deselected     # pytest -q -m 'not integration' --ignore=tests/test_server.py
```

El `ModuleNotFoundError: No module named 'ray.dag'` al final es ruido del atexit de
Ray durante el teardown del intérprete, no un test.

## Lo que esto cambia en el plan

La Task 1 del plan de fases 0–1 describe un inventario por archivo y una
reconstrucción selectiva. No hizo falta: la base canónica es `fae2c00` más un
cherry-pick. El resto del plan no se ve afectado — las capacidades que sus tareas
asumen presentes (`ObjectiveReading` con `miss_reason`, `best_of` filtrando por
`status == "completed"`) están todas en la rama resultante.
