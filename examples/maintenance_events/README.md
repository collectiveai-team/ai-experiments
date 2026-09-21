# maintenance-events

El testbed de CPU barato para iax: un workload real, con datos industriales
reales, que corre en decenas de segundos por trial.

**Qué predice.** Dada la historia reciente de un condensador de turbina de vapor
—vacío, temperaturas de entrada y salida de agua de circulación a ambos lados,
potencia activa, presión y temperatura ambiente— responde por sí o por no si en
los próximos **15 días** va a haber una limpieza. El modelo corre todos los
días, así que cada día del histórico es una ventana: 2020-2024 dan unas 1.700
ventanas etiquetadas.

**Por qué existe.** iax necesita un workload que se pueda arruinar. Este tiene
las tres trampas que un harness de experimentos tiene que respetar:

- **Fuga temporal.** La ventana de historia es semiabierta, `[as_of - window, as_of)`.
  Un fold que entrene con datos posteriores al `as_of` de su test infla la
  métrica sin que nada falle.
- **Cobertura de etiquetas.** Las señales van de 2020 a 2024; los registros de
  limpieza, no. El de MAPRO cubre 2022-08 → 2024-07 y el log de operarios
  2023-11 → 2024-12. Fuera del período de su fuente, "no hay evento" significa
  "no hay registro". El workload recorta los `as_of` a la cobertura de la fuente
  elegida; `--ignore-label-coverage` desactiva el recorte y muestra el daño.
- **Desbalance.** Las limpiezas son raras. Accuracy no dice nada; el objetivo es
  PR-AUC contra la tasa base, que el workload reporta en cada corrida.
- **Un objetivo que no es el último punto.** `extract_objective` se queda con el
  **mejor** valor observado, no con el último. Por eso las métricas por fold
  viajan bajo `fold_pr_auc` y `pr_auc` se emite **una sola vez**, al final, con
  el promedio de los folds. Si compartieran nombre, la campaña registraría el
  fold más afortunado de cada trial y el planner perseguiría suerte. Hay un test
  que lo fija: `tests/test_train.py::test_per_fold_metrics_never_use_the_objective_key`.

## El dataset

El artefacto es un zip de ~16 MB con dos tablas:

| archivo | contenido |
|---|---|
| `signals.parquet` | 16 señales en una grilla de 15 minutos, 2020-01-01 → 2024-12-31 (175.392 filas) |
| `events.csv` | 63 limpiezas con su fecha, su `source` (`mapro` / `operator` / `both`), la duración y si fue programada |
| `manifest.json` | sha256, cobertura por señal y la dataset card |

**El link de descarga es privado.** No está en el repo y no debe commitearse
nunca. Configuralo de una de estas dos formas:

```bash
export MAINTENANCE_EVENTS_URL="https://drive.google.com/file/d/<id>/view"
```

o copiá la plantilla y poné el link ahí (el archivo está gitigneado):

```bash
cp dataset.example.toml dataset.local.toml
$EDITOR dataset.local.toml
```

La primera corrida descarga y verifica el sha256 en `~/.cache/maintenance-events`
(o en `MAINTENANCE_EVENTS_CACHE`). Si no hay link configurado, el workload falla
con instrucciones en vez de inventar datos.

### Anonimización

La anonimización es **solo de identificadores**: se eliminaron el nombre de la
planta y del grupo, los tags del DCS, los nombres de los operarios, los textos
libres de motivo y observaciones, y la valorización económica. Las señales
mantienen las unidades de ingeniería reales, y **las marcas de tiempo son
reales**.

Eso es deliberado, y tiene una consecuencia: alguien del sector podría
reconocer la unidad generadora cruzando las fechas de salida de servicio con
registros públicos de despacho. El control de acceso es el link privado, no el
contenido del archivo. No lo publiques.

## Correrlo

```bash
cd examples/maintenance_events
uv sync --extra dev
```

Un trial suelto, sin dataset, contra datos sintéticos con una señal de
ensuciamiento real —sirve para verificar la instalación y el contrato con iax:

```bash
uv run python -m maintenance_events.train --self-test --n-folds 5
```

Un trial sobre los datos reales:

```bash
uv run python -m maintenance_events.train --window-days 90 --model hist_gb
```

La campaña completa, desde la raíz del repo:

```bash
uv run iax campaign validate examples/maintenance_events/goal.yaml
uv run iax campaign start    examples/maintenance_events/goal.yaml
uv run iax daemon
```

Los nombres de las flags de `train.py` son exactamente las claves de
`search_space` en `goal.yaml`: `build_trial_manifest` appendea cada parámetro
como `--nombre valor`, así que renombrar una flag sin renombrar la clave rompe
el trial en silencio.

### Costo por trial

Un trial de 5 folds sobre 435 ventanas tarda ~27 s; con los 12 folds que usa `goal.yaml`, del orden de un minuto. Dos cosas lo dominan y
conviene saberlas antes de tocar el espacio de búsqueda:

- **`max_bins`.** Con `class_weight=balanced` sklearn le pasa `sample_weight` al
  binner de `HistGradientBoosting`, que entonces calcula un percentil ponderado
  por feature y por bin en Python. El tiempo del fit es casi lineal en
  `max_bins`: 9,9 s con 255, 2,3 s con 32.
- **Hilos.** El workload limita BLAS/OpenMP a **un hilo** por trial, porque iax
  corre varios trials en paralelo y con 435 filas la sincronización cuesta más
  que el cómputo: el mismo fit tarda 22,9 s con los 16 hilos del host y 9,9 s
  con uno. `MAINTENANCE_EVENTS_THREADS=0` devuelve los defaults de sklearn.

## Las métricas

**Cada fold es una observación del objetivo**, no un paso hacia él. Un fold que
puntuó emite:

| clave | qué es |
|---|---|
| `pr_auc` | **el objetivo** en ese fold |
| `baseline_pr_auc` | la tasa base de *ese* bloque de test. El score del trial es `pr_auc - baseline_pr_auc`, apareado dentro de la observación |
| `fold_auroc` | AUROC del fold, más estable pero menos informativo con desbalance |
| `fold_positives` / `fold_n_test` | de qué tamaño fue la evidencia |

Un fold sin positivas **no emite ninguna de las dos claves objetivo**: PR-AUC no
está definido ahí, y un cero arrastraría el promedio por un fold que no midió
nada. El arnés saltea la observación incompleta y la cuenta en `valid_folds`.

El punto final resume, con claves distintas a propósito — repetir la clave
objetivo sería una observación de más, el promedio contado dos veces y un error
estándar más chico que el real:

| clave | qué es |
|---|---|
| `mean_pr_auc` / `mean_baseline` | el promedio que el arnés también calcula, para leer el log a ojo |
| `pr_auc_std` | dispersión entre folds; alta = el resultado depende del período |
| `auroc`, `recall_at_p50` | promedios; `recall_at_p50` es la lectura operativa |
| `valid_folds` | folds que puntuaron; si es 0 el trial no dice nada |

Quien puntúa es `objective.aggregate: mean`: promedia los folds, saca el error
estándar y con eso decide si el trial se distingue del ruido. Eso lo hace el
arnés, no este workload y no quien lea el reporte.

## Cuántos folds

`--n-folds 12` no es un número redondo elegido a ojo. Sale de la dispersión que
la campaña anterior midió: sobre sus 24 trials × 5 folds, la **sd del lift por
fold dentro de un mismo trial** fue **0,113** (agrupada; mediana por trial
0,105).

Con esa sd, el lift mínimo detectable a dos colas 95 % y poder 80 % es
`2,80 × 0,113 / √k`:

| folds válidos | lift mínimo detectable |
|---|---|
| 5 | 0,141 |
| 10 | 0,100 |
| 12 | 0,091 |
| 20 | 0,071 |
| 40 | 0,050 |

El mejor lift promedio que aquella campaña observó fue **0,104**. Con 5 folds no
podía distinguirlo del ruido ni aunque fuera real: el diseño estaba por debajo
de su propio resultado antes de empezar. Por eso `min_observations: 10` y
`--n-folds 12`, con dos de margen para folds que se queden sin positivas.

Dos salvedades que el número no cubre. Más folds parten la serie en bloques de
test más chicos, con menos positivas cada uno, así que la sd real con 12 folds
va a ser algo mayor que 0,113 y el mínimo detectable algo peor que 0,091. Y
`min_objective: 0.05` está por debajo de lo que 12 folds detectan con
confianza: es el umbral que cambiaría una decisión de mantenimiento, no el que
el diseño garantiza medir. Por eso el criterio pide *además*
`require_beats_baseline`, que se evalúa contra el error estándar **realmente
observado** y no contra este supuesto.

## Qué cuenta como éxito

`goal.yaml` lo declara en `success_criteria`, antes de correr nada:

| criterio | por qué |
|---|---|
| `min_objective: 0.05` | por debajo el modelo no paga el costo de operarlo |
| `min_observations: 10` | menos folds válidos dejan el intervalo demasiado ancho |
| `require_separation` | el mejor de 24 trials ruidosos le gana al segundo por construcción |
| `require_beats_baseline` | el intervalo del lift tiene que despejar el cero |

`iax campaign status` y `iax loop` reportan `met: true/false` con los criterios
que fallaron, y `iax loop` sale con código 4 si no se cumplen. Nadie decide
después de ver el número.

La validación es walk-forward: train expansivo, el bloque siguiente como test y
un **gap igual al horizonte** entre el fin del train y el inicio del test, para
que ninguna ventana de train pueda ver el evento que su test predice.

## Regenerar el artefacto

Los datos crudos no están en el repo. Con acceso a ellos:

```bash
uv run --extra prepare python -m prepare.build_dataset \
  --raw-data-dir ~/ruta/a/los/datos/crudos \
  --out-dir .cache/dist
```

Sale `maintenance-events-v1.zip` con su `manifest.json`. El pipeline detecta el
locale del export del SPPA-T3000 (hay dos: `;Tiempo;` con coma decimal y
`;Time;` con punto), lee como `NaN` los valores marcados con el flag de mala
calidad `?` del DCS, renombra los 19 tags crudos a las 16 señales del spec y
descarta todo campo identificatorio. Subir el zip a Drive es un paso manual.

## Tests

```bash
uv run --extra dev python -m pytest tests -q
```

64 tests, ~40 s. No necesitan el dataset: los del pipeline de preparación usan
fixtures chicas y los del workload usan `--self-test`.
