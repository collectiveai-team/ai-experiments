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

Un trial de 5 folds sobre 435 ventanas tarda ~27 s; con los 20 folds que usa `goal.yaml`, del orden de dos minutos. Dos cosas lo dominan y
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

Con esa sd, el lift mínimo detectable a dos colas 95 % y poder 80 % **se
suponía** `2,80 × 0,113 / √k`:

| folds válidos | lift mínimo detectable |
|---|---|
| 5 | 0,141 |
| 10 | 0,100 |
| 12 | 0,091 |
| 20 | 0,071 |
| 40 | 0,050 |

El mejor lift promedio que aquella campaña observó fue **0,104**. Con 5 folds no
podía distinguirlo del ruido ni aunque fuera real: el diseño estaba por debajo
de su propio resultado antes de empezar. De ahí salieron `min_observations: 10`
y `--n-folds 12`, con dos de margen para folds que se queden sin positivas.

Después se midió, y esa cuenta resultó estar mal.

### Lo que pasó cuando se midió

Esa tabla supone que la sd se queda en 0,113 cuando cambia `k`. **No se queda.**
Más folds parten la serie en bloques de test más chicos, con menos positivas
cada uno, y la sd crece casi exactamente a la misma velocidad que √k. Medido
sobre el dataset real, con la configuración por defecto:

| fuente | k | folds válidos | sd del lift | mínimo detectable |
|---|---|---|---|---|
| `union` | 12 | 10 | 0,129 | 0,115 |
| `union` | 20 | 17 | 0,170 | 0,116 |
| `operator` | 12 | **9** | 0,163 | 0,152 |
| `operator` | 20 | 15 | 0,173 | 0,125 |

**Subir folds no compra poder.** El mínimo detectable queda clavado en ~0,115 de
12 a 20 folds: el dataset tiene 63 eventos en cinco años y cortarlo más fino no
multiplica la información. Lo único que sí mejora es el conteo de folds válidos,
y ahí hay un problema concreto: con `--n-folds 12`, `operator` deja **9** folds
con positivas y no puede cumplir `min_observations: 10` haga lo que haga el
modelo. Con 20 las tres fuentes quedan en 14 o más.

Como referencia de cuánto ruido hay: el *mismo* config sobre los *mismos* datos
da lift +0,108 con 12 folds y +0,055 con 20.

`min_objective: 0.05` está por debajo de lo que el diseño detecta con confianza:
es el umbral que cambiaría una decisión de mantenimiento, no el que se garantiza
medir. Por eso el criterio pide *además* `require_beats_baseline`, que se evalúa
contra el error estándar **realmente observado**.

## Por qué el objetivo sigue siendo el promedio de folds

El promedio de folds tira información: resume ~800 ventanas en 10 números, cada
uno un PR-AUC de muestra chica. La alternativa obvia es puntuar una sola vez
sobre todas las predicciones out-of-fold juntas y sacar el intervalo
remuestreando esas ventanas. `evaluation.py` implementa eso —`block_bootstrap`,
con remuestreo **por bloques** porque las ventanas van día a día y su etiqueta
mira 15 días adelante, así que dos vecinas son casi la misma observación—.

Se probó y **no sirve como objetivo acá**:

| fuente | promedio de folds | agrupado (bootstrap, bloque 15) | agrupado, rango por fold |
|---|---|---|---|
| `union` | **+0,108** | −0,021 ± 0,048 | +0,084 |
| `mapro` | **+0,074** | −0,028 ± 0,047 | −0,012 |
| `operator` | **+0,057** | −0,024 ± 0,061 | +0,010 |

Los dos estimadores no coinciden ni en el signo. La tercera columna dice por
qué: normalizar los scores a rango **dentro de cada fold** antes de agrupar
mueve las tres fuentes hacia arriba y devuelve `union` a +0,084. Es decir,
**agrupar mezcla scores de modelos distintos**, entrenados con distinta cantidad
de historia, sobre períodos con distinta tasa base; el ranking conjunto mide
deriva de calibración entre períodos, no discriminación dentro de uno.
Walk-forward y pooling no se llevan. (La normalización tampoco recupera el
promedio de folds — no es un arreglo, es el diagnóstico.)

El bootstrap se reporta igual, con claves que **no** son la del objetivo
(`lift_mean`, `lift_stderr`, `pooled_windows`), porque es la medición que
sostiene esta decisión y porque su error estándar es el único número honesto
sobre cuánta suerte hay en qué ventanas tocaron. Vale mirar cuánto importa la
corrección por bloques: en `union`, el error estándar pasa de 0,015 remuestreando
ventanas sueltas a 0,048 remuestreando bloques de 15. Tres veces más ancho — todo
lo demás sería confianza inventada ignorando que las ventanas se solapan.

Y hay un problema de fondo que ninguno de los dos estimadores arregla: el lift
`pr_auc − tasa_base` **no es comparable entre folds con distinta tasa base**. Con
tasa base 0,92 el lift máximo posible es 0,08; con 0,023 es 0,977. Las tasas base
por fold acá van de 0,023 a 1,0, así que el promedio está dominado por a qué fold
le tocó una tasa base extrema: el 72 % del lift promedio de `mapro` sale de un
solo fold con tasa base 0,023, que aporta +0,477 él solo. Normalizar por el margen disponible —
`(pr_auc − base) / (1 − base)` — lo hace comparable, pero no mejora la relación
señal-ruido (medida: 2,5 → 3,1 en `union`, 1,3 → 0,9 en `mapro`).

## Qué cuenta como éxito

`goal.yaml` lo declara en `success_criteria`, antes de correr nada:

La pregunta es **"¿alguna configuración le gana a predecir la tasa base?"**, no
"¿cuál es la mejor?". Ver abajo por qué la segunda no tiene respuesta acá.

| criterio | por qué |
|---|---|
| `min_objective: 0.05` | por debajo el modelo no paga el costo de operarlo |
| `min_observations: 10` | con 20 folds nadie queda afuera por construcción; atrapa un trial degenerado |
| `require_separation: false` | apagado a propósito: la campaña no pregunta cuál gana |
| `require_beats_baseline` | **el criterio que contesta la pregunta**: el intervalo del lift tiene que despejar el cero |

`iax campaign status` y `iax loop` reportan `met: true/false` con los criterios
que fallaron, y `iax loop` sale con código 4 si no se cumplen. Nadie decide
después de ver el número.

### Por qué no se pregunta cuál gana

Separar pide que el mejor trial le saque al segundo más de
`1,96 × hypot(se, se) ≈ 0,114` de lift — más que el lift entero que consigue
cualquier configuración acá. Con 63 eventos en cinco años, el error estándar del
lift no baja de ~0,041 por más folds que se usen (ver arriba), así que ninguna
campaña de 24 trials sobre este dataset puede coronar un ganador distinguible.
Preguntarlo igual sólo produce un `met: false` que no informa nada sobre el
problema de mantenimiento.

Así que `require_separation` está en **`false` a propósito**, y la diferencia
importa: no es "no pudimos distinguir un ganador", es "no preguntamos por un
ganador". Lo que sí se pregunta lo contesta `require_beats_baseline`, y con
`se ≈ 0,041` el intervalo despeja el cero recién con un lift observado de
**~0,081** — o sea que el criterio que manda acá no es `min_objective: 0.05`
sino éste. Si la campaña contesta que sí, el resultado es "existe una
configuración que le gana a la tasa base", sin nombre propio: *cuál* de las
configuraciones es, este dataset no lo puede decir.

La validación es walk-forward: train expansivo, el bloque siguiente como test y
un **gap igual al horizonte** entre el fin del train y el inicio del test, para
que ninguna ventana de train pueda ver el evento que su test predice.

## Qué contestó la campaña

`cmp_8b677e32ecb6`, 24 trials, 24 completos, 0 fallos, 1256 s de máquina.

**Sí, con reservas.** El mejor trial da un lift de **0,162 ± 0,061** sobre 17
folds: el intervalo despeja el cero y los tres criterios se cumplen. Es
`logreg` — el modelo más simple de los dos — con ventanas de 180 días y
remuestreo diario, sobre `union`.

Lo que sostiene la respuesta no es ese trial solo, que al fin y al cabo es el
máximo de 24 y como tal está sesgado hacia arriba. Es que **los 24 trials dan
lift positivo** (de 0,036 a 0,162) y **9 de 24 despejan el cero por su cuenta**,
sin ser el máximo de nada. Los folds válidos fueron 13 a 17 en todos, así que
`min_observations: 10` no eliminó a nadie: el cambio de 12 a 20 folds hizo
exactamente lo que se le pidió.

Dos reservas que el `met: true` no dice:

- **El estimador agrupado concuerda en dirección pero no en magnitud.** Positivo
  en 22 de 24 trials, pero para el ganador da 0,071 ± 0,051, que *no* despeja el
  cero. La lectura honesta es que el efecto existe y es más chico que lo que
  sugiere el promedio de folds del trial seleccionado. (Esto también matiza la
  medición de la sección anterior: el desacuerdo *de signo* entre los dos
  estimadores era propio de la familia de configuraciones que se midió ahí, no
  general.)
- **No hay ganador.** `t021` le saca 0,011 a `t012`, dentro del ruido. La
  campaña lo reporta y no lo usa como criterio, que es la razón por la que
  `require_separation` está en `false`.

## Rondas que cambian el código

`goal.yaml` habilita variantes: una ronda puede agregar una feature o un modelo
en vez de mover perillas. El arnés copia este directorio bajo
`<campaign_dir>/variants/<variant_id>/`, aplica los archivos enteros que le
pasan, y corre el smoke check antes de dejarle gastar un solo trial:

```bash
# escribí el archivo nuevo donde quieras, completo
iax campaign variant <campaign_id> \
  --edit maintenance_events/features.py=/tmp/features_con_lag.py \
  --hypothesis "lag de 7 días del vacío normalizado" --json
iax campaign variants <campaign_id>
iax campaign suggest <campaign_id> --params '{"model": "hist_gb", "window_days": 90}' \
  --variant var_1a2b3c4d
```

Tres cosas que este ejemplo decide y conviene no aflojar:

- **Sólo `features.py` y `models.py` son editables.** `evaluation.py` queda
  afuera a propósito: una variante que puede reescribir su propia evaluación
  optimiza el termómetro, y todo número posterior es suyo.
- **El smoke check es `--self-test --n-folds 2`**, sobre datos sintéticos: no
  necesita el dataset ni el link privado, y falla si la variante no importa, no
  featuriza o deja de reportar `pr_auc`/`baseline_pr_auc`. Tarda ~16 s, la mitad
  en crear el venv de la copia; por eso `smoke_timeout_seconds: 300`.
- **Un modelo nuevo necesita además ensanchar el espacio.** `--model` valida
  contra `MODELS`, pero `search_space.model` es un `choice` fijo: agregar el
  valor va por `iax campaign edit`, no por la variante.

El procedimiento completo está en la skill `proposing-variants`.

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
