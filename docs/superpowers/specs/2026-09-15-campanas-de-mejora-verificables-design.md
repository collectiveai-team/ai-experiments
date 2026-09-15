# Diseño: campañas de mejora verificables

**Fecha:** 2026-09-15
**Estado:** aprobado en discusión; pendiente de revisión del spec
**Responde a:** [assessment-y-plan-2026-09-07.md](../../assessment-y-plan-2026-09-07.md)

## Problema

El assessment del 2026-09-07 diagnostica bien el estado del runtime, pero ordena el
plan alrededor de una suposición que la discusión posterior corrigió: trata la mejora
de código como una segunda modalidad, para fase 6, después de cinco fases de
endurecimiento de la búsqueda de hiperparámetros.

La tesis del producto es la inversa. Lo que se busca de iax es que **un agente pueda
disparar la creación de implementaciones nuevas —feature engineering, modelos— en el
proyecto del usuario**, y que el arnés determine si esa implementación es mejor. La
búsqueda de hiperparámetros es el caso fácil, no el objetivo. El código generado vive
en el proyecto donde se usa iax; iax nunca se modifica a sí mismo.

Ese cambio de centro reordena las prioridades, porque introduce un modo de falla que
el assessment archiva como higiene de sandboxing (hallazgo I, P1) y que en realidad es
la columna de corrección de todo el producto.

## La falla que ordena el plan

El loop es un optimizador apuntado a una métrica. Si el candidato puede escribir
código, **todo camino sin proteger entre ese código y el número reportado es parte del
espacio de búsqueda.** No hace falta un agente malicioso: basta con que el loop
conserve la variante que sube la métrica.

Verificado en `c4f0453`, el camino más barato hoy:

```python
# variants.py:99 — la protección es opt-in; editable_paths vacío es el default
if spec.editable_paths and not any(fnmatch(relative, p) for p in spec.editable_paths):
    raise VariantError(...)
```

Con el default, el candidato puede reescribir el archivo que calcula y reporta la
métrica. Un `report_result(test_acc=0.99)` pasa el smoke check, sale con exit code 0,
queda `completed`, declara un resultado, es elegible, gana y dispara `target_reached`.
Satisface todas las garantías del arnés sin violar ninguna.

Los casos que importan, además, no se parecen a hacer trampa. Se parecen a bugs: una
feature que calcula estadísticas sobre el dataset completo, un dataloader que termina
evaluando sobre datos vistos, un refactor que puntúa un subset más fácil. Todos suben
la métrica, todos pasan el smoke check, y el loop construye encima.

Un prompt puede pedir intención. No puede sacar algo del espacio de búsqueda.

### Defensa en capas

Cada capa ataca una falla distinta. Ninguna cubre a las otras.

| Falla | Defensa | Naturaleza |
|---|---|---|
| El candidato fabrica el número | `evaluate` protegido, proceso aparte, árbol pristino | Estructural |
| El candidato leakea test en features | El proceso de train no tiene acceso al test set | Estructural, si los datos van por referencia |
| La búsqueda sobreajusta val | val conduce la búsqueda, test confirma al final | Por protocolo |
| El split ya venía mal de origen | — | Fuera de alcance: del usuario |

## Decisiones

| # | Decisión | Reemplaza a |
|---|---|---|
| 1 | `IAX_RESULT` es el único canal que puntúa; `IAX_METRIC` queda para progreso | Agregación `min`/`max` sobre la curva |
| 2 | El workload declara `train` y `evaluate`; `evaluate` corre desde el árbol pristino | Un entrypoint, evaluación dentro de la variante |
| 3 | El proceso de train no recibe ruta al test set | Sin borde de datos |
| 4 | B se parte en B1/B2 (P0 chicos) y B3 (la tesis) | B como un P0 monolítico |
| 5 | El implementador es un despacho de tarea a un agente con herramientas | `VariantEdit[]` en el payload del planner |
| 6 | SQLite local como fuente transaccional de estado, intentos y reservas | JSON con escritura atómica |
| 7 | Reservas en transacción + techo duro por intento en la ejecución | Contabilidad por polling |

### 1. `IAX_RESULT` como canal único de puntuación

Hoy existe un solo canal, `IAX_METRIC {json}`, y `report.py:47` documenta el patrón de
progreso (`report_metric(step=epoch, loss=..., val_loss=...)`). El scorer se queda con
`min(values)` sobre esa serie.

El contrato deseado —una métrica de evaluación única sobre test— **no es expresable**:
un workload no tiene forma de decir "esto es el resultado" en vez de "esto es
progreso". El mínimo de una curva ruidosa además es un estimador sesgado; un candidato
que entrena más epochs gana por tener más tiradas de dado.

```python
for epoch in range(n):
    report_metric(step=epoch, loss=l, val_loss=vl)   # progreso: monitoreo, plateau, NaN

test_acc = evaluate(model, test_set)
report_result(test_acc=test_acc)                     # lo único que puntúa
```

- `extract_objective` deja de agregar curvas. No hace falta un campo `aggregation`.
- Ausencia de resultado → `miss_reason="no_result"`, sumado al vocabulario que
  `c4f0453` ya introdujo (`no_metrics`, `metric_absent`, `not_finite`).
- Se combina con `best_of(status == "completed")` (#12): ganar exige terminar bien
  **y** haber declarado un resultado.
- **Ruptura dura, sin fallback al último `IAX_METRIC`.** Un fallback silencioso
  reintroduce el modo de falla que la decisión elimina.

### 2. Split train/evaluate

```
un solo job, un solo nodo
  entrypoint = lanzador de iax
    1. train    desde variants/<id>/   (editable por el agente) -> model.pt, disco local
    2. evaluate desde el árbol pristino (el agente nunca lo tocó) -> report_result(...)
```

El agente solo puede producir **artefactos**. El evaluador protegido es el único que
puede producir un **resultado**. Como refuerzo, iax descarta cualquier `IAX_RESULT`
emitido por el proceso de train: la garantía deja de depender de configuración.

Es estructural y no requiere almacenamiento compartido: ambas fases corren en el mismo
job sobre el mismo nodo, con handoff por disco local. `submit_job(entrypoint=...)` lo
soporta tal como está. El transporte de artefactos sigue haciendo falta para
**recuperar el modelo ganador**, no para puntuarlo, y por eso sale del camino crítico.

Frente a la alternativa de `editable_paths` obligatorio: esa es una garantía por
configuración, no por estructura. El evaluador seguiría corriendo dentro de la copia
del agente, alcanzable por monkeypatching, `sitecustomize.py` o un import intermedio.

### 3. Borde de datos

El split protege el código que puntúa; no protege contra que el candidato entrene con
información que no debería tener. Eso es una propiedad del entorno, no del código:

```
train    (variante, editable)   ->  train + val      ... nunca ve test
evaluate (pristino, protegido)  ->  test + artefacto
```

Si el proceso de train no tiene ruta al test set, ninguna feature puede leakearlo, sin
importar cuán creativo o cuán buggy sea el código generado.

Requiere que los datos lleguen **por referencia declarada**. Un path hardcodeado o una
consulta a base anula el borde.

La disciplina de holdout —val conduce, test confirma— vuelve al alcance por la tesis:
cuando el usuario escribía el código, era una buena práctica suya; ahora el código lo
escribe el agente, y no se le puede pedir al usuario que responda por código que no
escribió.

### 4. B se parte en tres

Verificado: `materialize_variant` y `smoke_check` se llaman **únicamente desde
`tests/test_variants.py`**. El orquestador sabe consumir un `variant_id`
(`orchestrator.py:587,741`) pero nada lo produce. `AgentStrategy.plan` devuelve solo
params. La modalidad de código está escrita, testeada y es inalcanzable.

| | Defecto | Tamaño | ¿Necesita máquina de etapas? |
|---|---|---|---|
| B1 | `loop.py` no importa el daemon: NaN, timeout y procesos muertos no se chequean en el camino autónomo | S | No |
| B2 | `_review` corre después de `advance`, con la cohorte siguiente ya enviada y pagada | S/M | No |
| B3 | Variantes inalcanzables end-to-end | L | Sí |

B1 es el más grave por contradicción: el modo diseñado para correr de noche sin
supervisión es exactamente el que no supervisa.

### 5. El implementador es un despacho de tarea

`CliAgentRunner` es una llamada one-shot (`claude -p --output-format json`), y
`VariantEdit` es `path + content`: el planner debe escupir archivos completos como
strings dentro de un JSON, sin haber leído el repo, sin correr nada, sin iterar. Sirve
para editar un archivo chico y conocido; no para crear implementaciones.

```
1. iax materializa    variants/var_ab12/   (copia del proyecto, sin test data)
2. iax despacha       agente con herramientas, cwd = la variante, brief = la hipótesis
3. el agente          explora, edita, corre el smoke check, itera
4. iax registra       diff contra source -> edited_paths; transcript de la sesión
```

Los roles se separan: **planner** (qué hipótesis, qué params) sigue siendo una llamada
barata de JSON; **implementer** (escribí esto) es una sesión.

Esto es viable *porque* las decisiones 2 y 3 son bordes estructurales. Una protección
basada en validar `editable_paths` sobre un payload se rompería al darle herramientas
de archivo al agente; los bordes no dependen de inspeccionar lo que el agente propone.

### 6. SQLite como fuente transaccional

```python
def write_state(self, state: CampaignState) -> None:   # store/campaign.py:35
    atomic_write_text(..., json.dumps(state.model_dump(mode="json")))
```

Escritura atómica del documento entero, sin lock ni versión: CLI, daemon y servidor
hacen read-modify-write sobre el mismo archivo y el último que escribe pisa al otro.
Con las decisiones anteriores un trial tiene más estados que antes —variante
materializada, sesión del agente, train, evaluate— así que hay más para perder.

SQLite guarda estado, intentos, decisiones y reservas. Los archivos siguen para logs,
artefactos, transcripts y exportaciones legibles. No se ubica en un filesystem
compartido entre hosts: los workers remotos reportan al controlador.

Se pierde el `cat state.json` que hoy hace todo inspeccionable; se compensa con
exportación legible desde la fachada.

### 7. Presupuesto

- **Reservas en la misma transacción** que la intención de submit: admisión con
  enforcement real, no con polling.
- **Techo duro por intento**: un límite de wall-clock derivado de los GPU-hours
  reservados y pasado a la ejecución, para que el worker se mate solo al vencer. El
  polling queda de red de seguridad. Sin esto el techo es una estimación con margen,
  no una promesa.
- **Dos presupuestos separados**: cómputo (GPU-hours) y agente (tokens o tiempo de
  sesión). `AgentSpec.max_calls: 20` acota un planner; no acota una sesión de
  codificación, que puede costar dos órdenes de magnitud más.
- **El agente no toca el techo.** `loop.py::_apply_changes` acepta hoy cambios sobre
  `budget`: se elimina esa clave. Reducir o redistribuir dentro del techo es válido;
  ampliarlo requiere autorización del usuario.

## Efecto sobre los hallazgos del assessment

| | Qué sobrevive | Qué cambia |
|---|---|---|
| **A** | Elegibilidad por trial completado; métrica ausente con causa | El núcleo es el canal de resultado, no el protocolo versionado. Réplicas, confirmación de finalistas y versionado de dataset/split se difieren |
| **B** | El diagnóstico, entero | Se parte: B1/B2 P0 chicos; B3 es la tesis, no fase 6 |
| **C** | Transacciones, identidad de intento, reconciliación | Baja de P0 a después de la puerta: su falla es perder trabajo, no mentir |
| **D** | Estado deseado vs. observado, reservas, techo inmutable | Gana la dimensión de presupuesto de agente por sesión |
| **E** | Contrato de adapter, argv exacto, recursos, observaciones con `attempt_id` | Artefactos bajan de prioridad: el handoff train→evaluate es local al nodo |
| **F** | Snapshot inmutable, deps bloqueadas, distinguir reintentar/reproducir/re-ejecutar | El linaje es diff contra source + transcript de sesión |
| **G** | Runner, transcripts, validación, fallback según causa | Se agrega el contrato del implementador, que es la pieza de la tesis |
| **H** | Supervisión breve, agente y tracking en tareas aparte | "Monitorear por fase" deja de ser aspiracional: las fases existen |
| **I** | Evaluador y holdout fuera del mandato | **Sube de P1 a columna de corrección.** El resto de I (secretos, aislamiento, auth) queda según modo de despliegue |
| **J** | Fachada única, `report.json`, informe legible | El informe muestra el diff del candidato, no solo params |

## Plan

El assessment ordena el trabajo para endurecer primero y probar la tesis al final. Con
la tesis en el centro eso invierte el riesgo: no se sabe si el producto existe hasta
haber pagado todo. Pero tampoco se puede probar antes, porque sobre la base actual el
resultado no significa nada.

El orden correcto es el **circuito honesto más corto**: lo mínimo para que "el agente
mejoró la métrica" sea una afirmación creíble, y recién después la puerta.

| Fase | Entregables | Puerta de salida |
|---|---|---|
| **0. Consolidar** | Matriz de integración de `fae2c00` y `c4f0453`; fixes de elegibilidad, métrica ausente, validación y agotamiento; base canónica instalable | Una revisión exacta instalable, sin asumir que la rama más nueva contiene todo |
| **1. Circuito honesto** | `IAX_RESULT` · split train/evaluate · borde de datos (ver preguntas abiertas) · argv de Ray · `loop` supervisa y cierra cohorte antes de admitir | Un candidato que escribe su propio resultado no puntúa; un trial fallido no gana; una campaña local completa y supervisada |
| **2. La tesis** | Variantes conectadas end-to-end · despacho al implementador · contratos tipados · presupuesto de sesión · diff y transcript como linaje | Una variante rota se rechaza antes de costar una ronda; una variante evaluada produce evidencia |
| — | **PUERTA: ¿produce vjepa una mejora confirmada, revisable y reproducible?** | Todo lo de abajo se paga después de esta respuesta |
| **3. Durabilidad** | SQLite, reservas, cancelación confirmada, identidad de intento, reconciliación | Dos clientes y reinicios no duplican trabajo ni pierden gasto |
| **4. Ray completo** | Recursos y capacidades, tiempos reales, observaciones incrementales, transporte de artefactos | Campaña remota con desconexión, artefacto recuperado, costo reconciliado |
| **5. Linaje** | Snapshots, deps bloqueadas, bundle reproducible | Otro entorno reproduce el candidato o reporta qué falta |
| **6. Producto** | Fachada única, dashboard de decisiones, informe, docs | Usuario nuevo completa una campaña sin conocer internals |

C, D y E bajan de P0 a después de la puerta. No porque no importen: porque su falla es
*perder trabajo*, y la de la fase 1 es *producir una conclusión falsa con evidencia
impecable*. Solo una de las dos lleva a una decisión equivocada sobre vjepa.

**Riesgo aceptado:** correr la fase 2 sin durabilidad significa que un crash de
madrugada pierde la campaña. Se acota con campañas chicas y techo duro por intento. Es
barato frente a endurecer durante meses un circuito cuyo valor todavía no se demostró.

Cada fase entrega una sección vertical ejecutable.

**Alcance del plan de implementación:** fases 0 a 2, hasta la puerta. Las fases 3 a 6
quedan descriptas acá para fijar el orden, pero se planifican después de conocer la
respuesta de vjepa — planificarlas antes sería comprometerse a construirlas sin saber
si la tesis se sostiene.

## Criterios de aceptación

Los de la fase 1 y 2 son los que definen si el producto es confiable:

1. Un candidato que escribe `report_result(...)` en su código de train **no puntúa**.
2. El proceso de train **no puede abrir** el test set.
3. Un trial que falla no puede ganar, aunque haya reportado un resultado.
4. Un workload sin `report_result` falla ruidosamente, sin caer al último `IAX_METRIC`.
5. Una variante que no arranca se descarta antes de costar una ronda.
6. Ningún reintento, propuesta o cambio de goal amplía el techo de presupuesto.
7. El mismo comando que corre la campaña supervisa NaN, timeout y procesos muertos.
8. Una cohorte se cierra y se revisa antes de admitir la siguiente.
9. El informe de un ganador incluye el diff contra el source y el transcript que lo
   produjo.

## Preguntas abiertas

- **Acceso a datos (bloquea la decisión 3).** Cómo accede hoy un workload a sus
  datasets: ruta montada, variable de entorno, descarga desde bucket o path hardcodeado
  en el repo. De eso depende si el borde train/test se impone con configuración o si
  el proyecto del usuario debe declarar sus datasets primero.
- **Modo de despliegue.** Se asume herramienta individual, ejecución local de
  confianza. Un agente con herramientas corre código que él mismo escribió, con el
  entorno del proceso, donde viven `MLFLOW_TRACKING_URI` y credenciales. Si pasa a
  acceso compartido, hace falta perfil aislado, referencias a secretos y auth.
- **Identidad de protocolo en el leaderboard.** Dos `test_acc` de datasets distintos
  siguen agrupando juntos. Diferido: solo molesta al comparar entre campañas.
- **Smoke command por defecto** para una variante recién generada.

## Lo que no se hace

Microservicios, Kubernetes propio, framework genérico de agentes, plataforma
multiusuario, alta disponibilidad, múltiples controladores distribuidos, optimizador
sofisticado. Ray sigue como ejecución y MLflow como integración opcional, sin
delegarles la verdad de la campaña.
