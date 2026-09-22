"""El workload: un trial de la campaña maintenance-events.

Contrato con iax: los parámetros llegan como ``--nombre-valor`` (las claves de
``search_space`` son identificadores de Python y ``build_trial_manifest`` las
traduce a la grafía con guiones que argparse declara por convención), y las
observaciones salen por stdout como líneas ``IAX_METRIC {json}``.

Cada fold es una observación del objetivo, no un paso hacia él: emite
``pr_auc`` junto a su ``baseline_pr_auc`` en la misma línea. La meta declara
``objective.aggregate: mean``, así que el arnés promedia los folds, calcula el
error estándar y decide con eso si el trial se distingue del ruido.

El resumen final va con otras claves a propósito — repetir la clave objetivo
sería una observación de más, el promedio contado dos veces. Ahí viaja también
el bootstrap por bloques sobre las predicciones agrupadas, como diagnóstico:
por qué no es el objetivo está en el README, sección "Por qué el objetivo sigue
siendo el promedio de folds".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from maintenance_events import dataset as ds
from maintenance_events.evaluation import (
    HORIZON_DAYS_DEFAULT as BOOTSTRAP_BLOCK,
)
from maintenance_events.evaluation import block_bootstrap, evaluate, folds_to_frame
from maintenance_events.features import build_feature_table
from maintenance_events.models import MODELS, make_model
from maintenance_events.synthetic import synthetic_dataset
from maintenance_events.windows import (
    HORIZON_DAYS,
    build_windows,
    label_coverage,
    select_events,
)


def report_metric(**values: object) -> None:
    print("IAX_METRIC " + json.dumps(values))
    sys.stdout.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--resample-freq", default="1h")
    parser.add_argument(
        "--label-source", default="union", choices=["mapro", "operator", "union"]
    )
    parser.add_argument("--model", default="hist_gb", choices=list(MODELS))
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--max-bins", type=int, default=255)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=500,
        help="réplicas del remuestreo; cada una es una observación del objetivo",
    )
    parser.add_argument(
        "--bootstrap-block",
        type=int,
        default=BOOTSTRAP_BLOCK,
        help="largo del tramo contiguo que se remuestrea, en ventanas (1 día c/u)",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=20,
        help="positivas out-of-fold por debajo de las cuales no se reporta el diagnóstico",
    )
    parser.add_argument("--offline-power-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ignore-label-coverage",
        action="store_true",
        help="etiqueta negativas las ventanas fuera del período registrado por la fuente",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="corre sobre datos sintéticos"
    )
    return parser.parse_args(argv)


def resolve_thread_limit() -> int | None:
    """Cuántos hilos puede usar BLAS/OpenMP dentro de un trial.

    iax corre varios trials en paralelo, así que cada uno oversubscribe la
    máquina. Con 435 ventanas el trabajo por hilo es minúsculo y la
    sincronización domina: medido, el mismo fit tarda 22.9 s con los 16 hilos
    del host y 9.9 s con uno solo. El default es 1; ``0`` devuelve los límites
    de sklearn por si alguien corre un trial aislado.
    """
    raw = os.environ.get("MAINTENANCE_EVENTS_THREADS", "1")
    limit = int(raw)
    return None if limit <= 0 else limit


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        signals, events = synthetic_dataset(seed=args.seed)
    else:
        signals, events = ds.load()

    resampled = signals.resample(args.resample_freq).mean()
    event_dates = select_events(events, args.label_source)
    coverage = (
        None
        if args.ignore_label_coverage
        else label_coverage(events, args.label_source)
    )
    windows = build_windows(
        resampled, event_dates, window_days=args.window_days, coverage=coverage
    )
    X, y, as_of = build_feature_table(
        windows, offline_power_threshold=args.offline_power_threshold
    )

    if X.empty:
        print("sin ventanas válidas para esta configuración", file=sys.stderr)
        # Sin folds válidos no hay objetivo que reportar, y no hay que
        # inventar uno: el arnés marca el trial como `not_finite` con el
        # motivo, que es más útil que un cero indistinguible de un modelo malo.
        report_metric(
            step=0,
            mean_pr_auc=0.0,
            pr_auc_std=0.0,
            auroc=0.0,
            recall_at_p50=0.0,
            mean_baseline=0.0,
            valid_folds=0,
            pooled_windows=0,
            pooled_positives=0,
            replicates=0,
            lift_mean=0.0,
            lift_stderr=0.0,
        )
        return 0

    factory = make_model(
        args.model,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2=args.l2,
        max_bins=args.max_bins,
        class_weight=None
        if args.class_weight in ("", "none", "None")
        else args.class_weight,
        seed=args.seed,
    )
    with threadpool_limits(limits=resolve_thread_limit()):
        folds, summary, pooled = evaluate(
            X, y, as_of, factory, n_folds=args.n_folds, horizon_days=HORIZON_DAYS
        )

    for fold in folds:
        report_metric(**fold.as_metric())

    # El bootstrap por bloques sobre las predicciones agrupadas, como
    # *diagnóstico*. Ninguna de sus claves es la del objetivo: medido sobre
    # estos datos el estimador agrupado contradice a los folds —da lift
    # negativo mientras cada fold da positivo— porque agrupa scores de modelos
    # distintos sobre períodos con distinta tasa base, y eso mide deriva de
    # calibración, no discriminación. Ver README § "Por qué el objetivo sigue
    # siendo el promedio de folds". Se reporta igual porque es la medición que
    # sostiene esa decisión, y porque `lift_stderr` con bloque 15 es el único
    # número honesto sobre cuánta suerte hay en una ventana.
    replicates: list[tuple[float, float]] = []
    if summary["pooled_positives"] >= args.min_positives and args.bootstrap_resamples:
        replicates = block_bootstrap(
            pooled,
            n_resamples=args.bootstrap_resamples,
            block=args.bootstrap_block,
            seed=args.seed,
        )
    elif args.bootstrap_resamples:
        print(
            f"sólo {summary['pooled_positives']} positivas out-of-fold "
            f"(mínimo {args.min_positives}): sin diagnóstico de bootstrap",
            file=sys.stderr,
        )

    lifts = [pr - base for pr, base in replicates]
    summary["replicates"] = len(replicates)
    summary["lift_mean"] = float(np.mean(lifts)) if lifts else 0.0
    summary["lift_stderr"] = float(np.std(lifts, ddof=1)) if len(lifts) > 1 else 0.0
    report_metric(step=len(folds), **summary)

    artifacts = os.environ.get("IAX_ARTIFACTS_DIR")
    if artifacts:
        out = Path(artifacts)
        folds_to_frame(folds).to_csv(out / "folds.csv", index=False)
        pd.concat([X, y], axis=1).to_parquet(out / "features.parquet")
        (out / "summary.json").write_text(
            json.dumps({**summary, "params": vars(args)}, indent=2, default=str)
        )

    fold_lift = summary["mean_pr_auc"] - summary["mean_baseline"]
    print(
        f"lift={fold_lift:+.4f} sobre {summary['valid_folds']} folds válidos "
        f"({len(X)} ventanas); diagnóstico agrupado "
        f"{summary['lift_mean']:+.4f} ± {summary['lift_stderr']:.4f} "
        f"sobre {summary['pooled_windows']} ventanas out-of-fold, "
        f"{summary['replicates']} réplicas de bloque {args.bootstrap_block}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
