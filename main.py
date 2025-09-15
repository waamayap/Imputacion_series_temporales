"""
Benchmark multivariable para imputación de datos de polución,
con conteo de secuencias NO solapadas limpias y huecos sintéticos por clase.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List
from collections import Counter

import numpy as np
import pandas as pd

# ──────────── imports del proyecto ──────────────────────────────────────
from timeseriesdata import ConcreteTimeSeriesData
from benchmark import ConcreteBenchmark
from statistical import (
    MeanImputer,
    MedianImputer,
    ForwardFillImputer,
    BackwardFillImputer,
)
from machine_learning import KNNImputer, LSTMImputer, CNNImputer
from interpolation import LinearInterpolationImputer, SplineInterpolationImputer
from hybrid_imputer import HybridImputer  # interpolación lineal ≤6h + CNN

# ──────────── logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ──────────── argumentos CLI ────────────────────────────────────────────
parser = argparse.ArgumentParser("Pollution-imputation benchmark")
parser.add_argument("--csv",      default="Base_consolidada_santiago.csv")
parser.add_argument("--start",    default="2020-01-01")
parser.add_argument("--end",      default="2024-12-31")
parser.add_argument("--time_int", type=int, default=360)  # tamaño de ventana (horas)
parser.add_argument("--no_gui",   action="store_true")
args = parser.parse_args()

PROJECT_DIR = os.path.dirname(__file__)

# ──────────── 1. Cargar y preparar el tensor ────────────────────────────
csv_path = os.path.join(PROJECT_DIR, args.csv)
ts = ConcreteTimeSeriesData()
df_raw = ts.load_data(csv_path)
log.info("CSV cargado: %s filas=%d columnas=%d", csv_path, *df_raw.shape)

stations, variables, mapping = ts.identify_stations_and_variables(df_raw, "fecha_hora")
seleccion: Dict[str, List[str]] = ts.seleccionar_estacion_y_variables_gui(mapping, headless=args.no_gui)
if not any(seleccion.values()):
    log.error("No se seleccionó ninguna variable ➜ abortando.")
    sys.exit(1)

df_filtrado = ts.set_index(
    ts.filter_data_by_date(df_raw, args.start, args.end),
    "fecha_hora"
)
tensor = ts.build_tensor(df_filtrado, seleccion, mapping)
log.info("Tensor construido: %s", tensor.shape)

variables_global = sorted({v for vars_list in seleccion.values() for v in vars_list})

# ──────────── 2. Resumen de secuencias  limpias ────────────
ventana = args.time_int
train_ratio = 0.7

mat_global, _ = ConcreteBenchmark(
    ts=ts,
    data_tensor=tensor,
    imputers=[],
    time_interval=ventana,
    estaciones=list(seleccion.keys()),
    variables=variables_global,
    seleccion=seleccion,
    seed=0
)._build_mat_global()

T = mat_global.shape[0]
L = ventana
N = T // L

# 1) Extraer solo las ventanas  limpias
clean_nonoverlap = []
for i in range(N):
    start = i * L
    end   = start + L
    block = mat_global[start:end, :]
    if not np.isnan(block).any():
        clean_nonoverlap.append(block)

log.info("Ventanas independientes limpias: %d de %d posibles", len(clean_nonoverlap), N)

# 2) División 70/30 sobre las limpias
split_n    = int(len(clean_nonoverlap) * train_ratio)
train_seqs = clean_nonoverlap[:split_n]
val_seqs   = clean_nonoverlap[split_n:]

log.info("  → Train (70%%): %d secuencias", len(train_seqs))
log.info("  → Val   (30%%): %d secuencias", len(val_seqs))


# ──────────── 3. Lista de imputadores ───────────────────────────────────
imputers: List[BaseImputer] = [
    MeanImputer(name="Mean"),
    ForwardFillImputer(name="ForwardFill"),

    BackwardFillImputer(name="BackwardFill"),
    #KNNImputer(name="KNN", n_neighbors=4, weights="distance"),
    #LSTMImputer(name="LSTM", window_size=5, epochs=200, verbose=1),
    #CNNImputer(name="CNN", window_size=5, epochs=200, verbose=1),
    LinearInterpolationImputer(name="LinearInt"),

    # ⬇⬇⬇ AQUÍ EL HÍBRIDO MODIFICADO (antes usaba cnn_imputer=...) ⬇⬇⬇
    #HybridImputer(
     #   knn_imputer=KNNImputer(name="KNN_for_Hybrid", n_neighbors=24, weights="distance"),
      #  max_lin_gap=6,
       # name="Hybrid (Lin≤6h + KNN)"
    #),
]

# ──────────── 4. Ejecutar benchmark ─────────────────────────────────────
bench = ConcreteBenchmark(
    ts=ts,
    data_tensor=tensor,
    imputers=imputers,
    time_interval=ventana,
    estaciones=list(seleccion.keys()),
    variables=variables_global,
    seleccion=seleccion,
    seed=100
)

gap_percentage = 0.2
class_probs   = {"class_1": .25, "class_2": .25, "class_3": .25, "class_4": .25}

results = bench.evaluate(
    gap_percentage=gap_percentage,
    train_ratio=train_ratio,
    class_probs=class_probs,
    show_bar=True
)

# ─── Conteo de huecos sintéticos por clase en validación ────────────────
# ─── Conteo de huecos sintéticos por clase en validación ────────────────
if val_seqs:
    # Conteo total en todas las secuencias de validación
    total_counts = Counter()
    for seq in val_seqs:
        _, gaps_info = bench._inject_by_column(seq, gap_percentage, class_probs)
        flat = [cls for col in gaps_info for (cls, _, _) in col]
        total_counts.update(flat)

    log.info("Total huecos sintéticos en validación por clase (todas secuencias):")
    for clase, cnt in total_counts.items():
        log.info("  %s: %d", clase, cnt)

    # Conteo en la primera secuencia de validación (opcional)
    seq0 = val_seqs[0]
    _, gaps_info0 = bench._inject_by_column(seq0, gap_percentage, class_probs)
    counts0 = Counter([cls for col in gaps_info0 for (cls, _, _) in col])
    log.info("Huecos sintéticos en la primera secuencia de validación por clase:")
    for clase, cnt in counts0.items():
        log.info("  %s: %d", clase, cnt)

# ──────────── 5. Guardar resultados ────────────────────────────────────
if results is None or results.empty:
    log.warning("No se obtuvieron resultados del benchmark (¿pocas ventanas limpias?).")
else:
    output_dir = bench.output_dir
    pd.DataFrame(results).to_excel(
        os.path.join(output_dir, "resultados_benchmark.xlsx"), index=False
    )
    log.info("Resultados exportados en %s", os.path.join(output_dir, "resultados_benchmark.xlsx"))

log.info("✅ Benchmark completo. Archivos en: %s", bench.output_dir)
