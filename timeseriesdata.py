

from __future__ import annotations
import logging
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from approach import TimeSeriesData


log = logging.getLogger(__name__)

class ConcreteTimeSeriesData(TimeSeriesData):
    """Manejo de:
      • Lectura y limpieza del CSV
      • Mapeo de columnas a (estación, variable)
      • Construcción del tensor 3D (estación × variable × tiempo)
    """

    def __init__(self, data_tensor: Optional[np.ndarray] = None):
        """Inicializa la clase con un tensor de datos opcional."""
        self.data = data_tensor  # Se setea luego cuando se construye el tensor

    # ------------------------------------------------------------------
    # 1. LECTURA Y LIMPIEZA DEL CSV
    # ------------------------------------------------------------------
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Carga el archivo CSV, normaliza encabezados (minúsculas y sin espacios)
        y elimina la columna 'anio_mes' si existe."""
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()

        if "anio_mes" in df.columns:
            df = df.drop(columns="anio_mes")

        return df

    # ------------------------------------------------------------------
    # 2. IDENTIFICACIÓN DE ESTACIONES Y VARIABLES
    # ------------------------------------------------------------------
    _col_regex = re.compile(r"([a-z0-9_]+)_([a-z0-9_]+)", flags=re.I)  # Patrón para separar estación_variable

    def identify_stations_and_variables(
        self, df: pd.DataFrame, datetime_col: str | None = None
    ) -> Tuple[List[str], List[str], Dict[str, Dict[str, str]]]:
        """Identifica estaciones, variables y mapea columnas del DataFrame."""
        cols = [c for c in df.columns if c != datetime_col]
        stations, variables, mapping = set(), set(), {}

        for col in cols:
            m = self._col_regex.fullmatch(col)
            if m:
                st, var = m.group(1).lower(), m.group(2).lower()
                stations.add(st)
                variables.add(var)
                mapping.setdefault(st, {})[var] = col

        # Si no se encuentra ningún patrón, se asigna todo a una estación ficticia
        if not stations:
            stations.add("station1")
            num_cols = df.select_dtypes("number").columns
            mapping = {"station1": {c: c for c in num_cols}}
            variables.update(num_cols)

        return sorted(stations), sorted(variables), mapping

    # ------------------------------------------------------------------
    # 3. FILTRADO POR FECHA Y DEFINICIÓN DE ÍNDICE
    # ------------------------------------------------------------------
    def filter_data_by_date(self, df: pd.DataFrame, start_date: str, end_date: str):
        """Filtra el DataFrame por un rango de fechas."""
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
        return df[(df["fecha_hora"] >= start_date) & (df["fecha_hora"] <= end_date)]

    def set_index(self, df: pd.DataFrame, index_column: str):
        """Define una columna como índice de fechas y ordena el DataFrame."""
        df[index_column] = pd.to_datetime(df[index_column])
        return df.set_index(index_column).sort_index()

    # ------------------------------------------------------------------
    # 4. SELECCIÓN DE VARIABLES (CON O SIN INTERFAZ GRÁFICA)
    # ------------------------------------------------------------------
    def seleccionar_estacion_y_variables_gui(
        self, variable_columns: Dict[str, Dict[str, str]], *, headless=False
    ) -> Dict[str, List[str]]:
        """Permite seleccionar interactivamente qué variables utilizar,
        o selecciona todas en modo 'headless'."""

        if headless:
            # Si no hay GUI, seleccionar todas las variables
            return {st: list(vars.keys()) for st, vars in variable_columns.items()}

        # ── Ventana interactiva usando Tkinter ──
        import tkinter as tk
        from tkinter import ttk

        seleccion: Dict[str, List[str]] = {}

        def _confirmar():
            """Función que recoge la selección al presionar 'Confirmar'."""
            for estacion, checks in checks_por_estacion.items():
                seleccion[estacion] = [v for v, obj in checks.items() if obj.get()]
            root.destroy()

        root = tk.Tk()
        root.title("Seleccionar Variables")
        root.geometry("500x600")

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Selecciona las variables que deseas usar",
                  font=("Arial", 12, "bold")).pack(pady=10)

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        checks_por_estacion: Dict[str, Dict[str, tk.BooleanVar]] = {}
        for est in sorted(variable_columns):
            ttk.Label(scroll_frame, text=est.upper(), font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 0))
            ck_dict: Dict[str, tk.BooleanVar] = {}
            for var in sorted(variable_columns[est]):
                col_name = variable_columns[est][var]
                var_obj = tk.BooleanVar()
                tk.Checkbutton(scroll_frame, text=col_name, variable=var_obj).pack(anchor="w")
                ck_dict[var] = var_obj
            checks_por_estacion[est] = ck_dict

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        ttk.Button(root, text="Confirmar selección", command=_confirmar).pack(pady=10)
        root.mainloop()

        return seleccion

    # ------------------------------------------------------------------
    # 5. CONSTRUCCIÓN DEL TENSOR 3D
    # ------------------------------------------------------------------
    def build_tensor(
        self,
        df: pd.DataFrame,
        seleccion: Dict[str, List[str]],
        mapping: Dict[str, Dict[str, str]],
    ) -> np.ndarray:
        """Construye un tensor 3D con dimensiones (estaciones × variables × tiempo)."""
        series_dict, lens = {}, []

        for st, vars_sel in seleccion.items():
            for var in vars_sel:
                col = mapping[st][var]
                arr = df[col].to_numpy()
                series_dict[(st, var)] = arr
                lens.append(arr.size)

        min_len = min(lens)  # Longitud mínima entre todas las series
        stations_sorted = sorted(seleccion)
        vars_sorted = sorted({v for vars_ in seleccion.values() for v in vars_})

        tensor = np.full((len(stations_sorted), len(vars_sorted), min_len), np.nan)

        for i, st in enumerate(stations_sorted):
            for j, var in enumerate(vars_sorted):
                if (st, var) in series_dict:
                    tensor[i, j, :] = series_dict[(st, var)][:min_len]

        self.data = tensor
        return tensor

    # ------------------------------------------------------------------
    # 6. EXTRACCIÓN Y DIVISIÓN DE SECUENCIAS
    # ------------------------------------------------------------------
    def get_complete_sequences(self, time_interval: int) -> np.ndarray:
        """Extrae sub-secuencias completas (sin NaNs) de longitud 'time_interval'."""
        if self.data is None:
            raise RuntimeError("Se debe construir el tensor antes de extraer secuencias.")

        est, var, t = self.data.shape
        seqs = [
            self.data[e, v, i:i + time_interval]
            for e in range(est)
            for v in range(var)
            for i in range(0, t - time_interval + 1, time_interval)
            if not np.isnan(self.data[e, v, i:i + time_interval]).any()
        ]
        return np.asarray(seqs, dtype=float)

    def split_sequences(
        self, sequences: np.ndarray, train_ratio: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Divide las secuencias en conjuntos de entrenamiento y validación."""
        n_train = int(len(sequences) * train_ratio)
        return sequences[:n_train], sequences[n_train:]
    








# ------------------------------------------------------------------
# 7. ESCALADOR GLOBAL MIN-MAX
# ------------------------------------------------------------------
class MinMaxScaler:
    """Escalador Min-Max sencillo que ignora valores NaN.

    Funcionalidades:
    - fit: calcula mínimos y máximos globales
    - transform: normaliza al rango [0,1]
    - inverse_transform: revierte la normalización
    """

    def __init__(self):
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None

    def fit(self, sequences):
        """Calcula el mínimo y el máximo para cada variable."""
        data = np.concatenate([
            seq.reshape(-1, seq.shape[-1]) if seq.ndim == 2
            else seq[:, np.newaxis]
            for seq in sequences
        ], axis=0)
        self.min = np.nanmin(data, axis=0)
        self.max = np.nanmax(data, axis=0)

    def transform(self, sequence):
        """Aplica la normalización Min-Max."""
        return (sequence - self.min) / (self.max - self.min + 1e-8)

    def inverse_transform(self, sequence):
        """Revierte la normalización Min-Max."""
        return sequence * (self.max - self.min + 1e-8) + self.min

