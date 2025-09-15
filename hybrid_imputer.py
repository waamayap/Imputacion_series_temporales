
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, List
from approach import BaseImputer

def _remove_outliers_frame(frame: np.ndarray, qlow=0.005, qhigh=0.995) -> np.ndarray:
    """Marca como NaN los valores fuera de los cuantiles [qlow, qhigh] por columna."""
    X = pd.DataFrame(frame)
    for c in X.columns:
        lo = X[c].quantile(qlow)
        hi = X[c].quantile(qhigh)
        X.loc[(X[c] < lo) | (X[c] > hi), c] = np.nan
        # opcional: descartar negativos en PM2.5
        X.loc[X[c] < 0, c] = np.nan
    return X.values

class HybridImputer(BaseImputer):
    """
    Híbrido:
      - Huecos ≤ max_lin_gap -> interpolación lineal columna a columna
      - Huecos > max_lin_gap -> KNN (u otro imputador con misma interfaz)
    """
    def __init__(self, knn_imputer: BaseImputer, max_lin_gap: int = 6,
                 name: str = "Hybrid (Lin≤6 + KNN)"):
        self.name = name
        self.knn_imputer = knn_imputer
        self.max_lin_gap = max_lin_gap

    def fit(self, sequences: Sequence[np.ndarray]) -> None:
        # 🔹 Filtrado básico de atípicos ANTES de entrenar KNN
        clean_sequences: List[np.ndarray] = [
            _remove_outliers_frame(seq) for seq in sequences
        ]
        self.knn_imputer.fit(clean_sequences)

    def predict(self, sequence_with_nan: np.ndarray) -> np.ndarray:
        arr = sequence_with_nan.copy()
        T, V = arr.shape

        # 1) Interpolación lineal para tramos cortos por columna
        arr_lin = arr.copy()
        for j in range(V):
            col = arr[:, j]
            isnan = np.isnan(col)
            if not isnan.any():
                continue
            nan_idx = np.where(isnan)[0]
            segments: List[Tuple[int, int]] = []
            start = nan_idx[0]; prev = nan_idx[0]
            for idx in nan_idx[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    segments.append((start, prev - start + 1))
                    start = idx; prev = idx
            segments.append((start, prev - start + 1))

            col_series = pd.Series(col, dtype="float64")
            for seg_start, seg_len in segments:
                if seg_len <= self.max_lin_gap:
                    lower = seg_start - 1
                    upper = seg_start + seg_len
                    if lower < 0 and upper >= T:
                        continue  # toda la columna NaN
                    elif lower < 0:
                        col_series.iloc[seg_start: seg_start+seg_len] = float(col_series.iloc[upper])
                    elif upper >= T:
                        col_series.iloc[seg_start: seg_start+seg_len] = float(col_series.iloc[lower])
                    else:
                        x0, x1 = lower, upper
                        y0 = float(col_series.iloc[lower]); y1 = float(col_series.iloc[upper])
                        xs = np.arange(seg_start, seg_start + seg_len)
                        ys = np.interp(xs, [x0, x1], [y0, y1])
                        col_series.iloc[seg_start: seg_start+seg_len] = ys
            arr_lin[:, j] = col_series.values

        # 2) 🔹 Filtrar atípicos ANTES de KNN en la secuencia completa
        seq_for_knn = _remove_outliers_frame(sequence_with_nan)
        knn_pred = self.knn_imputer.predict(seq_for_knn)

        # 3) Fusionar: mantener lo imputado por lineal; resto tomar de KNN
        final = arr_lin.copy()
        final[np.isnan(arr_lin)] = knn_pred[np.isnan(arr_lin)]
        return final

    # Helpers
    def set_knn_imputer(self, knn_imputer: BaseImputer) -> None:
        self.knn_imputer = knn_imputer

    def set_max_linear_gap(self, max_lin_gap: int) -> None:
        self.max_lin_gap = max_lin_gap
