"""
Métodos estadísticos simples para imputar valores faltantes
(tanto en series **univariables** como **multivariables**),
usando relleno local por huecos.
"""
from __future__ import annotations

import numpy as np
from typing import Any, List
from approach import BaseImputer   # interfaz común que todos los imputadores deben cumplir


# ───────────────────────── FUNCIONES AUXILIARES ─────────────────────────
def _to_2d(arr: np.ndarray) -> np.ndarray:
    """
    Asegura que la entrada sea siempre 2-D (T, n_features).

    Si llega un vector 1-D (T,), lo convierte a (T, 1) sin copiar datos.
    """
    if arr.ndim == 1:
        return arr[:, np.newaxis]   # añade un eje de características
    return arr                      # si ya es 2-D se devuelve tal cual


def _from_2d(arr2d: np.ndarray, original_ndim: int) -> np.ndarray:
    """
    Devuelve la salida con la misma dimensionalidad que la entrada original.
    Si la entrada era 1-D, «aplana» el array 2-D; si era 2-D lo deja igual.
    """
    if original_ndim == 1:
        return arr2d.ravel()
    return arr2d


# ───────────────────────── IMPUTADORES LOCALES ─────────────────────
class MeanImputer(BaseImputer):
    """
    Imputa bloques de NaN usando la media local de cada hueco,
    calculada a partir de los dos valores vecinos inmediatos.
    """
    def __init__(self, name: str = "Mean Imputation"):
        self.name = name

    def fit(self, sequences: List[np.ndarray]) -> Any:
        # No requiere entrenamiento
        return self

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        seq2d = _to_2d(sequence).copy()
        T, n_feat = seq2d.shape

        for f in range(n_feat):
            col = seq2d[:, f]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue

            # vector binario de NaNs para detectar bloques
            mask_int = nan_mask.astype(int)
            d = np.diff(np.concatenate([[0], mask_int, [0]]))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]

            for st, ed in zip(starts, ends):
                # vecinos inmediatos antes y después del hueco
                left_val = col[st - 1] if st > 0 else np.nan
                right_val = col[ed] if ed < T else np.nan

                if np.isnan(left_val) and np.isnan(right_val):
                    fill = np.nan
                elif np.isnan(left_val):
                    fill = right_val
                elif np.isnan(right_val):
                    fill = left_val
                else:
                    fill = (left_val + right_val) / 2

                # rellenar todo el hueco
                col[st:ed] = fill

            seq2d[:, f] = col

        return _from_2d(seq2d, sequence.ndim)


class MedianImputer(BaseImputer):
    """
    Imputa bloques de NaN usando la mediana local de cada hueco,
    calculada a partir de los dos valores vecinos inmediatos.
    """
    def __init__(self, name: str = "Median Imputation"):
        self.name = name

    def fit(self, sequences: List[np.ndarray]) -> Any:
        # No requiere entrenamiento
        return self

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        seq2d = _to_2d(sequence).copy()
        T, n_feat = seq2d.shape

        for f in range(n_feat):
            col = seq2d[:, f]
            nan_mask = np.isnan(col)
            if not nan_mask.any():
                continue

            mask_int = nan_mask.astype(int)
            d = np.diff(np.concatenate([[0], mask_int, [0]]))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]

            for st, ed in zip(starts, ends):
                left_val = col[st - 1] if st > 0 else np.nan
                right_val = col[ed] if ed < T else np.nan

                if np.isnan(left_val) and np.isnan(right_val):
                    fill = np.nan
                elif np.isnan(left_val):
                    fill = right_val
                elif np.isnan(right_val):
                    fill = left_val
                else:
                    fill = np.median([left_val, right_val])

                col[st:ed] = fill

            seq2d[:, f] = col

        return _from_2d(seq2d, sequence.ndim)


class ForwardFillImputer(BaseImputer):
    """
    Imputa propagando hacia delante el **último valor válido**.
    """
    def __init__(self, name: str = "Forward Fill"):
        self.name = name

    def fit(self, sequences: List[np.ndarray]) -> Any:
        return self

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        seq2d = _to_2d(sequence).copy()
        T, n_feat = seq2d.shape

        for f in range(n_feat):
            last = np.nan
            for t in range(T):
                if not np.isnan(seq2d[t, f]):
                    last = seq2d[t, f]
                elif not np.isnan(last):
                    seq2d[t, f] = last
        return _from_2d(seq2d, sequence.ndim)


class BackwardFillImputer(BaseImputer):
    """
    Imputa propagando hacia atrás el **próximo valor válido**.
    """
    def __init__(self, name: str = "Backward Fill"):
        self.name = name

    def fit(self, sequences: List[np.ndarray]) -> Any:
        return self

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        seq2d = _to_2d(sequence).copy()
        T, n_feat = seq2d.shape

        for f in range(n_feat):
            nxt = np.nan
            for t in range(T - 1, -1, -1):
                if not np.isnan(seq2d[t, f]):
                    nxt = seq2d[t, f]
                elif not np.isnan(nxt):
                    seq2d[t, f] = nxt
        return _from_2d(seq2d, sequence.ndim)
