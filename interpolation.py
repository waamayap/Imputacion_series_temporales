"""
Interpolación lineal y spline para series uni- o multivariables.
Cada columna (variable) se interpola de forma independiente para rellenar NaNs.
"""

from __future__ import annotations
import numpy as np
from scipy import interpolate           # SciPy se usa para la interpolación spline/lineal
from approach import BaseImputer        # Interfaz que deben cumplir todos los imputadores


# ─────────────────────────── HELPERS ────────────────────────────
def _to_2d(arr: np.ndarray) -> np.ndarray:
    """
    Asegura que la entrada sea 2-D.
    • Si es 1-D (T,), la convierte a (T,1) sin copiar datos.
    • Si ya es 2-D (T, n_features), la devuelve tal cual.
    """
    return arr[:, np.newaxis] if arr.ndim == 1 else arr


def _from_2d(arr2d: np.ndarray, original_ndim: int) -> np.ndarray:
    """
    Devuelve la salida con la forma de la entrada original:
    • Si la entrada era 1-D, aplana de (T,1) a (T,)
    • Si era 2-D, mantiene la forma (T, n_features)
    """
    return arr2d.ravel() if original_ndim == 1 else arr2d


# ────────────────── 1. INTERPOLACIÓN LINEAL ────────────────────
class LinearInterpolationImputer(BaseImputer):
    """Imputador que rellena NaNs mediante interpolación lineal simple."""

    def __init__(self, name: str = "Linear Interpolation"):
        self.name = name            # nombre que aparecerá en reportes

    # Para interpolación lineal no se necesita entrenamiento
    def fit(self, sequences):
        return self

    def predict(self, sequence):
        """
        Rellena valores faltantes por variable usando interpolación lineal.
        Cada variable (columna) se procesa de forma independiente.
        """
        seq2d = _to_2d(sequence).copy()      # Garantizar forma (T, n_feat)
        T, n_feat = seq2d.shape

        # Recorrer cada columna (variable)
        for f in range(n_feat):
            col = seq2d[:, f]
            valid = ~np.isnan(col)           # Máscara de valores no NaN
            if np.sum(valid) == 0:
                continue  # ⚡ Si toda la columna es NaN, no se puede interpolar

            # Índices y valores válidos
            x_valid = np.where(valid)[0]
            y_valid = col[valid]

            # Función de interpolación lineal; extrapola fuera del rango
            f_lin = interpolate.interp1d(
                x_valid, y_valid, bounds_error=False, fill_value="extrapolate"
            )

            # Valores interpolados para todos los tiempos
            col_interp = f_lin(np.arange(T))

            # Sustituir solo en posiciones NaN
            col[np.isnan(col)] = col_interp[np.isnan(col)]
            seq2d[:, f] = col

        return _from_2d(seq2d, sequence.ndim)


# ──────────────── 2. INTERPOLACIÓN SPLINE ───────────────────────
class SplineInterpolationImputer(BaseImputer):
    """Imputador que rellena NaNs mediante spline cúbico (o grado k)."""

    def __init__(self, k: int = 3, name: str = "Spline Interpolation"):
        """
        Parámetros:
        • k: grado del spline (k=3 = cúbico, k=1 = lineal…)
        """
        self.k = k
        self.name = name

    def fit(self, sequences):
        return self  # No hay entrenamiento; la interpolación es directa

    # ----------- método auxiliar: interpola UNA columna ----------------
    def _interp_col(self, col: np.ndarray):
        valid = ~np.isnan(col)
        if np.sum(valid) == 0:
            return col  # ⚡ Si no hay datos válidos, se devuelve tal cual

        # Si hay muy pocos puntos para spline de grado k, caer a lineal
        if np.sum(valid) < self.k + 1:
            x_valid = np.where(valid)[0]
            y_valid = col[valid]
            f_lin = interpolate.interp1d(
                x_valid, y_valid, bounds_error=False, fill_value="extrapolate"
            )
            return f_lin(np.arange(len(col)))

        # Caso normal: spline de grado k (intenta; si falla, usa k=1)
        x_valid = np.where(valid)[0]
        y_valid = col[valid]
        try:
            tck = interpolate.splrep(x_valid, y_valid, k=self.k)
        except Exception:
            tck = interpolate.splrep(x_valid, y_valid, k=1)
        return interpolate.splev(np.arange(len(col)), tck)

    # ------------------------------------------------------------------
    def predict(self, sequence):
        """
        Rellena valores faltantes mediante spline para cada variable.
        Aplica fallback lineal si el spline no es viable.
        """
        seq2d = _to_2d(sequence).copy()

        for f in range(seq2d.shape[1]):
            col = seq2d[:, f]
            interp_col = self._interp_col(col)                 # vector interpolado
            col[np.isnan(col)] = interp_col[np.isnan(col)]     # sustituir NaNs
            seq2d[:, f] = col

        return _from_2d(seq2d, sequence.ndim)

