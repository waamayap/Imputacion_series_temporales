from __future__ import annotations

import numpy as np
from sklearn.impute import KNNImputer as SklearnKNNImputer
from approach import BaseImputer      # interfaz común para todos los imputadores

# --- TensorFlow / Keras (modelos profundos) -----------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Flatten,
    Dropout,
    LayerNormalization,
    MaxPooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Escalador global usado en LSTM/CNN
from timeseriesdata import MinMaxScaler


def _ensure_list_of_arrays(sequences):
    if isinstance(sequences, np.ndarray):
        if sequences.ndim == 3:
            return [sequences[i] for i in range(sequences.shape[0])]
        if sequences.ndim == 2:
            return [sequences]
        raise ValueError(f"Unsupported array ndim={sequences.ndim}")
    if isinstance(sequences, list):
        return sequences
    raise TypeError("sequences must be ndarray or list of ndarray")


class KNNImputer(BaseImputer):
    """
    Imputación multivariante usando K-Nearest Neighbors.  
    Incorpora todas las variables más la posición temporal normalizada.
    """
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", name: str = "KNN Imputer"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = name

    def fit(self, sequences: np.ndarray | list[np.ndarray]) -> KNNImputer:
        # No requiere entrenamiento previo
        return self

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        # Asegurar forma 2D: (T, n_features)
        arr = sequence.copy()
        single = False
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
            single = True

        T, n_feat = arr.shape
        # Índice temporal normalizado
        idx = np.arange(T).reshape(-1, 1) / float(T - 1)
        # Espacio de características: [tiempo, todas las variables]
        features = np.hstack([idx, arr])

        # KNN imputación multivariante
        knn = SklearnKNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
        imputed = knn.fit_transform(features)

        # Extraer solo las columnas de variables (descartar índice temporal)
        out = imputed[:, 1:]
        # Restaurar formato original
        return out.ravel() if single else out


class LSTMImputer(BaseImputer):
    def __init__(
        self,
        window_size: int = 24,
        epochs: int = 10,
        batch_size: int = 480,
        verbose: int = 1,
        name: str = "LSTM Imputer",
    ):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.name = name
        self.model: Sequential | None = None
        self.scaler = MinMaxScaler()
        self.n_features: int | None = None
        self.history: dict = {}

    def _build_model(self) -> Sequential:
        model = Sequential([
            LayerNormalization(input_shape=(self.window_size, self.n_features)),
            LSTM(
                64,
                return_sequences=True,
                kernel_regularizer=l2(1e-4),
                recurrent_regularizer=l2(1e-4),
                recurrent_dropout=0.2
            ),
            Dropout(0.4),
            LSTM(
                32,
                kernel_regularizer=l2(1e-4),
                recurrent_regularizer=l2(1e-4),
                recurrent_dropout=0.2
            ),
            Dropout(0.4),
            Dense(self.n_features, activation='relu', kernel_regularizer=l2(1e-4)),
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mse'
        )
        return model

    def fit(self, sequences: np.ndarray | list[np.ndarray]) -> LSTMImputer:
        seqs = _ensure_list_of_arrays(sequences)
        self.n_features = seqs[0].shape[1]
        self.scaler.fit(seqs)
        scaled = [self.scaler.transform(s) for s in seqs]

        X, y = [], []
        for s in scaled:
            for i in range(len(s) - self.window_size):
                win = s[i : i + self.window_size]
                tgt = s[i + self.window_size]
                if not np.isnan(win).any() and not np.isnan(tgt).any():
                    X.append(win)
                    y.append(tgt)
        X, y = np.array(X), np.array(y)

        if X.size == 0:
            print("[WARN] LSTMImputer no entrenado: datos insuficientes")
            return self

        self.model = self._build_model()
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        history = self.model.fit(
            X, y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[early_stop]
        )
        self.history = history.history
        return self

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        arr = sequence.copy()
        single = False
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
            single = True

        if self.model is None:
            return sequence

        nan_mask = np.isnan(arr)
        seq = arr.copy()
        nan_rows = np.where(nan_mask.any(axis=1))[0]
        for idx in nan_rows:
            start = idx - self.window_size
            if start < 0:
                continue
            window = seq[start:idx]
            if window.shape[0] != self.window_size or np.isnan(window).any():
                continue
            scaled_window = self.scaler.transform(window)
            x = scaled_window.reshape(1, self.window_size, self.n_features)
            y_scaled = self.model.predict(x, verbose=0)[0]
            # Clipping scaled prediction to [0,1]
            y_scaled = np.clip(y_scaled, 0, 1)
            y_pred = self.scaler.inverse_transform(y_scaled.reshape(1, -1))[0]
            # Avoid negative values
            y_pred = np.maximum(0, y_pred)
            cols_to_fill = nan_mask[idx]
            seq[idx, cols_to_fill] = y_pred[cols_to_fill]

        return seq.ravel() if single else seq


class CNNImputer(BaseImputer):
    def __init__(self, window_size: int = 24, epochs: int = 10, verbose: int = 1, name: str = "CNN Imputer"):
        self.window_size = window_size
        self.epochs = epochs
        self.verbose = verbose
        self.name = name
        self.model: Sequential | None = None
        self.scaler = MinMaxScaler()
        self.n_features = None
        self.history: Dict[str, list] = {}

    def _build_model(self) -> Sequential:
        model = Sequential([
            LayerNormalization(input_shape=(self.window_size, self.n_features)),
            Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            Flatten(),
            Dense(self.n_features, activation='relu', kernel_regularizer=l2(1e-4)),
        ])
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
        return model

    def fit(self, sequences: np.ndarray | list[np.ndarray]) -> CNNImputer:
        seqs = _ensure_list_of_arrays(sequences)
        self.n_features = seqs[0].shape[1]
        self.scaler.fit(seqs)
        scaled = [self.scaler.transform(s) for s in seqs]
        X, y = [], []
        for s in scaled:
            for i in range(len(s) - self.window_size):
                win = s[i:i+self.window_size]
                tgt = s[i+self.window_size]
                if not np.isnan(win).any() and not np.isnan(tgt).any():
                    X.append(win)
                    y.append(tgt)
        X, y = np.array(X), np.array(y)
        if X.size == 0:
            print('[WARN] CNNImputer no entrenado: datos insuficientes')
            return self
        self.model = self._build_model()
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(
            X, y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=480,
            verbose=self.verbose,
            callbacks=[early_stop]
        )
        self.history = history.history
        return self

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        arr = sequence.copy()
        single = False
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
            single = True

        if self.model is None:
            return sequence

        nan_mask = np.isnan(arr)
        seq = arr.copy()
        nan_rows = np.where(nan_mask.any(axis=1))[0]
        for idx in nan_rows:
            start = idx - self.window_size
            if start < 0:
                continue
            window = seq[start:idx]
            if window.shape[0] != self.window_size or np.isnan(window).any():
                continue
            scaled_window = self.scaler.transform(window)
            x = scaled_window.reshape(1, self.window_size, self.n_features)
            y_scaled = self.model.predict(x, verbose=0)[0]
            # Clip scaled prediction
            y_scaled = np.clip(y_scaled, 0, 1)
            y_pred = self.scaler.inverse_transform(y_scaled.reshape(1, -1))[0]
            # Avoid negative
            y_pred = np.maximum(0, y_pred)
            cols_to_fill = nan_mask[idx]
            seq[idx, cols_to_fill] = y_pred[cols_to_fill]

        return seq.ravel() if single else seq
