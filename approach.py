"""Abstract base classes that define the common interface for the whole
pipeline (data processing, imputers and benchmarking).  Do **not** add any
execution logic here – only interfaces that concrete classes must honour.
"""
from __future__ import annotations
import ssl
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Sequence, Any, Optional
import numpy as np

# ---------------------------------------------------------------------------
#                            DATA INTERFACE
# ---------------------------------------------------------------------------
class TimeSeriesData(ABC):
    """Contract that every concrete time‑series data loader/handler must obey."""

    @abstractmethod
    def load_data(self, filepath: str) -> Any:  # usually a pandas.DataFrame
        """Read the raw CSV and return an *unmodified* DataFrame."""

    @abstractmethod
    def identify_stations_and_variables(
        self, df, datetime_col: str | None = None
    ) -> Tuple[List[str], List[str], Dict[str, Dict[str, str]]]:
        """Return (stations, variables, mapping[station][variable] -> column‑name)."""

    # --- OPTIONAL helpers that concrete classes may provide ----------------
    @abstractmethod
    def filter_data_by_date(self, df, start_date: str, end_date: str):
        pass

    @abstractmethod
    def set_index(self, df, index_column: str):
        pass

    @abstractmethod
    def get_complete_sequences(self, time_interval: int) -> np.ndarray:
        pass

    @abstractmethod
    def split_sequences(
        self, sequences: np.ndarray, train_ratio: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

# ---------------------------------------------------------------------------
#                              IMPUTERS
# ---------------------------------------------------------------------------
class BaseImputer(ABC):

    name: str  # a short identifier to be used in reports

    @abstractmethod
    def fit(self, sequences: Sequence[np.ndarray]):
        pass

    @abstractmethod
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """Return *one* sequence with NaNs replaced."""

# ---------------------------------------------------------------------------
#                            BENCHMARK DRIVER
# ---------------------------------------------------------------------------
class BenchmarkApproach(ABC):
   
    def __init__(self, data_tensor: np.ndarray, imputers: List[BaseImputer]):
        self.data = data_tensor
        self.imputers = imputers

    @abstractmethod
    def create_gaps(self, sequence: np.ndarray, gap_percentage: float):
        pass

    @abstractmethod
    def evaluate(
        self,
        time_interval: int = 24,
        gap_percentage: float = 0.1,
        train_ratio: float = 0.7,
    ) -> Dict[str, Dict[str, float]]:
        pass


class HybridImputer(BaseImputer):


    @abstractmethod
    def fit(self, sequences: Sequence[np.ndarray]) -> None:
     
        pass

    @abstractmethod
    def predict(self, sequence: np.ndarray) -> np.ndarray:
    
        pass

    @abstractmethod
    def set_cnn_imputer(self, knn_imputer: BaseImputer) -> None:
      
        pass

    @abstractmethod
    def set_max_linear_gap(self, max_lin_gap: int) -> None:
       
        pass






