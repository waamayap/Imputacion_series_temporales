from __future__ import annotations

import os
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from approach import BenchmarkApproach, BaseImputer
from timeseriesdata import ConcreteTimeSeriesData

log = logging.getLogger(__name__)

class ConcreteBenchmark(BenchmarkApproach):
    def __init__(
        self,
        ts: ConcreteTimeSeriesData,
        data_tensor: np.ndarray,
        imputers: List[BaseImputer],
        time_interval: int,
        estaciones: List[str],
        variables: List[str],
        seleccion: Dict[str, List[str]],
        seed: Optional[int] = None,
    ):
        super().__init__(data_tensor, imputers)
        self.ts = ts
        self.data_tensor = data_tensor
        self.time_interval = time_interval
        self.estaciones = estaciones
        self.variables = variables
        self.seleccion = seleccion

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.output_dir = os.path.join(os.path.dirname(__file__), "output_graphs")
        os.makedirs(self.output_dir, exist_ok=True)
        log.info("Directorio de salida creado: %s", self.output_dir)

    def _rand_start(self, T: int, L: int) -> Optional[int]:
        max_start = T - L - 1
        return None if max_start <= 1 else int(self.rng.integers(1, max_start + 1))

    def _build_mat_global(self) -> Tuple[np.ndarray, List[str]]:
        idxs, cols = [], []
        for i, est in enumerate(self.estaciones):
            if est not in self.seleccion:
                continue
            for var in self.seleccion[est]:
                v_idx = self.variables.index(var)
                idxs.append((i, v_idx))
                cols.append(f"{est},{var}")
        mat = np.stack([self.data_tensor[e, v, :] for e, v in idxs], axis=1)
        return mat, cols

    def _inject_by_column(
        self,
        mat: np.ndarray,
        pct: float,
        class_probs: Dict[str, float],
    ) -> Tuple[np.ndarray, List[List[Tuple[str, int, int]]]]:
        T, V = mat.shape
        total = max(1, int(T * pct))
        duration = {
            "class_1": (1, 6),
            "class_2": (7, 24),
            "class_3": (25, 168),
            "class_4": (169, min(330, T // 2)),
        }
        valid = [c for c, (mn, mx) in duration.items() if mx >= mn]
        probs = None
        if class_probs:
            arr = np.array([class_probs.get(c, 0.0) for c in valid], dtype=float)
            s = arr.sum()
            if s > 0:
                probs = arr / s

        out = mat.copy()
        gaps_info: List[List[Tuple[str, int, int]]] = []
        for v in range(V):
            col_gaps: List[Tuple[str, int, int]] = []
            inserted = 0
            for cls in valid:
                mn, mx = duration[cls]
                L = int(self.rng.integers(mn, mx + 1))
                start = self._rand_start(T, L)
                if start is not None:
                    out[start:start + L, v] = np.nan
                    col_gaps.append((cls, start, L))
                    inserted += L
            while inserted < total:
                cls = (
                    self.rng.choice(valid, p=probs) if probs is not None
                    else self.rng.choice(valid)
                )
                mn, mx = duration[cls]
                L = int(self.rng.integers(mn, mx + 1))
                start = self._rand_start(T, L)
                if start is None or any(abs(start - s) < ln for (_c, s, ln) in col_gaps):
                    continue
                out[start:start + L, v] = np.nan
                col_gaps.append((cls, start, L))
                inserted += L
            gaps_info.append(col_gaps)
        return out, gaps_info

    def _plot_lines(
        self,
        seq0: np.ndarray,
        seq_nan0: np.ndarray,
        gaps_info: List[List[Tuple[str, int, int]]],
        cols: List[str],
        metric_name: str,
        metric_func: Callable[[np.ndarray, np.ndarray], float],
        filename_prefix: str,
    ) -> None:
        preds_full = {imp.name: imp.predict(seq_nan0) for imp in self.imputers}
        for j, col_name in enumerate(cols):
            orig = seq0[:, j]
            for cls, start, length in gaps_info[j]:
                end = start + length
                fig, ax = plt.subplots(figsize=(10, 4))

                ax.plot(
                    np.arange(start, end), orig[start:end],
                    label='Original', linestyle='-', linewidth=1,
                    alpha=0.7, zorder=1, color='grey'
                )

                for imp in self.imputers:
                    pred_seg = preds_full[imp.name][start:end, j]
                    score = metric_func(orig[start:end], pred_seg)
                    ax.plot(
                        np.arange(start, end), pred_seg,
                        label=f"{imp.name} ({metric_name}={score:.2f})",
                        linestyle='--', linewidth=1, alpha=0.6, zorder=2
                    )

                est, var = map(str.strip, col_name.split(','))
                ax.set_title(f"{est} – {var} – {cls} – Comparación ({metric_name})")
                ax.set_xlabel('Paso de tiempo')
                ax.set_ylabel('Valor')
                ax.legend(title='Imputador', bbox_to_anchor=(1.02, 1), loc='upper left')

                plt.tight_layout(rect=(0, 0, 0.85, 1))
                fname = f"{filename_prefix}_{est}_{var}_{cls}.pdf"
                plt.savefig(os.path.join(self.output_dir, fname))
                plt.close()
                log.info("✅ %s exportado: %s", metric_name, fname)

    def _plot_rmse_multi_class(
        self,
        seq0: np.ndarray,
        seq_nan0: np.ndarray,
        gaps_info: List[List[Tuple[str, int, int]]],
        cols: List[str],
    ) -> None:
        preds_full = {imp.name: imp.predict(seq_nan0) for imp in self.imputers}

        for j, col_name in enumerate(cols):
            est, var = map(str.strip, col_name.split(','))
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            global_handles: List[Any] = []
            global_labels: List[str] = []

            for idx, (cls, start, length) in enumerate(gaps_info[j]):
                ax = axes[idx]
                end = start + length
                orig = seq0[start:end, j]

                ax.plot(
                    np.arange(start, end),
                    orig,
                    label='Original',
                    color='grey',
                    alpha=0.7
                )

                for imp in self.imputers:
                    pred_seg = preds_full[imp.name][start:end, j]
                    rmse_score = np.sqrt(np.mean((orig - pred_seg)**2))
                    line, = ax.plot(
                        np.arange(start, end),
                        pred_seg,
                        linestyle='--',
                        label=f"{imp.name} (RMSE={rmse_score:.2f})"
                    )
                    if idx == 0:
                        global_handles.append(line)
                        global_labels.append(imp.name)

                ax.set_title(cls)
                ax.set_xlabel('Paso')
                ax.set_ylabel('Valor')

                handles_all, labels_all = ax.get_legend_handles_labels()
                rmse_items = [
                    (h, lbl[lbl.find('('):]) 
                    for h, lbl in zip(handles_all, labels_all) 
                    if 'RMSE=' in lbl
                ]
                if rmse_items:
                    rmse_handles, rmse_labels = zip(*rmse_items)
                    ax.legend(
                        rmse_handles,
                        rmse_labels,
                        title='',
                        fontsize=7,
                        bbox_to_anchor=(1.02, 1),
                        loc='upper left',
                        borderaxespad=0
                    )

            fig.suptitle(f"{est} – {var} – Comparación RMSE por clase", y=0.92)

            fig.legend(
                global_handles,
                global_labels,
                title='Imputador',
                loc='upper center',
                ncol=len(global_labels),
                bbox_to_anchor=(0.5, 1.00)
            )

            plt.tight_layout(rect=(0, 0, 1, 0.88))

            fname = f"lineplot_rmse_{est}_{var}_multiclass.pdf"
            plt.savefig(os.path.join(self.output_dir, fname))
            plt.close()
            log.info("✅ RMSE multiclas exportado: %s", fname)

    def _boxplot_by_class(self, df: pd.DataFrame, metric_col: str, title: str, filename: str, ylim: Optional[Tuple[float, float]] = None) -> None:
        classes = sorted(df['clase'].unique())
        methods = sorted(df['imputador'].unique())
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        width = 0.8 / len(methods)
        base = np.arange(len(classes))

        fig, ax = plt.subplots(figsize=(8, 5))
        for k, m in enumerate(methods):
            data = [
                df[(df['imputador'] == m) & (df['clase'] == cls)][metric_col].values
                for cls in classes
            ]
            positions = base - 0.4 + width/2 + k*width
            ax.boxplot(
                data, positions=positions, widths=width,
                boxprops=dict(color=colors[k]), whiskerprops=dict(color=colors[k]),
                capprops=dict(color=colors[k]), medianprops=dict(color=colors[k])
            )
        ax.set_title(title)
        ax.set_xticks(base + 0.4)
        ax.set_xticklabels(classes)
        ax.set_xlabel('Clase tamaño de hueco')
        ax.set_ylabel(metric_col)
        if ylim:
            ax.set_ylim(*ylim)
        for k, m in enumerate(methods):
            ax.plot([], [], color=colors[k], label=m)
        ax.legend(title='Imputador', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        log.info("✅ Boxplot exportado: %s", filename)

    def _boxplots_by_variable(self, df: pd.DataFrame, metric_col: str, prefix: str, per_page: int = 12) -> None:
        variables = sorted(df['variable'].unique())
        methods = sorted(df['imputador'].unique())
        classes = sorted(df['clase'].unique())
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        width = 0.8 / len(methods)
        base = np.arange(len(classes))

        pages = [variables[i:i+per_page] for i in range(0, len(variables), per_page)]
        for idx, vars_page in enumerate(pages, 1):
            rows, cols_grid = 4, 3
            fig, axes = plt.subplots(rows, cols_grid, figsize=(12, 8))
            axes = axes.flatten()

            for ax, var in zip(axes, vars_page):
                for k, m in enumerate(methods):
                    data = [
                        df[(df['imputador']==m) & (df['variable']==var) & (df['clase']==cls)][metric_col].values
                        for cls in classes
                    ]
                    pos = base - 0.4 + width/2 + k*width
                    ax.boxplot(
                        data, positions=pos, widths=width,
                        boxprops=dict(color=colors[k]), whiskerprops=dict(color=colors[k]),
                        capprops=dict(color=colors[k]), medianprops=dict(color=colors[k])
                    )
                ax.set_title(var, fontsize=8)
                ax.set_xticks(base)
                ax.set_xticklabels(classes, fontsize=6)
                ax.set_ylabel(metric_col, fontsize=6)
                ax.set_ylim(0, None)

            for ax in axes[len(vars_page):]:
                fig.delaxes(ax)

            handles = [plt.Line2D([], [], color=colors[i], label=m) for i, m in enumerate(methods)]
            fig.legend(handles=handles, title='Imputador', bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=6)
            plt.tight_layout(rect=(0, 0, 0.85, 1))
            fname = f"{prefix}_por_variable_p{idx}.pdf"
            plt.savefig(os.path.join(self.output_dir, fname))
            plt.close()
            log.info("✅ Boxplots por variable exportados: %s", fname)

    def evaluate(self, gap_percentage: float, train_ratio: float = 0.8, show_bar: bool = True, class_probs: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        mat_global, cols = self._build_mat_global()
        T = mat_global.shape[0]
        L = self.time_interval
        split_idx = int(T * train_ratio)

        train_seqs = []
        for i in range(0, split_idx - L + 1, L):
            window = mat_global[i : i + L, :]
            if not np.isnan(window).any():
                train_seqs.append(window)
        train = np.array(train_seqs)

        val_seqs = []
        for i in range(split_idx, T - L + 1, L):
            window = mat_global[i : i + L, :]
            if not np.isnan(window).any():
                val_seqs.append(window)
        val = np.array(val_seqs)

        if train.size == 0 or val.size == 0:
            log.warning("No hay suficientes ventanas limpias para train/val.")
            return pd.DataFrame()

        for imp in self.imputers:
            imp.fit(train)
            if hasattr(imp, 'history'):
                hist = imp.history
                plt.figure(figsize=(6,4))
                plt.plot(hist.get('loss', []), label='loss')
                if 'val_loss' in hist:
                    plt.plot(hist['val_loss'], label='val_loss')
                plt.title(f'Entrenamiento {imp.name}')
                plt.xlabel('Épocas')
                plt.ylabel('MSE')
                plt.legend(loc='best')
                plt.tight_layout()
                png_name = f"history_{imp.name.replace(' ', '_')}.png"
                plt.savefig(os.path.join(self.output_dir, png_name))
                plt.close()
                df_hist = pd.DataFrame(hist)
                csv_name = f"history_{imp.name.replace(' ', '_')}.csv"
                df_hist.to_csv(os.path.join(self.output_dir, csv_name), index=False)

        scaler = MinMaxScaler()
        scaler.fit(mat_global.reshape(-1, 1))

        records: List[Dict[str, Any]] = []
        for i, seq0 in enumerate(val):
            seq_nan0, gaps_info = self._inject_by_column(seq0, gap_percentage, class_probs or {})
            for imp in self.imputers:
                pred_full = imp.predict(seq_nan0)
                for v_idx, col_gaps in enumerate(gaps_info):
                    for cls, start, length in col_gaps:
                        true_vals = seq0[start:start+length, v_idx]
                        pred_vals = pred_full[start:start+length, v_idx]
                        true_norm = scaler.transform(true_vals.reshape(-1,1)).flatten()
                        pred_norm = scaler.transform(pred_vals.reshape(-1,1)).flatten()
                        rmse_norm = np.sqrt(np.mean((true_norm - pred_norm) ** 2))
                        records.append({
                            'ventana': i+1,
                            'imputador': imp.name,
                            'clase': cls,
                            'variable': cols[v_idx],
                            'inicio': start,
                            'porcentaje': length,
                            'rmse_norm': rmse_norm,
                        })

        df_long = pd.DataFrame(records)
        self._boxplot_by_class(df_long, 'rmse_norm', 'Boxplot de RMSE por clase de hueco', 'rmse_norm_gap_boxplot.pdf', (0, 1))
        self._boxplots_by_variable(df_long, 'rmse_norm', 'rmse')
        return df_long

    def create_gaps(self, mat: np.ndarray, pct: float, class_probs: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, float, str]]]:
        out, gaps = self._inject_by_column(mat, pct, class_probs or {})
        mask = np.isnan(out)
        return out, mask, gaps
