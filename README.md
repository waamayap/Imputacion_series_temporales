# Imputación de Series Temporales

Este proyecto implementa y compara distintos métodos de **imputación de datos faltantes** en series temporales, aplicado principalmente a datos de polución del aire.  
Se evalúan enfoques estadísticos, de interpolación, machine learning y deep learning (LSTM y CNN).

---

## 📂 Estructura del Proyecto

- `approach.py` → Define interfaces abstractas para datos, imputadores y benchmarks.
- `timeseriesdata.py` → Carga CSV, mapea estaciones/variables y construye el tensor 3D (estaciones × variables × tiempo).
- `interpolation.py` → Métodos de imputación por interpolación (lineal y spline).
- `machine_learning.py` → Imputadores basados en KNN, LSTM y CNN.
- `statistical.py` → Imputadores simples: media, mediana, forward-fill, backward-fill.
- `hybrid_imputer.py` → Imputador híbrido que combina interpolación lineal (para huecos pequeños) y KNN (para huecos largos).
- `benchmark.py` → Implementa `ConcreteBenchmark`, que inserta huecos artificiales, entrena imputadores y genera métricas y gráficos.
- `main.py` → Script principal para ejecutar el benchmark completo desde línea de comandos.

---

## ⚙️ Instalación

Clona el repositorio y entra al proyecto:



```bash
git clone https://github.com/waamayap/Imputacion_series_temporales.git
cd Imputacion_series_temporales
