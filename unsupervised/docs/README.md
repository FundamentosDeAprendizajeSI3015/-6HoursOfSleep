# Análisis No Supervisado — Clustering

Este módulo agrega una etapa de análisis no supervisado al proyecto.

## Objetivo

- Explorar patrones de departamentos y años sin usar la variable objetivo.
- Implementar algoritmos de clustering como K-Means y DBSCAN.
- Generar visualizaciones y métricas de validación interna.

## Estructura

- `main.py` → Orquesta el pipeline no supervisado.
- `src/data_loader.py` → Carga y limpia los datos para clustering.
- `src/Clustering.py` → Implementa K-Means, DBSCAN, Agglomerative y evaluación.
- `reports/` → Guarda resultados y figuras.

## Uso

Ejecución desde la raíz del proyecto:

```bash
python unsupervised/main.py
```

## Salidas

- `unsupervised/reports/eda_figures/unsupervised_*.png`
- `unsupervised/reports/cluster_assignments_*.csv`
- `unsupervised/reports/unsupervised_validation_summary.csv`
