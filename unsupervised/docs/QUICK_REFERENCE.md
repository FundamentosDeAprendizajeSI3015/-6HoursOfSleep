# Quick Reference — No Supervisado

## Ejecución

```bash
python unsupervised/main.py
```

## Qué hace

- `unsupervised/src/data_loader.py` carga y limpia los datos.
- `unsupervised/src/Clustering.py` ejecuta K-Means, DBSCAN y Agglomerative.
- Genera métricas internas de clustering y visualizaciones PCA 2D.

## Salidas

- `unsupervised/reports/eda_figures/unsupervised_*.png`
- `unsupervised/reports/cluster_assignments_*.csv`
- `unsupervised/reports/unsupervised_validation_summary.csv`
