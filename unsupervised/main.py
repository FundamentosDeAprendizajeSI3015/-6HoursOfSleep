"""
main.py
=======
Runner principal del pipeline no supervisado.
Responsable: Equipo no supervisado

Flujo:
0. Definir objetivo de apoyo a la predicción supervisada de deserción
1. Cargar datos y prepararlos para clustering
2. Escalar features numéricas
3. Ejecutar K-Means y seleccionar k óptimo
4. Ejecutar DBSCAN con búsqueda de parámetros
5. Ejecutar clustering jerárquico (Agglomerative)
6. Guardar resultados y visualizaciones
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_loader import load_and_prepare
from Clustering import (
    scale_features,
    run_kmeans_grid,
    run_dbscan_grid,
    run_agglomerative,
    plot_pca_clusters,
    export_cluster_assignments,
    validate_with_labels,
)


def main():
    print("\n" + "="*70)
    print("🎯 PIPELINE NO SUPERVISADO - CLUSTERING DE DESERCIÓN")
    print("Objetivo: explorar patrones en los indicadores que puedan apoyar la predicción de la tasa de deserción estudiantil.")
    print("="*70)

    data_path = Path(r"c:\Users\Usuario\Desktop\EAFIT\Semestre 6\Fundamentos de Aprendizaje Automático\-6HoursOfSleep\data_simulada\processed\data_simulado_1980_2026.csv")
    X, y, loader = load_and_prepare(str(data_path))
    df_base = loader.df_clean.copy()

    X_scaled = scale_features(X)

    print("\n[1/5] K-MEANS")
    kmeans_results, kmeans_model = run_kmeans_grid(X_scaled)
    kmeans_labels = kmeans_model.predict(X_scaled)
    plot_pca_clusters(
        X_scaled,
        kmeans_labels,
        method_name="KMeans",
        title="K-Means Clustering (PCA 2D)",
        guardar=True,
    )
    export_cluster_assignments(df_base, kmeans_labels, "KMeans")
    kmeans_counts = pd.Series(kmeans_labels).value_counts().sort_index()
    print(f"   K-Means counts: {kmeans_counts.to_dict()}")

    print("\n[2/5] DBSCAN")
    dbscan_results, dbscan_model = run_dbscan_grid(X_scaled)
    if dbscan_model is not None:
        dbscan_labels = dbscan_model.labels_
        plot_pca_clusters(
            X_scaled,
            dbscan_labels,
            method_name="DBSCAN",
            title="DBSCAN Clustering (PCA 2D)",
            guardar=True,
        )
        export_cluster_assignments(df_base, dbscan_labels, "DBSCAN")
        dbscan_counts = pd.Series(dbscan_labels).value_counts().sort_index()
        noise_pct = 100 * dbscan_counts.get(-1, 0) / len(dbscan_labels)
        print(f"   DBSCAN counts: {dbscan_counts.to_dict()}")
        print(f"   Ruido: {dbscan_counts.get(-1, 0)} observaciones ({noise_pct:.1f}% del dataset)")
    else:
        dbscan_labels = None
        print("   [WARNING] No se guardaron resultados DBSCAN debido a falta de configuración válida.")

    print("\n[3/5] Agglomerative Clustering")
    n_clusters = int(kmeans_model.n_clusters) if kmeans_model is not None else 3
    agg_model, agg_metrics = run_agglomerative(X_scaled, n_clusters=n_clusters)
    agg_labels = agg_model.fit_predict(X_scaled)
    plot_pca_clusters(
        X_scaled,
        agg_labels,
        method_name="Agglomerative",
        title="Agglomerative Clustering (PCA 2D)",
        guardar=True,
    )
    export_cluster_assignments(df_base, agg_labels, "Agglomerative")
    agg_counts = pd.Series(agg_labels).value_counts().sort_index()
    print(f"   Agglomerative counts: {agg_counts.to_dict()}")

    print("\n[4/5] Validación de etiquetas (si está disponible)")
    y_cat = None
    if y is not None:
        try:
            y_cat = pd.qcut(y, q=4, labels=False, duplicates="drop")
        except Exception:
            y_cat = y.copy()

    validation_rows = []
    validation_rows.append(validate_with_labels(kmeans_labels, y_cat, "KMeans", len(np.unique(kmeans_labels[kmeans_labels != -1]))))
    if dbscan_labels is not None:
        validation_rows.append(validate_with_labels(dbscan_labels, y_cat, "DBSCAN", len(np.unique(dbscan_labels[dbscan_labels != -1]))))
    validation_rows.append(validate_with_labels(agg_labels, y_cat, "Agglomerative", len(np.unique(agg_labels[agg_labels != -1]))))

    df_validation = pd.DataFrame(validation_rows)
    validation_path = Path(__file__).parent / "reports" / "unsupervised_validation_summary.csv"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    df_validation.to_csv(validation_path, index=False, encoding="utf-8")
    print(f"\n[OK] Validación guardada: {validation_path}")

    print("\n[5/5] Resumen final")
    print("\nK-Means mejores parámetros:")
    print(kmeans_results.head(3).to_string(index=False))
    print("\nDBSCAN mejores parámetros:")
    print(dbscan_results.head(3).to_string(index=False))
    print("\nAgglomerative metrics:")
    print(agg_metrics)

    print("\n" + "="*70)
    print("✅ PIPELINE NO SUPERVISADO COMPLETADO")
    print("="*70)
    print("\nArchivos generados:")
    print("   • Gráficos: unsupervised/reports/eda_figures/unsupervised_*.png")
    print("   • Asignaciones: unsupervised/reports/cluster_assignments_*.csv")
    print("   • Validación: unsupervised/reports/unsupervised_validation_summary.csv")
    print("\n[6/5] Guardando resultados en JSON para el dashboard")
    docs_path = Path(__file__).parent / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)
    
    import json
    
    def replace_nan(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {k: replace_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_nan(i) for i in obj]
        return obj

    dbscan_list = dbscan_results.to_dict(orient="records") if dbscan_results is not None else []
    
    results_json = {
        "kmeans_results": kmeans_results.to_dict(orient="records"),
        "dbscan_results": dbscan_list,
        "agg_metrics": agg_metrics,
        "validation": df_validation.to_dict(orient="records"),
    }
    
    with open(docs_path / "unsupervised_results.json", "w", encoding="utf-8") as f:
        json.dump(replace_nan(results_json), f, indent=4, ensure_ascii=False)
    print(f"   [OK] JSON guardado en: {docs_path / 'unsupervised_results.json'}")

    print("\n" + "="*70 + "\n")

    return {
        "X": X,
        "y": y,
        "kmeans_results": kmeans_results,
        "dbscan_results": dbscan_results,
        "validation": df_validation,
    }


if __name__ == "__main__":
    main()
