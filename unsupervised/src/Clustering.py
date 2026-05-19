"""
Clustering.py
=============
Implementación de algoritmos de clustering para la etapa no supervisada.
Responsable: Equipo no supervisado

Incluye:
- K-Means con búsqueda de k óptimo
- DBSCAN con búsqueda de parámetros
- Métricas de validación interna
- Visualizaciones de clusters en 2D
- Exportación de resultados
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "eda_figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    if n_clusters <= 1:
        return {
            "n_clusters": n_clusters,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    mask = labels != -1
    try:
        sil = silhouette_score(X[mask], labels[mask]) if mask.sum() > 1 else np.nan
    except Exception:
        sil = np.nan

    try:
        ch = calinski_harabasz_score(X[mask], labels[mask]) if mask.sum() > 1 else np.nan
    except Exception:
        ch = np.nan

    try:
        db = davies_bouldin_score(X[mask], labels[mask]) if mask.sum() > 1 else np.nan
    except Exception:
        db = np.nan

    return {
        "n_clusters": n_clusters,
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
    }


def run_kmeans_grid(X: pd.DataFrame, k_range: range = range(2, 8)) -> tuple:
    results = []
    print("\n[INFO] Buscando k óptimo para K-Means...")
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        metrics = clustering_metrics(X.values, labels)
        results.append({
            "algorithm": "KMeans",
            "k": k,
            "n_clusters": metrics["n_clusters"],
            "silhouette": metrics["silhouette"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "davies_bouldin": metrics["davies_bouldin"],
        })
        print(f"   k={k} | sil={metrics['silhouette']:.4f} | ch={metrics['calinski_harabasz']:.1f} | db={metrics['davies_bouldin']:.4f}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["silhouette"], ascending=False)
    best_row = df_results.iloc[0]

    best_model = KMeans(n_clusters=int(best_row["k"]), random_state=42, n_init=20)
    best_model.fit(X)

    return df_results, best_model


def run_dbscan_grid(
    X: pd.DataFrame,
    eps_values: list = None,
    min_samples_values: list = None,
) -> tuple:
    if eps_values is None:
        eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    if min_samples_values is None:
        min_samples_values = [3, 5, 7]

    results = []
    print("\n[INFO] Buscando parámetros para DBSCAN...")
    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            metrics = clustering_metrics(X.values, labels)
            results.append({
                "algorithm": "DBSCAN",
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": metrics["n_clusters"],
                "silhouette": metrics["silhouette"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "davies_bouldin": metrics["davies_bouldin"],
            })
            print(f"   eps={eps:.2f} min_samples={min_samples} | n_clusters={metrics['n_clusters']} | sil={metrics['silhouette']:.4f}")

    df_results = pd.DataFrame(results)
    df_results = df_results[df_results["n_clusters"] > 1].copy()
    if df_results.empty:
        print("   [WARNING] Ninguna configuración DBSCAN válida con más de 1 cluster.")
        return pd.DataFrame(results), None

    df_results = df_results.sort_values(by=["silhouette"], ascending=False)
    best = df_results.iloc[0]
    best_model = DBSCAN(eps=float(best["eps"]), min_samples=int(best["min_samples"]))
    best_model.fit(X)

    return df_results, best_model


def run_agglomerative(X: pd.DataFrame, n_clusters: int = 3) -> tuple:
    print(f"\n[INFO] Aplicando Agglomerative Clustering con n_clusters={n_clusters}...")
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    metrics = clustering_metrics(X.values, labels)
    print(f"   n_clusters={metrics['n_clusters']} | sil={metrics['silhouette']:.4f} | ch={metrics['calinski_harabasz']:.1f}")
    return model, metrics


def plot_pca_clusters(
    X: pd.DataFrame,
    labels: np.ndarray,
    method_name: str,
    title: str = None,
    guardar: bool = True,
):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_plot = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": labels.astype(str),
    })

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("husl", n_colors=df_plot["cluster"].nunique())
    sns.scatterplot(
        data=df_plot,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette=palette,
        s=90,
        alpha=0.8,
        ax=ax,
        edgecolor="k",
        linewidth=0.5,
    )

    ax.set_title(title or f"Clusters {method_name} sobre PCA 2D")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if guardar:
        path = FIGURES_DIR / f"unsupervised_{method_name.lower()}_pca.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"   [OK] Guardado: {path}")
    plt.close()


def export_cluster_assignments(
    df: pd.DataFrame,
    labels: np.ndarray,
    method_name: str,
    file_name: str = "cluster_assignments",
) -> Path:
    df_export = df.copy()
    df_export[f"cluster_{method_name}"] = labels
    path = REPORTS_DIR / f"{file_name}_{method_name}.csv"
    df_export.to_csv(path, index=False, encoding="utf-8")
    print(f"   [OK] Asignaciones guardadas: {path}")
    return path


def validate_with_labels(
    labels: np.ndarray,
    y_true: pd.Series,
    method_name: str,
    n_clusters: int,
) -> dict:
    metrics = {
        "algorithm": method_name,
        "n_clusters": n_clusters,
        "adjusted_rand_index": np.nan,
    }
    if y_true is None or len(y_true) != len(labels):
        return metrics

    mask = y_true.notna() & (labels != -1)
    if mask.sum() < 2:
        return metrics

    try:
        ari = adjusted_rand_score(y_true[mask], labels[mask])
    except Exception:
        ari = np.nan

    metrics["adjusted_rand_index"] = ari
    print(f"   [VALIDACIÓN] {method_name} | ARI = {ari:.4f}")
    return metrics
