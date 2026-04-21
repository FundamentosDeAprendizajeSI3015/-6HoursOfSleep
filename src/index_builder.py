"""
index_builder.py
================
Construcción del índice de vulnerabilidad económica-educativa.
Responsable: Isabela | Rama: feature/unsupervised-index

El índice es una combinación lineal de variables económicas que resume
el nivel de riesgo de deserción estudiantil para cada departamento-año.

Dos enfoques:
- PCA: pesos automáticos desde la primera componente principal
- Teórico: pesos definidos por literatura económica
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


# ── Variables base del índice ─────────────────────────────────────────────────

VARIABLES_INDICE = [
    "desempleo",
    "pobreza",
    "inflacion",
    "tasa_interes",
    "ingreso_real",          # negativo: más ingreso = menos vulnerabilidad
    "ratio_desempleo_pobreza"
]

# Pesos teóricos (basados en revisión de literatura)
# Positivo: aumenta vulnerabilidad | Negativo: disminuye vulnerabilidad
PESOS_TEORICOS = {
    "desempleo":               0.30,
    "pobreza":                 0.25,
    "inflacion":               0.15,
    "tasa_interes":            0.15,
    "ingreso_real":           -0.15,
    "ratio_desempleo_pobreza": 0.00   # complementario, peso a ajustar
}


# ── Índice por PCA ────────────────────────────────────────────────────────────

def construir_indice_pca(df: pd.DataFrame,
                          variables: list = None) -> pd.DataFrame:
    """
    Construye el índice usando las cargas de la primera componente principal.

    El signo se corrige para que valores altos = más vulnerabilidad
    (se verifica con la correlación con la variable desercion).
    """
    if variables is None:
        variables = [v for v in VARIABLES_INDICE if v in df.columns]

    df = df.copy()
    X = df[variables].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    indice_raw = pca.fit_transform(X_scaled).flatten()

    # Corregir signo: índice debe correlacionar positivamente con deserción
    if "desercion" in df.columns:
        corr = np.corrcoef(indice_raw, df.loc[X.index, "desercion"])[0, 1]
        if corr < 0:
            indice_raw = -indice_raw

    df.loc[X.index, "indice_vulnerabilidad_pca"] = indice_raw

    # Información de cargas
    cargas = pd.Series(pca.components_[0], index=variables, name="carga_pca")
    varianza = pca.explained_variance_ratio_[0]
    print(f"✅ Índice PCA construido | Varianza explicada: {varianza:.2%}")
    print(cargas.sort_values(ascending=False))

    return df, cargas, varianza


# ── Índice teórico ────────────────────────────────────────────────────────────

def construir_indice_teorico(df: pd.DataFrame,
                              pesos: dict = None) -> pd.DataFrame:
    """
    Construye el índice como suma ponderada con pesos definidos por teoría.

    Los pesos deben sumar 1 (en valor absoluto) para interpretabilidad.
    """
    if pesos is None:
        pesos = PESOS_TEORICOS

    df = df.copy()
    variables_disponibles = [v for v in pesos.keys() if v in df.columns]

    # Normalizar variables antes de ponderar
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df[variables_disponibles]),
        columns=variables_disponibles,
        index=df.index
    )

    indice = sum(X_scaled[v] * pesos[v] for v in variables_disponibles)
    df["indice_vulnerabilidad_teorico"] = indice

    print(f"✅ Índice teórico construido | Variables usadas: {variables_disponibles}")
    print(f"   Pesos: {pesos}")

    if "desercion" in df.columns:
        from scipy.stats import spearmanr
        rho, p = spearmanr(df["indice_vulnerabilidad_teorico"].dropna(),
                           df["desercion"].dropna())
        print(f"   Correlación Spearman con deserción: ρ={rho:.3f} (p={p:.4f})")

    return df


# ── Clustering ────────────────────────────────────────────────────────────────

def clustering_kmeans(df: pd.DataFrame,
                       variables: list = None,
                       k_range: range = range(2, 8)) -> pd.DataFrame:
    """
    Aplica K-Means con selección de k óptimo por silhouette score.
    """
    if variables is None:
        variables = [v for v in VARIABLES_INDICE if v in df.columns]

    df = df.copy()
    X = df[variables].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    resultados = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        resultados.append({"k": k, "silhouette": sil, "inertia": km.inertia_})
        print(f"  k={k}: silhouette={sil:.3f}, inercia={km.inertia_:.1f}")

    df_resultados = pd.DataFrame(resultados)
    k_optimo = df_resultados.loc[df_resultados["silhouette"].idxmax(), "k"]
    print(f"\n✅ k óptimo seleccionado: {k_optimo}")

    # Aplicar con k óptimo
    km_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    df.loc[X.index, "cluster_km"] = km_final.fit_predict(X_scaled)
    df["cluster_km"] = df["cluster_km"].astype("Int64")

    return df, df_resultados, int(k_optimo)


# ── Validación de etiquetas ───────────────────────────────────────────────────

def validar_etiquetas(df: pd.DataFrame,
                       col_cluster: str = "cluster_km",
                       col_desercion: str = "desercion",
                       n_cuartiles: int = 4) -> dict:
    """
    Evalúa si la variable desercion es una buena etiqueta usando
    el Adjusted Rand Index entre clusters y cuartiles de deserción.

    ARI > 0.3  → etiqueta informativa
    ARI > 0.5  → etiqueta muy informativa
    ARI < 0.1  → poca correspondencia
    """
    df = df.dropna(subset=[col_cluster, col_desercion]).copy()

    desercion_cat = pd.qcut(df[col_desercion], q=n_cuartiles, labels=False,
                             duplicates="drop")
    ari = adjusted_rand_score(desercion_cat, df[col_cluster])

    print(f"📊 Adjusted Rand Index (clusters vs. cuartiles de deserción): {ari:.3f}")
    if ari > 0.5:
        print("   → ✅ Etiqueta muy informativa")
    elif ari > 0.3:
        print("   → ✅ Etiqueta informativa")
    elif ari > 0.1:
        print("   → ⚠️  Etiqueta moderadamente informativa")
    else:
        print("   → ❌ Poca correspondencia entre clusters y deserción")

    return {"ari": ari, "n_cuartiles": n_cuartiles}


# ── Uso de ejemplo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/processed/dataset_final.csv")

    # Construir índices
    df, cargas, varianza = construir_indice_pca(df)
    df = construir_indice_teorico(df)

    # Clustering
    df, resultados_km, k_opt = clustering_kmeans(df)

    # Validar etiquetas
    metricas = validar_etiquetas(df)

    # Guardar
    df.to_csv("data/processed/dataset_con_indice.csv", index=False)
    print("\n✅ Dataset con índice guardado en data/processed/dataset_con_indice.csv")
