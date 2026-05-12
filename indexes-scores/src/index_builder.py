# ================
# Construcción del índice de vulnerabilidad económica-educativa.

# El índice es una combinación lineal de variables económicas que resume
# el nivel de riesgo de deserción estudiantil para cada departamento-año.

# Flujo:
# 1. Carga del dataset y limpieza según EDA
# 2. Construcción de PCA y clustering
# 3. Validación de etiquetas
# 4. Índice por PCA y teórico
# 5. Exportación a data/final

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


# ── Rutas ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_FINAL = BASE_DIR / "data" / "final"
REPORTS_DIR = BASE_DIR / "reports" / "ind_figures"

# Crear carpeta final si no existe
DATA_FINAL.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuración visual ────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


# ── Limpieza del dataset según conclusiones del EDA ────────────────────────────

def limpiar_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Limpia el dataset según hallazgos del EDA:
    
    1. Elimina data leakage (spadies_td_anual_universitario = outcome)
    2. Excluye columnas técnicas (outcome_merge_pendiente)
    3. Maneja variables con alto % de nulos (>50%)
    4. Selecciona variables útiles para análisis
    
    No elimina variables por nombre, sino por condiciones.
    """
    df = df.copy()
    n_orig = len(df)
    
    if verbose:
        print("\n" + "="*70)
        print("  LIMPIEZA DEL DATASET SEGÚN EDA")
        print("="*70)
    
    # 1. Detectar y eliminar data leakage
    if "spadies_td_anual_universitario" in df.columns and "outcome_tasa_desercion_snies" in df.columns:
        mask_both = df[["spadies_td_anual_universitario", "outcome_tasa_desercion_snies"]].notna().all(axis=1)
        if mask_both.any():
            match = (df.loc[mask_both, "spadies_td_anual_universitario"].round(8) ==
                    df.loc[mask_both, "outcome_tasa_desercion_snies"].round(8)).sum()
            if match > 0:
                if verbose:
                    print(f"\nDATA LEAKAGE DETECTADO:")
                    print(f"   spadies_td_anual_universitario = outcome_tasa_desercion_snies")
                    print(f"   Eliminando: spadies_td_anual_universitario")
                df = df.drop(columns=["spadies_td_anual_universitario"])
    
    # 2. Eliminar columnas técnicas (outcome_merge_pendiente)
    cols_to_drop = [c for c in df.columns if "merge_pendiente" in c.lower()]
    if cols_to_drop:
        if verbose:
            print(f"\n📋 Columnas técnicas eliminadas: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # 3. Identificar variables con muy altos nulos (>55%, según EDA)
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_null_threshold = 55
    high_null_cols = null_pct[null_pct > high_null_threshold].index.tolist()
    
    if high_null_cols:
        if verbose:
            print(f"\nColumnas con >{high_null_threshold}% nulos (excluidas del análisis):")
            for col in high_null_cols:
                print(f"   {col}: {null_pct[col]:.1f}%")
    
    # 4. Variables macroeconómicas nacionales: solo 2 valores (uno por año)
    # Estas tienen baja utilidad discriminativa entre departamentos
    macro_cols = [c for c in df.columns if "ipc_nacional" in c.lower() or 
                  ("geih" in c.lower() and "nacional" in c.lower())]
    if macro_cols and verbose:
        print(f"\n📊 Variables macroeconómicas nacionales (baja discriminación):")
        print(f"   {len(macro_cols)} variables identificadas (pueden excluirse en modelado)")
    
    if verbose:
        print(f"\nDataset limpio: {len(df)} filas × {df.shape[1]} columnas")
        print(f"   Variables totales: {df.shape[1]}")
        print(f"   Variables numéricas: {df.select_dtypes(include=np.number).shape[1]}")
    
    return df


# ── Variables base para índice (mapeadas al dataset) ──────────────────────────

def identificar_variables_indice(df: pd.DataFrame) -> list:
    """
    Identifica las variables disponibles en el dataset para construir el índice.
    Mapea a: desempleo, inflación, ingreso relativo, variables de educación.
    """
    variables = []
    
    # Desempleo: geih_td (tasa de desempleo nacional)
    if "geih_td_nacional_media_anual" in df.columns:
        variables.append("geih_td_nacional_media_anual")
    
    # Inflación: IPC (tomar variación media)
    if "ipc_nacional_total_var_mensual_media" in df.columns:
        variables.append("ipc_nacional_total_var_mensual_media")
    
    # Ingreso relativo (proxy): PIB por matriculado
    if "proxy_pib_miles_mm_cop_por_matriculado" in df.columns:
        variables.append("proxy_pib_miles_mm_cop_por_matriculado")
    
    # Matriculación: total de matriculados
    if "total_matriculados" in df.columns:
        variables.append("total_matriculados")
    
    # SPADIES tasas (excepto la de leakage): tecnológico
    if "spadies_td_anual_tecnologico" in df.columns:
        variables.append("spadies_td_anual_tecnologico")
    
    # PIB (crecimiento económico)
    if "pib_variacion_pct_anual_vs_anio_previo" in df.columns:
        variables.append("pib_variacion_pct_anual_vs_anio_previo")
    
    # Tasa de ocupación (GEIH)
    if "geih_to_nacional_media_anual" in df.columns:
        variables.append("geih_to_nacional_media_anual")
    
    return variables


# ── Pesos teóricos (literatura económica) ─────────────────────────────────────

PESOS_TEORICOS = {
    "geih_td_nacional_media_anual": 0.25,           # Desempleo: aumenta vulnerabilidad
    "ipc_nacional_total_var_mensual_media": 0.15,   # Inflación: aumenta vulnerabilidad
    "proxy_pib_miles_mm_cop_por_matriculado": -0.20,  # Ingreso: disminuye vulnerabilidad
    "total_matriculados": -0.10,                    # Acceso a educación: disminuye vuln
    "spadies_td_anual_tecnologico": 0.15,           # Deserción técnica: correlación
    "pib_variacion_pct_anual_vs_anio_previo": -0.10,  # Crecimiento: disminuye vuln
    "geih_to_nacional_media_anual": -0.15,          # Ocupación: disminuye vulnerabilidad
}

# ── Análisis PCA (reducción de dimensiones) ──────────────────────────────────

def analizar_pca(df: pd.DataFrame,
                 variables: list = None,
                 n_components: int = 2) -> tuple:
    """
    Realiza PCA para reducción de dimensiones y visualización.
    Retorna: pca, X_pca, varianza explicada.
    """
    if variables is None:
        return None, None, None
    
    df_clean = df[variables].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    varianza = pca.explained_variance_ratio_
    varianza_acum = np.cumsum(varianza)
    
    print(f"\n✅ PCA realizado:")
    print(f"   Varianza explicada: {varianza}")
    print(f"   Varianza acumulada: {varianza_acum}")
    print(f"   PC1: {varianza[0]:.1%} | PC2: {varianza[1]:.1%} | Total: {varianza_acum[1]:.1%}")
    
    # Retornar también los índices para mapeo
    return pca, X_pca, varianza_acum, df_clean.index, scaler


# ── Índice por PCA ────────────────────────────────────────────────────────────

def construir_indice_pca(df: pd.DataFrame,
                          variables: list = None) -> tuple:
    """
    Construye el índice usando las cargas de la primera componente principal.
    El signo se corrige para correlacionar positivamente con deserción.
    """
    if variables is None or len(variables) == 0:
        print("⚠️  No hay variables para PCA")
        return df, None, None
    
    df = df.copy()
    df_clean = df[variables].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    pca = PCA(n_components=1)
    indice_raw = pca.fit_transform(X_scaled).flatten()
    
    # Corregir signo: correlacionar positivamente con outcome
    if "outcome_tasa_desercion_snies" in df.columns:
        mask_outcome = df.loc[df_clean.index, "outcome_tasa_desercion_snies"].notna()
        if mask_outcome.any():
            corr = np.corrcoef(
                indice_raw[mask_outcome.values],
                df.loc[df_clean.index[mask_outcome.values], "outcome_tasa_desercion_snies"].values
            )[0, 1]
            if np.isnan(corr) or corr < 0:
                indice_raw = -indice_raw
                print("   (Signo invertido para correlación positiva con outcome)")
    
    df.loc[df_clean.index, "indice_vulnerabilidad_pca"] = indice_raw
    
    cargas = pd.Series(pca.components_[0], index=variables, name="carga_pca")
    varianza = pca.explained_variance_ratio_[0]
    
    print(f"\n✅ Índice PCA construido | Varianza explicada: {varianza:.1%}")
    print("   Cargas (importancia por variable):")
    for feat, carga in cargas.sort_values(ascending=False).items():
        print(f"      {feat:45s}: {carga:7.4f}")
    
    return df, cargas, varianza


# ── Índice teórico ────────────────────────────────────────────────────────────

def construir_indice_teorico(df: pd.DataFrame,
                              pesos: dict = None) -> tuple:
    """
    Construye el índice como suma ponderada con pesos definidos por teoría.
    """
    if pesos is None:
        pesos = PESOS_TEORICOS
    
    df = df.copy()
    variables_disponibles = [v for v in pesos.keys() if v in df.columns]
    
    if len(variables_disponibles) == 0:
        print("⚠️  No hay variables para índice teórico")
        return df, None
    
    df_clean = df[variables_disponibles].dropna()
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(df_clean),
        columns=variables_disponibles,
        index=df_clean.index
    )
    
    # Calcular índice ponderado
    indice = pd.Series(0.0, index=df.index)
    for var in variables_disponibles:
        indice.loc[df_clean.index] += X_scaled[var] * pesos[var]
    
    df["indice_vulnerabilidad_teorico"] = indice
    
    print(f"\n✅ Índice teórico construido | {len(variables_disponibles)} variables")
    print("   Variables y pesos:")
    for var in variables_disponibles:
        print(f"      {var:45s}: {pesos[var]:7.3f}")
    
    # Correlación con outcome
    if "outcome_tasa_desercion_snies" in df.columns:
        mask = df[["indice_vulnerabilidad_teorico", "outcome_tasa_desercion_snies"]].notna().all(axis=1)
        if mask.any():
            rho, p = spearmanr(
                df.loc[mask, "indice_vulnerabilidad_teorico"],
                df.loc[mask, "outcome_tasa_desercion_snies"]
            )
            print(f"   Correlación Spearman con outcome: ρ={rho:.3f} (p={p:.4f})")
    
    return df, pesos


# ── Clustering: K-Means ─────────────────────────────────────────────────────

def clustering_kmeans(df: pd.DataFrame,
                      variables: list = None,
                      k_range: range = range(2, 8)) -> tuple:
    """
    Aplica K-Means con selección de k óptimo por silhouette score.
    """
    if variables is None or len(variables) == 0:
        print("⚠️  No hay variables para clustering")
        return df, None, None
    
    df = df.copy()
    df_clean = df[variables].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    resultados = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        resultados.append({
            "k": k,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "inertia": km.inertia_
        })
    
    df_resultados = pd.DataFrame(resultados)
    k_optimo = int(df_resultados.loc[df_resultados["silhouette"].idxmax(), "k"])
    
    print(f"\n✅ K-Means - Selección de k:")
    print(df_resultados.to_string(index=False))
    print(f"\n   k óptimo: {k_optimo} (máximo silhouette score)")
    
    # Aplicar con k óptimo
    km_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_scaled)
    df.loc[df_clean.index, "cluster_km"] = labels
    
    return df, df_resultados, k_optimo


# ── Clustering: Jerárquico ─────────────────────────────────────────────────

def clustering_jerarquico(df: pd.DataFrame,
                          variables: list = None,
                          n_clusters: int = 3) -> tuple:
    """
    Aplica clustering jerárquico aglomerativo.
    """
    if variables is None or len(variables) == 0:
        print("⚠️  No hay variables para clustering jerárquico")
        return df, None
    
    df = df.copy()
    df_clean = df[variables].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hc.fit_predict(X_scaled)
    
    df.loc[df_clean.index, "cluster_hierarchical"] = labels
    
    print(f"\n✅ Clustering jerárquico aplicado: {n_clusters} clusters")
    
    return df, hc


# ── Validación de etiquetas (ARI) ────────────────────────────────────────────

def validar_etiquetas(df: pd.DataFrame,
                      col_cluster: str = "cluster_km",
                      col_outcome: str = "outcome_tasa_desercion_snies",
                      n_cuartiles: int = 4) -> dict:
    """
    Evalúa si el outcome es una buena etiqueta mediante Adjusted Rand Index.
    ARI > 0.3 = informativa, ARI > 0.5 = muy informativa
    """
    df_clean = df[[col_cluster, col_outcome]].dropna()
    
    if len(df_clean) < n_cuartiles:
        print(f"⚠️  Insuficientes observaciones ({len(df_clean)}) para {n_cuartiles} cuartiles")
        return {"ari": np.nan, "n_cuartiles": n_cuartiles}
    
    # Discretizar outcome en cuartiles
    outcome_cat = pd.qcut(df_clean[col_outcome], q=n_cuartiles, 
                         labels=False, duplicates="drop")
    
    # Calcular ARI
    ari = adjusted_rand_score(outcome_cat, df_clean[col_cluster])
    
    print(f"\n✅ Validación de etiquetas (Adjusted Rand Index):")
    print(f"   ARI = {ari:.3f}")
    if ari > 0.5:
        print("   → ✅ Etiqueta MUY INFORMATIVA (clusters coinciden bien con outcome)")
    elif ari > 0.3:
        print("   → ✅ Etiqueta INFORMATIVA")
    elif ari > 0.1:
        print("   → ⚠️  Etiqueta moderadamente informativa")
    else:
        print("   → ❌ Poca correspondencia entre clusters y outcome")
    
    return {"ari": ari, "n_cuartiles": n_cuartiles}


# ── Visualizaciones ────────────────────────────────────────────────────────────

def generar_visualizaciones(df: pd.DataFrame,
                           pca_result: tuple = None,
                           variables: list = None):
    """
    Genera visualizaciones del análisis.
    """
    if not variables or len(variables) < 2:
        return
    
    print(f"\n📊 Generando visualizaciones...")
    
    # 1. Scree plot (varianza explicada por componentes)
    if pca_result is not None and pca_result[0] is not None:
        pca, X_pca, _, idx_pca, scaler = pca_result
        
        fig, ax = plt.subplots(figsize=(10, 6))
        varianza_exp = pca.explained_variance_ratio_
        ax.bar(range(1, len(varianza_exp) + 1), varianza_exp, alpha=0.8, color='steelblue')
        ax.plot(range(1, len(varianza_exp) + 1), np.cumsum(varianza_exp), 
               'ro-', linewidth=2, markersize=8, label='Acumulada')
        ax.set_xlabel('Componente Principal')
        ax.set_ylabel('Varianza Explicada')
        ax.set_title('Scree Plot - Varianza Explicada por PCA')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / '01_scree_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   [Guardado] 01_scree_plot.png")
        
        # 1b. Biplot PC1 vs PC2
        if X_pca.shape[1] >= 2 and variables and len(variables) > 0:
            fig, ax = plt.subplots(figsize=(12, 9))
            
            # Plotear observaciones coloreadas por outcome
            colors = df.loc[idx_pca, "outcome_tasa_desercion_snies"].fillna(0)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='RdYlGn_r',
                               s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Plotear flechas de cargas (loadings) con etiquetas sin sobreposición
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            n_vars = min(len(variables), 7)  # Limitar a 7 variables por claridad
            
            # Calcular ángulos para distribuir etiquetas uniformemente
            for i, var in enumerate(variables[:n_vars]):
                arrow_len = 3.0
                # Flecha
                ax.arrow(0, 0, loadings[i, 0]*arrow_len, loadings[i, 1]*arrow_len,
                        head_width=0.15, head_length=0.15, fc='red', ec='red', alpha=0.7, zorder=3)
                
                # Posicionar etiqueta a mayor distancia para evitar sobreposición
                label_distance = 4.2
                ax.text(loadings[i, 0]*label_distance, loadings[i, 1]*label_distance, 
                       var[:20],
                       fontsize=8, ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                                alpha=0.75, edgecolor='red', linewidth=1),
                       zorder=4)
                
                # Línea de conexión desde punta de flecha a etiqueta (opcional, para claridad)
                ax.plot([loadings[i, 0]*arrow_len, loadings[i, 0]*label_distance],
                       [loadings[i, 1]*arrow_len, loadings[i, 1]*label_distance],
                       'r--', alpha=0.4, linewidth=0.8, zorder=2)
            
            ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, weight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, weight='bold')
            ax.set_title('Biplot PCA - Observaciones y Cargas de Variables', fontsize=13, weight='bold', pad=20)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Tasa de Deserción', rotation=270, labelpad=20)
            
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / '01b_biplot_pca.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   [Guardado] 01b_biplot_pca.png")
    
    
    # 2. Índice PCA vs Outcome
    if "indice_vulnerabilidad_pca" in df.columns and "outcome_tasa_desercion_snies" in df.columns:
        df_plot = df[["indice_vulnerabilidad_pca", "outcome_tasa_desercion_snies", 
                     "departamento"]].dropna()
        
        if len(df_plot) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df_plot["indice_vulnerabilidad_pca"],
                               df_plot["outcome_tasa_desercion_snies"] * 100,
                               s=100, alpha=0.6, edgecolors='black', linewidth=1)
            for idx, row in df_plot.iterrows():
                ax.annotate(row["departamento"][:3], 
                           (row["indice_vulnerabilidad_pca"], row["outcome_tasa_desercion_snies"]*100),
                           fontsize=7, alpha=0.7)
            
            ax.set_xlabel("Índice de Vulnerabilidad (PCA)")
            ax.set_ylabel("Tasa de Deserción (%)")
            ax.set_title("Índice PCA vs. Outcome (Tasa de Deserción)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / '02_indice_pca_vs_outcome.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   [Guardado] 02_indice_pca_vs_outcome.png")
    
    # 3. Índice teórico vs Outcome
    if "indice_vulnerabilidad_teorico" in df.columns and "outcome_tasa_desercion_snies" in df.columns:
        df_plot = df[["indice_vulnerabilidad_teorico", "outcome_tasa_desercion_snies",
                     "departamento"]].dropna()
        
        if len(df_plot) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_plot["indice_vulnerabilidad_teorico"],
                      df_plot["outcome_tasa_desercion_snies"] * 100,
                      s=100, alpha=0.6, color='green', edgecolors='black', linewidth=1)
            for idx, row in df_plot.iterrows():
                ax.annotate(row["departamento"][:3],
                           (row["indice_vulnerabilidad_teorico"], row["outcome_tasa_desercion_snies"]*100),
                           fontsize=7, alpha=0.7)
            
            ax.set_xlabel("Índice de Vulnerabilidad (Teórico)")
            ax.set_ylabel("Tasa de Deserción (%)")
            ax.set_title("Índice Teórico vs. Outcome (Tasa de Deserción)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / '03_indice_teorico_vs_outcome.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   [Guardado] 03_indice_teorico_vs_outcome.png")
    
    # 3b. Comparación de índices (PCA vs Teórico)
    if ("indice_vulnerabilidad_pca" in df.columns and 
        "indice_vulnerabilidad_teorico" in df.columns):
        df_plot = df[["indice_vulnerabilidad_pca", "indice_vulnerabilidad_teorico",
                     "departamento"]].dropna()
        
        if len(df_plot) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_plot["indice_vulnerabilidad_pca"],
                      df_plot["indice_vulnerabilidad_teorico"],
                      s=100, alpha=0.6, color='purple', edgecolors='black', linewidth=1)
            for idx, row in df_plot.iterrows():
                ax.annotate(row["departamento"][:3],
                           (row["indice_vulnerabilidad_pca"], row["indice_vulnerabilidad_teorico"]),
                           fontsize=7, alpha=0.7)
            
            ax.set_xlabel("Índice PCA")
            ax.set_ylabel("Índice Teórico")
            ax.set_title("Comparación: Índice PCA vs. Índice Teórico")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / '03b_comparacion_indices.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   [Guardado] 03b_comparacion_indices.png")
    
    # 4. Clusters vs Outcome
    if "cluster_km" in df.columns and "outcome_tasa_desercion_snies" in df.columns:
        df_plot = df[["cluster_km", "outcome_tasa_desercion_snies",
                     "departamento"]].dropna()
        
        if len(df_plot) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_cluster = plt.cm.Set3(df_plot["cluster_km"])
            ax.scatter(df_plot["cluster_km"], df_plot["outcome_tasa_desercion_snies"] * 100,
                      s=150, c=df_plot["cluster_km"], cmap='Set3', 
                      edgecolors='black', linewidth=1, alpha=0.7)
            
            for idx, row in df_plot.iterrows():
                ax.annotate(row["departamento"][:3],
                           (row["cluster_km"], row["outcome_tasa_desercion_snies"]*100),
                           fontsize=7, alpha=0.7)
            
            ax.set_xlabel("Cluster K-Means")
            ax.set_ylabel("Tasa de Deserción (%)")
            ax.set_title("Clusters K-Means vs. Outcome")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / '04_clusters_vs_outcome.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   [Guardado] 04_clusters_vs_outcome.png")


# ── Función principal ─────────────────────────────────────────────────────────

def pipeline_completo():
    """
    Ejecuta el pipeline completo:
    1. Carga y limpieza
    2. Análisis PCA
    3. Construcción de índices
    4. Clustering
    5. Validación
    6. Exportación
    """
    print("="*70)
    print("  PIPELINE: ANÁLISIS NO SUPERVISADO E ÍNDICE DE VULNERABILIDAD")
    print("="*70)
    
    # 1. CARGA Y LIMPIEZA
    print("\n📥 Cargando dataset...")
    df_path = DATA_PROCESSED / "panel_desercion_socioeconomico_completo 1.csv"
    df = pd.read_csv(df_path)
    print(f"   Dataset original: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    df = limpiar_dataset(df, verbose=True)
    
    # 2. IDENTIFICAR VARIABLES PARA ÍNDICE
    print("\n🔍 Identificando variables disponibles...")
    variables_indice = identificar_variables_indice(df)
    print(f"   Variables seleccionadas ({len(variables_indice)}):")
    for v in variables_indice:
        print(f"      - {v}")
    
    # 3. PCA Y COMPONENTES PRINCIPALES
    print("\n📐 Análisis PCA...")
    pca_result = analizar_pca(df, variables_indice, n_components=2)
    
    # 4. CONSTRUCCIÓN DE ÍNDICES
    print("\n🔨 Construcción de índices...")
    df, cargas_pca, varianza_pca = construir_indice_pca(df, variables_indice)
    df, pesos_teoricos = construir_indice_teorico(df, PESOS_TEORICOS)
    
    # 5. CLUSTERING (K-Means)
    print("\n🎯 Clustering K-Means...")
    df, df_km_resultados, k_opt = clustering_kmeans(df, variables_indice, k_range=range(2, 8))
    
    # 6. CLUSTERING (Jerárquico)
    print("\n🌳 Clustering Jerárquico...")
    df, hc = clustering_jerarquico(df, variables_indice, n_clusters=k_opt)
    
    # 7. VALIDACIÓN DE ETIQUETAS
    print("\n✔️  Validación de etiquetas...")
    metricas_km = validar_etiquetas(df, "cluster_km", "outcome_tasa_desercion_snies", n_cuartiles=4)
    metricas_hc = validar_etiquetas(df, "cluster_hierarchical", "outcome_tasa_desercion_snies", n_cuartiles=4)
    
    # 8. VISUALIZACIONES
    generar_visualizaciones(df, pca_result, variables_indice)
    
    # 9. AGREGAR PC1 Y PC2 AL DATASET
    if pca_result[0] is not None:
        pca, X_pca, _, idx_pca, scaler = pca_result
        df_pca_full = df.loc[idx_pca].copy()
        df.loc[idx_pca, "pc1"] = X_pca[:, 0]
        if X_pca.shape[1] > 1:
            df.loc[idx_pca, "pc2"] = X_pca[:, 1]
    
    # 10. EXPORTACIÓN
    print("\n📤 Exportando dataset limpio...")
    output_path = DATA_FINAL / "dataset_con_indice.csv"
    df.to_csv(output_path, index=False)
    print(f"   ✅ {output_path}")
    print(f"   Dimensiones finales: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    # Resumen
    print("\n" + "="*70)
    print("  RESUMEN DEL ANÁLISIS")
    print("="*70)
    print(f"\n✅ Variables en índice: {len(variables_indice)}")
    print(f"✅ Varianza explicada (PC1): {varianza_pca:.1%}")
    print(f"✅ Clusters K-Means óptimo: {k_opt}")
    print(f"✅ ARI validación (K-Means): {metricas_km['ari']:.3f}")
    print(f"✅ ARI validación (Jerárquico): {metricas_hc['ari']:.3f}")
    print(f"✅ Dataset exportado: data/final/dataset_con_indice.csv")
    print("\n" + "="*70)
    
    return df


# ── Punto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_final = pipeline_completo()
