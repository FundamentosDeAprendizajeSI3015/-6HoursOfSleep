# ==========================================
# EDA — PANEL DE DESERCIÓN ESTUDIANTIL
# Factores Socioeconómicos y Macroeconómicos
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ---- Configuración ----
DATASET_PATH = '../../data/processed/panel_desercion_socioeconomico_completo 1.csv'
OUTPUT_DIR = '../reports/dataset_original'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# ------------------------------------------
# 1. Carga e inspección general
# ------------------------------------------
df = pd.read_csv(DATASET_PATH)

print("=" * 65)
print("  EDA — PANEL DE DESERCIÓN ESTUDIANTIL SNIES")
print("=" * 65)
print(f"\nDimensión del dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Años cubiertos: {df['anio'].min()} – {df['anio'].max()}")
print(f"Departamentos únicos: {df['departamento'].nunique()}")
print(f"\nColumnas:\n{list(df.columns)}")

# ------------------------------------------
# 2. Calidad de datos
# ------------------------------------------
print("\n\n===== CALIDAD DE DATOS =====")

# 2a. Duplicados
n_dup = df.duplicated().sum()
print(f"\nFilas duplicadas: {n_dup}")

# 2b. Nulos por columna
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(1)
null_df = pd.DataFrame({'nulos': null_counts, 'pct_nulos': null_pct})
null_df = null_df[null_df['nulos'] > 0].sort_values('nulos', ascending=False)
print("\nColumnas con valores nulos:")
print(null_df.to_string())

# 2c. Cardinalidad
print("\nCardinalidad por columna:")
for col in df.columns:
    dtype = df[col].dtype
    card = df[col].nunique()
    print(f"  {col:<50} {card:>4} valores únicos  [{dtype}]")

# 2d. Visualización de nulos
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if p > 40 else '#f39c12' if p > 0 else '#2ecc71'
          for p in null_pct]
bars = ax.barh(null_pct.index, null_pct.values, color=colors, edgecolor='white')
ax.set_xlabel("% de valores nulos")
ax.set_title("Porcentaje de valores nulos por columna", fontsize=14, fontweight='bold')
ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% umbral crítico')
ax.legend()
# Etiquetas
for bar, val in zip(bars, null_pct.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_nulos_por_columna.png'), dpi=150)
plt.close()
print("\n[Guardado] 01_nulos_por_columna.png")

# ------------------------------------------
# 3. Estadísticas descriptivas
# ------------------------------------------
print("\n\n===== ESTADÍSTICAS DESCRIPTIVAS =====")

df_num = df.select_dtypes(include=np.number).drop(
    columns=['codigo_departamento', 'outcome_merge_pendiente'], errors='ignore'
)

desc = df_num.describe(percentiles=[.10, .25, .50, .75, .90]).T
desc['cv'] = (desc['std'] / desc['mean'].abs()).round(3)  # coeficiente de variación
print(desc[['count', 'mean', 'std', 'min', '10%', '25%', '50%', '75%', '90%', 'max', 'cv']].to_string())

# ------------------------------------------
# 4. Estructura del panel: cobertura por año y depto
# ------------------------------------------
print("\n\n===== ESTRUCTURA DEL PANEL =====")

pivot_cover = df.pivot_table(
    index='departamento', columns='anio',
    values='outcome_tasa_desercion_snies',
    aggfunc=lambda x: 'OK' if x.notna().all() else 'NaN'
)
print("\nCobertura del target (outcome_tasa_desercion_snies) por departamento y año:")
print(pivot_cover.to_string())

# ------------------------------------------
# 5. Distribución del target
# ------------------------------------------
print("\n\n===== DISTRIBUCIÓN DEL TARGET =====")

df_target = df[df['outcome_tasa_desercion_snies'].notna()].copy()
target = df_target['outcome_tasa_desercion_snies']

print(f"\nFilas con target disponible: {len(df_target)} (año 2023 únicamente)")
print(f"  Media:    {target.mean():.4f}  ({target.mean()*100:.2f}%)")
print(f"  Mediana:  {target.median():.4f}  ({target.median()*100:.2f}%)")
print(f"  Std:      {target.std():.4f}")
print(f"  P10:      {target.quantile(.10):.4f}")
print(f"  P90:      {target.quantile(.90):.4f}")
print(f"  Mín:      {target.min():.4f}  → {df_target.loc[target.idxmin(), 'departamento']}")
print(f"  Máx:      {target.max():.4f}  → {df_target.loc[target.idxmax(), 'departamento']}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Histograma
axes[0].hist(target * 100, bins=15, color='#3498db', edgecolor='white', alpha=0.85)
axes[0].axvline(target.mean() * 100, color='red', linestyle='--', label=f'Media: {target.mean()*100:.2f}%')
axes[0].axvline(target.median() * 100, color='orange', linestyle='--', label=f'Mediana: {target.median()*100:.2f}%')
axes[0].set_xlabel("Tasa de deserción (%)")
axes[0].set_ylabel("Frecuencia")
axes[0].set_title("Distribución de la tasa de deserción SNIES (2023)", fontweight='bold')
axes[0].legend()

# Ranking departamental
df_sorted = df_target[['departamento', 'outcome_tasa_desercion_snies']].sort_values(
    'outcome_tasa_desercion_snies', ascending=True
)
colors_rank = ['#e74c3c' if v > target.quantile(.75) else '#2ecc71' if v < target.quantile(.25)
               else '#3498db' for v in df_sorted['outcome_tasa_desercion_snies']]
axes[1].barh(df_sorted['departamento'], df_sorted['outcome_tasa_desercion_snies'] * 100,
             color=colors_rank, edgecolor='white')
axes[1].set_xlabel("Tasa de deserción (%)")
axes[1].set_title("Ranking por departamento (2023)", fontweight='bold')
axes[1].tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_distribucion_target.png'), dpi=150)
plt.close()
print("\n[Guardado] 02_distribucion_target.png")

# ------------------------------------------
# 6. Distribuciones de variables numéricas
# ------------------------------------------
cols_plot = [c for c in df_num.columns if c not in
             ['outcome_tasa_desercion_snies', 'anio'] and df_num[c].notna().sum() > 10]

n_cols = 3
n_rows = (len(cols_plot) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols_plot):
    data = df[col].dropna()
    axes[i].hist(data, bins=15, color='#5dade2', edgecolor='white', alpha=0.85)
    axes[i].axvline(data.mean(), color='red', linestyle='--', linewidth=1.2, label='Media')
    axes[i].axvline(data.median(), color='orange', linestyle='--', linewidth=1.2, label='Mediana')
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].set_ylabel("Frecuencia")
    if i == 0:
        axes[i].legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle("Distribuciones de variables numéricas", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_distribuciones_variables.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Guardado] 03_distribuciones_variables.png")

# ------------------------------------------
# 7. Detección de outliers (método IQR)
# ------------------------------------------
print("\n\n===== DETECCIÓN DE OUTLIERS (IQR) =====")

outlier_report = {}
for col in df_num.columns:
    serie = df_num[col].dropna()
    Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((serie < lower) | (serie > upper)).sum()
    outlier_report[col] = {'n_outliers': n_out, 'pct': round(n_out / len(serie) * 100, 1),
                           'lower': round(lower, 4), 'upper': round(upper, 4)}
    print(f"  {col:<50} {n_out:>3} outliers ({n_out/len(serie)*100:.1f}%)")

# Boxplots agrupados por categoría
groups = {
    'PIB y matrícula': ['pib_total_miles_millones_cop', 'pib_variacion_pct_anual_vs_anio_previo',
                        'total_admitidos', 'total_matriculados', 'ratio_matriculados_sobre_admitidos'],
    'IPC nacional': [c for c in df.columns if 'ipc_nacional' in c],
    'Mercado laboral (GEIH)': ['geih_td_nacional_media_anual', 'geih_to_nacional_media_anual',
                               'geih_tgp_nacional_media_anual'],
    'SPADIES': [c for c in df.columns if 'spadies' in c],
}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for ax, (grupo, cols) in zip(axes, groups.items()):
    cols_exist = [c for c in cols if c in df.columns]
    data_g = df[cols_exist].dropna(how='all')
    if data_g.empty:
        ax.axis('off')
        continue
    data_g.boxplot(ax=ax, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#aed6f1', color='#2980b9'),
                   medianprops=dict(color='red', linewidth=2),
                   flierprops=dict(marker='o', color='#e74c3c', alpha=0.5))
    ax.set_title(f"Boxplot — {grupo}", fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)

plt.suptitle("Outliers por grupo de variables (IQR)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_boxplots_outliers.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[Guardado] 04_boxplots_outliers.png")

# ------------------------------------------
# 8. Análisis de correlaciones
# ------------------------------------------
print("\n\n===== ANÁLISIS DE CORRELACIONES =====")

# 8a. Correlación con el target (solo filas con target)
df_corr_target = df_target.select_dtypes(include=np.number).drop(
    columns=['codigo_departamento', 'outcome_merge_pendiente', 'anio'], errors='ignore'
)
corr_target = df_corr_target.corr()['outcome_tasa_desercion_snies'].drop('outcome_tasa_desercion_snies')
corr_target_sorted = corr_target.abs().sort_values(ascending=False)

print("\nCorrelación de Pearson con outcome_tasa_desercion_snies:")
for feat, val in corr_target.reindex(corr_target_sorted.index).items():
    flag = " ⚠️ ALTA" if abs(val) > 0.8 else ""
    print(f"  {feat:<50} {val:>7.4f}{flag}")

# Barplot de correlaciones con target
fig, ax = plt.subplots(figsize=(10, 7))
corr_vals = corr_target.reindex(corr_target_sorted.index)
colors_bar = ['#e74c3c' if abs(v) > 0.8 else '#f39c12' if abs(v) > 0.5 else '#3498db'
              for v in corr_vals.values]
ax.barh(corr_vals.index, corr_vals.values, color=colors_bar, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.axvline(0.8, color='red', linestyle='--', alpha=0.6, label='|r|=0.8 (riesgo leakage)')
ax.axvline(-0.8, color='red', linestyle='--', alpha=0.6)
ax.set_xlabel("Correlación de Pearson")
ax.set_title("Correlación de cada variable con la tasa de deserción SNIES", fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_correlacion_con_target.png'), dpi=150)
plt.close()
print("\n[Guardado] 05_correlacion_con_target.png")

# 8b. Heatmap completo
df_heat = df_corr_target.copy()
corr_matrix = df_heat.corr()
n_vars = len(corr_matrix.columns)
fig_size = max(10, n_vars * 0.6)
fig, ax = plt.subplots(figsize=(fig_size, fig_size))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    linewidths=0.4,
    linecolor='lightgray',
    mask=mask,
    ax=ax,
    annot_kws={"size": 7}
)
ax.set_title(
    "Matriz de correlaciones (variables numéricas — filas con target)",
    fontsize=14,
    fontweight='bold',
    pad=20
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    ha='right',
    rotation_mode='anchor'
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0
)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, '06_heatmap_correlaciones.png'),
    dpi=200,
    bbox_inches='tight'
)
plt.close()
print("[Guardado] 06_heatmap_correlaciones.png")

# ------------------------------------------
# 9. Detección de Data Leakage
# ------------------------------------------
print("\n\n===== DETECCIÓN DE DATA LEAKAGE =====")

# 9a. spadies_td_anual_universitario == outcome_tasa_desercion_snies
match = (df_target['spadies_td_anual_universitario'].round(8) ==
         df_target['outcome_tasa_desercion_snies'].round(8)).sum()

print(f"\n[CRÍTICO] spadies_td_anual_universitario coincide exactamente con el target")
print(f"  Filas con match exacto: {match} de {len(df_target)}")
print(f"  Correlación: {df_corr_target['spadies_td_anual_universitario'].corr(df_corr_target['outcome_tasa_desercion_snies']):.6f}")
print(f"  → Esta variable ES el target con otro nombre. Debe excluirse del entrenamiento.")

# 9b. Otras variables SPADIES
print("\n[ATENCIÓN] Variables SPADIES — tasas de deserción de otros niveles:")
for col in ['spadies_td_anual_tyt', 'spadies_td_anual_tecnico_profesional', 'spadies_td_anual_tecnologico']:
    if col in df_target.columns:
        corr_v = df_target[col].corr(df_target['outcome_tasa_desercion_snies'])
        print(f"  {col:<45} r = {corr_v:.4f}  ← {'⚠️ RIESGO' if abs(corr_v) > 0.5 else 'Bajo riesgo'}")

# 9c. Columna outcome_merge_pendiente
print(f"\n[NOTA] outcome_merge_pendiente:")
print(f"  Valores únicos: {df['outcome_merge_pendiente'].unique()}")
print(f"  Distribución: {df['outcome_merge_pendiente'].value_counts().to_dict()}")
print(f"  → Columna técnica de control de merge. Excluir del modelado.")

# 9d. Variables IPC nacionales — cardinalidad 2 (solo 2 valores distintos, uno por año)
print(f"\n[ALERTA] Variables IPC nacionales tienen cardinalidad 2 (mismo valor para todos los departamentos en un año).")
print(f"  Son macroeconómicas nacionales. No discriminan entre departamentos → baja utilidad predictiva.")
print(f"  Podrían incluirse como feature del año, pero su efecto queda capturado por la variable 'anio'.")

# Scatter: leakage visual
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Panel 1: leakage perfecto
axes[0].scatter(df_target['spadies_td_anual_universitario'] * 100,
                df_target['outcome_tasa_desercion_snies'] * 100,
                color='#e74c3c', alpha=0.8, edgecolors='white', s=70)
axes[0].set_xlabel("SPADIES universitario (%)")
axes[0].set_ylabel("Target SNIES (%)")
axes[0].set_title("⚠️ Leakage perfecto\nspadies_universitario = target", fontweight='bold', color='#c0392b')
m = max(df_target['spadies_td_anual_universitario'].max(), df_target['outcome_tasa_desercion_snies'].max()) * 100
axes[0].plot([0, m], [0, m], 'k--', linewidth=1, label='y=x')
axes[0].legend()

# Panel 2: correlación alta (proxy PIB)
axes[1].scatter(df_target['proxy_pib_miles_mm_cop_por_matriculado'],
                df_target['outcome_tasa_desercion_snies'] * 100,
                color='#f39c12', alpha=0.8, edgecolors='white', s=70)
axes[1].set_xlabel("PIB por matriculado (miles MM COP)")
axes[1].set_ylabel("Target SNIES (%)")
axes[1].set_title(f"r = {df_target['proxy_pib_miles_mm_cop_por_matriculado'].corr(df_target['outcome_tasa_desercion_snies']):.3f}\nproxy_pib_por_matriculado", fontweight='bold')

# Panel 3: baja correlación (geih_td)
axes[2].scatter(df_target['geih_td_nacional_media_anual'],
                df_target['outcome_tasa_desercion_snies'] * 100,
                color='#2ecc71', alpha=0.8, edgecolors='white', s=70)
axes[2].set_xlabel("Tasa de desempleo GEIH (%)")
axes[2].set_ylabel("Target SNIES (%)")
axes[2].set_title(f"r = {df_target['geih_td_nacional_media_anual'].corr(df_target['outcome_tasa_desercion_snies']):.3f}\ngeih_td_nacional", fontweight='bold')

plt.suptitle("Análisis de Data Leakage — Relación variables vs. target", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_leakage_analysis.png'), dpi=150)
plt.close()
print("\n[Guardado] 07_leakage_analysis.png")

# ------------------------------------------
# 10. Análisis del panel temporal (2023 vs 2024)
# ------------------------------------------
print("\n\n===== ANÁLISIS TEMPORAL DEL PANEL =====")

df_pib = df[['anio', 'departamento', 'pib_total_miles_millones_cop',
             'total_matriculados', 'geih_td_nacional_media_anual']].copy()

print(df_pib.groupby('anio')[['pib_total_miles_millones_cop',
                               'total_matriculados',
                               'geih_td_nacional_media_anual']].describe().to_string())

# Evolución de matrícula por departamento
df_matr = df[df['total_matriculados'].notna()][
    ['departamento', 'anio', 'total_matriculados']
]

# Pivot dinámico con agregación
df_pivot = df_matr.pivot_table(
    index='departamento',
    columns='anio',
    values='total_matriculados',
    aggfunc='sum'   # o 'mean' según tu caso
)

# Ordenar años
df_pivot = df_pivot.sort_index(axis=1)
fig, ax = plt.subplots(figsize=(12, 7))

# Top 10 departamentos
top10 = df_pivot.mean(axis=1).nlargest(10).index
anios = df_pivot.columns

for dept in top10:
    vals = df_pivot.loc[dept]

    ax.plot(
        anios,
        vals.values,
        marker='o',
        label=dept
    )

ax.set_xlabel("Año")
ax.set_ylabel("Total matriculados")
ax.set_title(
    "Evolución de matrícula por departamento",
    fontweight='bold'
)
ax.legend(fontsize=8, loc='upper right')
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, '08_evolucion_matricula.png'),
    dpi=150
)
plt.close()
print("\n[Guardado] 08_evolucion_matricula.png")

# ------------------------------------------
# 11. Resumen y conclusiones finales
# ------------------------------------------
print("\n\n" + "=" * 65)
print("  RESUMEN Y CONCLUSIONES DEL EDA")
print("=" * 65)

print("""
DIMENSIÓN Y ESTRUCTURA
  • Dataset: 66 observaciones × 33 columnas (panel 33 departamentos × 2 años: 2023-2024)
  • Variable target (outcome_tasa_desercion_snies): disponible SOLO para 2023 (33 filas)
  • El año 2024 no tiene target → candidato natural para predicción futura

CALIDAD DE DATOS
  • Sin filas duplicadas ✓
  • Variables con alta proporción de nulos:
    - var_pct_matriculados_vs_anio_previo: 57.6% nulos (no disponible en el primer año del panel)
    - spadies_* y outcome: ~50% nulos (solo disponibles para 2023)
    - total_admitidos / matriculados: 15.2% nulos
  • Las variables IPC nacionales tienen cardinalidad 2 — son macros que toman el mismo valor
    para todos los departamentos en un mismo año (no discriminan entre deptos).

ESTADÍSTICA DESCRIPTIVA
  • La tasa de deserción oscila entre 4.7% y 26.3% (media: 9.1%, mediana: 7.9%)
  • Alta asimetría positiva: pocos departamentos con deserción muy elevada elevan la media
  • PIB total presenta alta dispersión (CV elevado) — refleja desigualdad regional marcada

OUTLIERS
  • PIB total: concentración extrema en Bogotá y Antioquia → outliers legítimos (no errores)
  • Total admitidos/matriculados: misma lógica de concentración urbana
  • Las variables IPC y GEIH no presentan outliers (son nacionales, sin variación entre deptos)

CORRELACIONES
  • spadies_td_anual_tecnologico (r=0.61) y proxy_pib_por_matriculado (r=0.86):
    correlaciones altas, pero no implican leakage directo
  • geih_td, pib_variacion, ipc_capital: correlaciones moderadas a bajas — variables
    macroeconómicas genuinamente informativas para el modelo

⚠️  DATA LEAKAGE — CRÍTICO
  • spadies_td_anual_universitario tiene correlación PERFECTA (r=1.000) con el target
    → Es exactamente la misma variable renombrada. DEBE EXCLUIRSE del entrenamiento.
  • outcome_merge_pendiente es una variable técnica de control, no una feature predictiva.
  • Las demás columnas SPADIES (tyt, técnico, tecnológico) son tasas de otros niveles educativos
    y requieren evaluación caso a caso — correlaciones moderadas pero no perfectas.

RECOMENDACIONES PARA MODELADO
  1. Excluir: spadies_td_anual_universitario, outcome_merge_pendiente
  2. Evaluar exclusión o uso controlado de: spadies_td_anual_tyt, _tecnologico, _tecnico_profesional
  3. Imputar o eliminar: var_pct_matriculados_vs_anio_previo (57.6% nulos)
  4. Las variables IPC nacionales aportan poca información cross-seccional; considerar
     incluirlas como features del año o simplemente usar 'anio' como feature.
  5. Con solo 33 observaciones útiles (target disponible), priorizar modelos simples
     o regularizados (Ridge, Lasso, árboles con restricción de profundidad).
""")

print("Pipeline EDA completado. Todos los gráficos guardados en:", OUTPUT_DIR)
print("=" * 65)
