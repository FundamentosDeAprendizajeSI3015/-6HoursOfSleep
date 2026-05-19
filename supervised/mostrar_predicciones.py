"""
mostrar_predicciones.py
=======================
Ejecuta el pipeline completo y muestra una tabla comprensible:
  Departamento | Deserción Real | Predicción | Error

Ejecutar desde cualquier carpeta:
    python mostrar_predicciones.py
"""

import sys
import os
from pathlib import Path

# Configurar rutas para encontrar los módulos en src/
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / 'src'))

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from data_loader import DataLoader
from Models import preparar_datos, entrenar_todos_modelos
from Evaluation import plot_comparacion_departamentos, plot_predicho_vs_real

# ── 1. Encontrar el archivo de datos ─────────────────────────────────────────
# Probamos varias ubicaciones posibles
possible_data_paths = [
    REPO_ROOT / "data" / "final" / "dataset_con_indice.csv",
    SCRIPT_DIR.parent / "data" / "final" / "dataset_con_indice.csv",
    Path("data/final/dataset_con_indice.csv"),
    Path("../data/final/dataset_con_indice.csv")
]

DATA_PATH = None
for p in possible_data_paths:
    if p.exists():
        DATA_PATH = str(p.resolve())
        break

if DATA_PATH is None:
    print(f"ERROR: No se encontró el archivo dataset_con_indice.csv")
    print(f"Buscamos en: {[str(p) for p in possible_data_paths]}")
    sys.exit(1)

print("="*65)
print("  PREDICCIONES POR DEPARTAMENTO — ¿Qué predijimos realmente?")
print(f"  Archivo: {DATA_PATH}")
print("="*65)

loader = DataLoader(DATA_PATH)
loader.clean_data()
X, y = loader.prepare_features()

# Guardar info de departamentos ANTES del split (para etiquetar después)
df_raw = loader.df_clean[["departamento", "anio"]].reset_index(drop=True)

# ── 2. Split y entrenamiento ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = preparar_datos(X, y)

print("\nEntrenando modelos (puede tardar unos segundos)...")
modelos, cv_results = entrenar_todos_modelos(X_train, y_train)

# ── 3. Predicciones del mejor modelo en TEST ────────────────────────────────
mejor_modelo_nombre = "GradientBoosting"
if mejor_modelo_nombre not in modelos:
    mejor_modelo_nombre = list(modelos.keys())[0]

modelo_gb = modelos[mejor_modelo_nombre]
y_pred = modelo_gb.predict(X_test)

# Recuperar índices originales del test set
idx_test = X_test.index 

# ── 4. Tabla: Real vs Predicho ───────────────────────────────────────────────
df_resultado = pd.DataFrame({
    "departamento": df_raw.loc[idx_test, "departamento"].values,
    "anio":         df_raw.loc[idx_test, "anio"].values,
    "desercion_real_%":    (y_test.values * 100).round(2),
    "prediccion_%":        (y_pred * 100).round(2),
    "error_pp":            ((y_pred - y_test.values) * 100).round(2),
}).sort_values("desercion_real_%", ascending=False).reset_index(drop=True)

df_resultado["sobreestima"] = df_resultado["error_pp"].apply(
    lambda e: "⬆ Sobreestima" if e > 0.5 else ("⬇ Subestima" if e < -0.5 else "✓ Cercano")
)

print("\n")
print("="*75)
print(f"  MODELO: {mejor_modelo_nombre} — Predicciones en TEST SET (datos nunca vistos)")
print("="*75)
print(f"\n{'Departamento':<35} {'Año':>5} {'Real (%)':>10} {'Pred (%)':>10} {'Error (pp)':>12} {'':>15}")
print("-"*75)
for _, row in df_resultado.iterrows():
    print(f"  {row['departamento']:<33} {int(row['anio']):>5} {row['desercion_real_%']:>10.2f} "
          f"{row['prediccion_%']:>10.2f} {row['error_pp']:>+12.2f}  {row['sobreestima']}")

print("-"*75)
print(f"\n  {'MAE promedio:':30} ±{df_resultado['error_pp'].abs().mean():.2f} pp")
print(f"  {'Error máximo:':30}  {df_resultado['error_pp'].abs().max():.2f} pp")

print("\n")
print("="*65)
print("  TODOS LOS 33 DEPARTAMENTOS — PREDICCIÓN FULL DATASET")
print("="*65)

modelo_full = modelos[mejor_modelo_nombre]
# Reentrenar en todos los datos para visualización (Opcional)
X_all = X.copy().fillna(X.mean(numeric_only=True))
modelo_full.fit(X_all, y)
y_pred_all = modelo_full.predict(X_all)

df_full = pd.DataFrame({
    "departamento":     df_raw["departamento"].values[:len(y)],
    "anio":             df_raw["anio"].values[:len(y)],
    "desercion_real_%": (y.values * 100).round(2),
    "prediccion_%":     (y_pred_all * 100).round(2),
    "error_pp":         ((y_pred_all - y.values) * 100).round(2),
}).sort_values("desercion_real_%", ascending=False).reset_index(drop=True)

print(f"\n{'Departamento':<40} {'Año':>5} {'Real (%)':>10} {'Pred (%)':>10} {'Error (pp)':>12}")
print("-"*80)
for _, row in df_full.iterrows():
    flag = " ★" if abs(row["error_pp"]) < 1.5 else ""
    print(f"  {row['departamento']:<38} {int(row['anio']):>5} {row['desercion_real_%']:>10.2f} "
          f"{row['prediccion_%']:>10.2f} {row['error_pp']:>+12.2f}{flag}")

# ── 5. Generar Gráficas ──────────────────────────────────────────────────────
print("\nGenerando gráficas de predicciones...")

# 5.1 Gráfica de Dispersión (Existente)
plot_predicho_vs_real({mejor_modelo_nombre: modelo_full}, X_all.values, y.values, guardar=True)

# 5.2 Gráfica de Barras Comparativa (Nueva)
plot_comparacion_departamentos(df_full, mejor_modelo_nombre, guardar=True)

print("\n" + "="*75)
print("  RESUMEN DE PREDICCIONES (Tabla Comparativa)")
print("="*75)
# Mostrar los top 10 con mayor error para análisis
df_full["error_abs"] = df_full["error_pp"].abs()
print("\nTop 10 departamentos con mayor desviación:")
print(df_full.sort_values("error_abs", ascending=False)[["departamento", "anio", "desercion_real_%", "prediccion_%", "error_pp"]].head(10).to_string(index=False))

print("\n" + "="*75)
print(f"✅ Proceso completado. Gráficas guardadas en reports/eda_figures/")
print("="*75)


