"""
evaluation.py
=============
Evaluación comprehensiva de modelos supervisados para predicción de deserción.
Responsable: Juanes | Rama: feature/supervised-models

Componentes:
├-- Cálculo de métricas (RMSE, MAE, R², MAPE)
├-- Tabla comparativa de todos los modelos
├-- Visualizaciones de diagnóstico (scatter, residuales, comparación)
├-- Feature importance para modelos de árbol
└-- Exportación de resultados a CSV

Estándar del pipeline:
- RMSE, MAE, R², MAPE en test set
- Validación cruzada: Media ± Std
- Visualizaciones en reports/figures/
- Tabla comparativa en reports/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

FIGURES_DIR = Path("./reports/eda_figures")
REPORTS_DIR = Path("./reports")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# -- MÉTRICAS ESTÁNDAR DEL PIPELINE --------------------------------------------

def evaluar_modelo(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   nombre: str = "Modelo") -> dict:
    """
    Calcula las 4 métricas estándar del pipeline para un modelo.
    
    Métricas:
    - RMSE (Root Mean Squared Error): Penaliza errores grandes. Unidad: %
    - MAE (Mean Absolute Error): Error promedio absoluto. Unidad: %
    - R² (Coeficiente de Determinación): Proporción de varianza explicada [0,1]
    - MAPE (Mean Absolute Percentage Error): Error porcentual medio (%)
    
    Args:
        y_true: Valores reales (test set)
        y_pred: Valores predichos
        nombre: Nombre del modelo (para logging)
    
    Returns:
        dict: {modelo, RMSE, MAE, R2, MAPE} redondeados
    """
    # Calcular métricas básicas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE: Evitar división por cero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # Mostrar resultados
    print(f"\n[METRICS] {nombre}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print(f"   R²   : {r2:.4f}")
    print(f"   MAPE : {mape:.2f}%")
    
    return {
        "modelo": nombre,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2)
    }


def evaluar_todos(modelos_entrenados: dict,
                  X_test: np.ndarray,
                  y_test: np.ndarray) -> pd.DataFrame:
    """
    Evalúa TODOS los modelos en el test set y retorna tabla comparativa.
    
    Proceso:
    1. Itera sobre todos los modelos entrenados
    2. Realiza predicciones en X_test
    3. Calcula métricas para cada modelo
    4. Ordena por RMSE (métrica principal)
    5. Imprime tabla resumen
    
    Args:
        modelos_entrenados: Dict {nombre: Pipeline.fit()}
        X_test: Features de test set
        y_test: Target de test set
    
    Returns:
        DataFrame con resultados ordenados por RMSE ascendente
    """
    resultados = []
    
    print("\n" + "="*70)
    print(" EVALUACIÓN EN TEST SET (Evaluación Final)")
    print("="*70)
    
    for nombre, modelo in modelos_entrenados.items():
        y_pred = modelo.predict(X_test)
        metricas = evaluar_modelo(y_test, y_pred, nombre)
        resultados.append(metricas)
    
    # Crear DataFrame y ordenar por RMSE
    df = pd.DataFrame(resultados).set_index("modelo")
    df = df.sort_values("RMSE")
    
    # Mostrar tabla final
    print("\n" + "="*70)
    print("[TOP] RANKING FINAL DE MODELOS (por RMSE)")
    print("="*70)
    print(df.to_string())
    print("="*70)
    print(f"\n[OK] Mejor modelo: {df.index[0]} (RMSE: {df.iloc[0]['RMSE']:.4f})")
    
    return df


# -- VISUALIZACIONES DE DIAGNÓSTICO --------------------------------------------

def plot_predicho_vs_real(modelos_entrenados: dict,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          guardar: bool = True):
    """
    Scatter plot: Predicho vs. Real para diagnóstico visual.
    
    Interpretación:
    - Puntos sobre la diagonal roja → Predicción perfecta
    - Puntos arriba de diagonal → Subestimación
    - Puntos abajo de diagonal → Sobreestimación
    - Dispersión grande → Modelo con alta varianza
    
    Args:
        modelos_entrenados: Dict de modelos entrenados
        X_test: Features de test
        y_test: Target real de test
        guardar: Si se guarda imagen
    """
    n = len(modelos_entrenados)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    
    for i, (nombre, modelo) in enumerate(modelos_entrenados.items()):
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        ax = axes[i]
        # Scatter de predicciones
        ax.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors="k", linewidths=0.5)
        
        # Línea de predicción perfecta
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Predicción perfecta")
        
        ax.set_xlabel("Deserción real (%)", fontsize=10)
        ax.set_ylabel("Deserción predicha (%)", fontsize=10)
        ax.set_title(f"{nombre}\nR² = {r2:.3f} | RMSE = {rmse:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
    
    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Predicho vs. Real — Diagnóstico de todos los modelos", 
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    
    if guardar:
        path = FIGURES_DIR / "supervisado_predicho_vs_real.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[OK] Guardado: {path}")
    plt.show()


def plot_residuales(modelos_entrenados: dict,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   guardar: bool = True):
    """
    Gráfico de residuales vs. predichos.
    
    Interpretación:
    - Residuales aleatoriamente dispersos → Buen modelo
    - Patrón en forma de embudo → Heteroscedasticidad (varianza no constante)
    - Línea horizontal en 0 → Sin sesgo sistemático
    - Dispersión grande → Sobreestimación/subestimación frecuente
    
    Args:
        modelos_entrenados: Dict de modelos entrenados
        X_test: Features de test
        y_test: Target real de test
        guardar: Si se guarda imagen
    """
    n = len(modelos_entrenados)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    
    for i, (nombre, modelo) in enumerate(modelos_entrenados.items()):
        y_pred = modelo.predict(X_test)
        residuales = y_test - y_pred
        mae = mean_absolute_error(y_test, y_pred)
        
        ax = axes[i]
        ax.scatter(y_pred, residuales, alpha=0.6, s=50, edgecolors="k", linewidths=0.5)
        ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Residual = 0")
        
        ax.set_xlabel("Valores predichos (%)", fontsize=10)
        ax.set_ylabel("Residuales (%)", fontsize=10)
        ax.set_title(f"{nombre}\nMAE = {mae:.4f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle("Residuales vs. Predichos — Detección de sesgo", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if guardar:
        path = FIGURES_DIR / "supervisado_residuales.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[OK] Guardado: {path}")
    plt.show()


def plot_comparacion_metricas(df_metricas: pd.DataFrame,
                              guardar: bool = True):
    """
    Gráfico comparativo de RMSE y R² de todos los modelos.
    
    Visualización:
    - Izquierda: RMSE (menor = mejor) → Magnitud de error
    - Derecha: R² (mayor = mejor) → Proporción de varianza explicada
    
    Args:
        df_metricas: DataFrame con resultados
        guardar: Si se guarda imagen
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    colores = plt.cm.tab10(np.linspace(0, 1, len(df_metricas)))
    
    # Subplot 1: RMSE (menor = mejor)
    df_metricas["RMSE"].plot(kind="bar", ax=ax1, color=colores, alpha=0.8, edgecolor="k")
    ax1.set_title("RMSE - Error Cuadrático Medio\n(↓ menor = mejor)", 
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Modelo", fontsize=11)
    ax1.set_ylabel("RMSE", fontsize=11)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Subplot 2: R² (mayor = mejor)
    df_metricas["R2"].plot(kind="bar", ax=ax2, color=colores, alpha=0.8, edgecolor="k")
    ax2.set_title("R² - Proporción de Varianza Explicada\n(↑ mayor = mejor)", 
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Modelo", fontsize=11)
    ax2.set_ylabel("R²", fontsize=11)
    ax2.tick_params(axis="x", rotation=45)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle("Comparación de Modelos Supervisados — Métricas Clave", 
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if guardar:
        path = FIGURES_DIR / "supervisado_comparacion_metricas.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[OK] Guardado: {path}")
    plt.show()


def plot_importancia_variables(modelo_pipeline,
                               feature_names: list,
                               nombre_modelo: str = "RandomForest",
                               top_n: int = 10,
                               guardar: bool = True):
    """
    Gráfico de importancia de variables para modelos basados en árboles.
    
    Interpretación:
    - Importancia basada en reducción de impureza (Gini/MSE)
    - Variables con mayor importancia → Mayores contribuciones al modelo
    - Útil para entender qué variables impulsan las predicciones
    
    Args:
        modelo_pipeline: Pipeline con modelo ajustado
        feature_names: Lista de nombres de features
        nombre_modelo: Nombre para logging
        top_n: Número de features más importantes a mostrar
        guardar: Si se guarda imagen
    """
    try:
        # Extraer el modelo del pipeline
        model = modelo_pipeline.named_steps["model"]
        
        # Obtener importancias y crear Series
        importancias = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=True).tail(top_n)
        
        # Crear visualización
        fig, ax = plt.subplots(figsize=(10, 6))
        importancias.plot(kind="barh", ax=ax, color="steelblue", alpha=0.8, edgecolor="k")
        
        ax.set_title(f"Feature Importance — {nombre_modelo}\n(Top {top_n} variables)", 
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Importancia (Impureza reducida)", fontsize=11)
        ax.set_ylabel("Variable", fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        
        plt.tight_layout()
        
        if guardar:
            path = FIGURES_DIR / f"supervisado_importancia_{nombre_modelo.lower()}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[OK] Guardado: {path}")
        plt.show()
        
    except AttributeError:
        print(f"\n[WARNING]  {nombre_modelo} no tiene atributo 'feature_importances_'")
        print(f"   Nota: Feature importance está disponible solo para Random Forest y Gradient Boosting")


# -- EXPORTACIÓN DE RESULTADOS ------------------------------------------------

def guardar_tabla_resultados(df_metricas: pd.DataFrame,
                             cv_results: pd.DataFrame = None) -> pd.DataFrame:
    """
    Guarda tabla comparativa de resultados en CSV.
    
    Estructura del archivo:
    - Índice: Nombres de modelos
    - Columnas: RMSE, MAE, R², MAPE (test) + CV metrics (si aplica)
    
    Args:
        df_metricas: DataFrame con métricas de test
        cv_results: DataFrame opcionales con resultados de CV
    
    Returns:
        DataFrame completo guardado
    """
    # Combinar test metrics con CV results si están disponibles
    if cv_results is not None:
        df_completo = df_metricas.join(cv_results, how="left")
    else:
        df_completo = df_metricas
    
    # Guardar a CSV
    path = REPORTS_DIR / "tabla_comparativa_modelos_supervisados.csv"
    df_completo.to_csv(path)
    
    print(f"\n[OK] Tabla guardada: {path}")
    print(f"\n[METRICS] Contenido de tabla:")
    print(df_completo.to_string())
    
    return df_completo


# -- RESUMEN FINAL ------------------------------------------------------------

def generar_reporte(df_metricas: pd.DataFrame,
                   nombre_mejor: str,
                   mejora_baseline: float = None):
    """
    Genera un reporte textual con conclusiones del análisis supervisado.
    
    Args:
        df_metricas: DataFrame con métricas
        nombre_mejor: Nombre del modelo ganador
        mejora_baseline: Mejora porcentual vs OLS (opcional)
    """
    reporte = f"""
{'='*70}
[METRICS] REPORTE FINAL — MODELOS SUPERVISADOS
{'='*70}

[TOP] MODELO GANADOR: {nombre_mejor}

Métricas en Test Set:
  • RMSE: {df_metricas.loc[nombre_mejor, 'RMSE']:.4f}
  • MAE:  {df_metricas.loc[nombre_mejor, 'MAE']:.4f}
  • R²:   {df_metricas.loc[nombre_mejor, 'R2']:.4f}
  • MAPE: {df_metricas.loc[nombre_mejor, 'MAPE']:.2f}%

Benchmarks:
  • Modelos evaluados: {len(df_metricas)}
  • Modelo baseline (OLS): RMSE = {df_metricas.loc['OLS', 'RMSE']:.4f}
  
Mejora respecto a OLS:
  • Diferencia RMSE: {df_metricas.loc['OLS', 'RMSE'] - df_metricas.loc[nombre_mejor, 'RMSE']:.4f}
  • Mejora %: {100*(df_metricas.loc['OLS', 'RMSE'] - df_metricas.loc[nombre_mejor, 'RMSE'])/df_metricas.loc['OLS', 'RMSE']:.1f}%

{'='*70}
    """
    print(reporte)
    return reporte


# -- EJEMPLO DE USO ------------------------------------------------------------

if __name__ == "__main__":
    """
    Ejemplo de uso típico del módulo evaluation.py
    """
    from models import entrenar_todos_modelos, preparar_datos
    from data_loader import load_and_prepare
    
    # 1. Cargar y preparar datos
    X, y, loader = load_and_prepare()
    X_train, X_test, y_train, y_test = preparar_datos(X, y)
    
    # 2. Entrenar modelos
    modelos, cv_results = entrenar_todos_modelos(X_train, y_train)
    
    # 3. Evaluar en test
    df_metricas = evaluar_todos(modelos, X_test, y_test)
    
    # 4. Visualizar resultados
    plot_predicho_vs_real(modelos, X_test, y_test)
    plot_residuales(modelos, X_test, y_test)
    plot_comparacion_metricas(df_metricas)
    
    # 5. Feature importance (si aplica)
    if "RandomForest" in modelos:
        plot_importancia_variables(
            modelos["RandomForest"],
            X.columns.tolist(),
            "RandomForest"
        )
    
    # 6. Guardar resultados
    guardar_tabla_resultados(df_metricas, cv_results)
    
    # 7. Generar reporte
    mejor_modelo = df_metricas.index[0]
    generar_reporte(df_metricas, mejor_modelo)
