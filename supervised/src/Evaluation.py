"""
evaluation.py
=============
Métricas y visualización de resultados de modelos supervisados.
Responsable: Juanes | Rama: feature/supervised-models

Estándar de evaluación del pipeline:
- RMSE, MAE, R², MAPE en test set
- Tabla comparativa unificada
- Visualizaciones de diagnóstico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Función de evaluación estándar ───────────────────────────────────────────

def evaluar_modelo(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   nombre: str = "Modelo") -> dict:
    """
    Calcula las 4 métricas estándar del pipeline.

    Retorna dict con: modelo, RMSE, MAE, R2, MAPE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # Evitar división por cero en MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"\n📊 {nombre}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print(f"   R²   : {r2:.4f}")
    print(f"   MAPE : {mape:.2f}%")

    return {
        "modelo": nombre,
        "RMSE": round(rmse, 4),
        "MAE":  round(mae, 4),
        "R2":   round(r2, 4),
        "MAPE": round(mape, 2)
    }


def evaluar_todos(modelos_entrenados: dict,
                  X_test: np.ndarray,
                  y_test: np.ndarray) -> pd.DataFrame:
    """
    Evalúa todos los modelos en el test set y retorna tabla comparativa.
    """
    resultados = []
    print("\n── Evaluación en test set ──────────────────────────────────")
    for nombre, modelo in modelos_entrenados.items():
        y_pred = modelo.predict(X_test)
        metricas = evaluar_modelo(y_test, y_pred, nombre)
        resultados.append(metricas)

    df = pd.DataFrame(resultados).set_index("modelo")
    df = df.sort_values("RMSE")

    print("\n\n📋 TABLA COMPARATIVA FINAL")
    print("=" * 50)
    print(df.to_string())
    print("=" * 50)
    print(f"\n🏆 Mejor modelo (menor RMSE): {df.index[0]}")

    return df


# ── Visualizaciones de diagnóstico ────────────────────────────────────────────

def plot_predicho_vs_real(modelos_entrenados: dict,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           guardar: bool = True):
    """
    Scatter plot de valores predichos vs. reales para cada modelo.
    """
    n = len(modelos_entrenados)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, (nombre, modelo) in enumerate(modelos_entrenados.items()):
        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")
        ax.set_xlabel("Deserción real (%)")
        ax.set_ylabel("Deserción predicha (%)")
        ax.set_title(f"{nombre}\nR² = {r2:.3f}")
        ax.legend(fontsize=8)

    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Predicho vs. Real — todos los modelos", fontsize=14, y=1.01)
    plt.tight_layout()

    if guardar:
        path = FIGURES_DIR / "predicho_vs_real.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✅ Guardado en {path}")
    plt.show()


def plot_residuales(modelos_entrenados: dict,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    guardar: bool = True):
    """
    Gráfico de residuales para diagnóstico de sesgo del modelo.
    """
    n = len(modelos_entrenados)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, (nombre, modelo) in enumerate(modelos_entrenados.items()):
        y_pred = modelo.predict(X_test)
        residuales = y_test - y_pred

        ax = axes[i]
        ax.scatter(y_pred, residuales, alpha=0.6, edgecolors="k", linewidths=0.5)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Valores predichos")
        ax.set_ylabel("Residuales")
        ax.set_title(f"{nombre}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Residuales vs. Valores predichos", fontsize=14)
    plt.tight_layout()

    if guardar:
        path = FIGURES_DIR / "residuales.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✅ Guardado en {path}")
    plt.show()


def plot_comparacion_metricas(df_metricas: pd.DataFrame,
                               guardar: bool = True):
    """
    Gráfico de barras comparando RMSE y R² de todos los modelos.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colores = plt.cm.tab10(np.linspace(0, 1, len(df_metricas)))

    # RMSE (menor = mejor)
    df_metricas["RMSE"].plot(kind="bar", ax=ax1, color=colores)
    ax1.set_title("RMSE (↓ mejor)")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylabel("RMSE")

    # R² (mayor = mejor)
    df_metricas["R2"].plot(kind="bar", ax=ax2, color=colores)
    ax2.set_title("R² (↑ mejor)")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_ylabel("R²")
    ax2.axhline(0, color="black", linewidth=0.8)

    plt.suptitle("Comparación de modelos supervisados", fontsize=14)
    plt.tight_layout()

    if guardar:
        path = FIGURES_DIR / "comparacion_modelos.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✅ Guardado en {path}")
    plt.show()


def plot_importancia_variables(modelo_pipeline,
                                feature_names: list,
                                nombre_modelo: str = "RandomForest",
                                top_n: int = 10,
                                guardar: bool = True):
    """
    Gráfico de importancia de variables para modelos de árbol.
    Solo funciona con RandomForest y GradientBoosting.
    """
    try:
        model = modelo_pipeline.named_steps["model"]
        importancias = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(8, 5))
        importancias.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Importancia de variables — {nombre_modelo}")
        ax.set_xlabel("Importancia (Gini / MSE reduction)")
        plt.tight_layout()

        if guardar:
            path = FIGURES_DIR / f"importancia_{nombre_modelo.lower()}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"✅ Guardado en {path}")
        plt.show()

    except AttributeError:
        print(f"⚠️  {nombre_modelo} no tiene feature_importances_. Usa RF o GBM.")


# ── Guardar tabla de resultados ───────────────────────────────────────────────

def guardar_tabla_resultados(df_metricas: pd.DataFrame,
                              cv_results: pd.DataFrame = None):
    """
    Guarda la tabla comparativa en CSV para el reporte final.
    """
    if cv_results is not None:
        df_completo = df_metricas.join(cv_results, how="left")
    else:
        df_completo = df_metricas

    path = Path("reports/tabla_comparativa_modelos.csv")
    path.parent.mkdir(exist_ok=True)
    df_completo.to_csv(path)
    print(f"✅ Tabla guardada en {path}")
    return df_completo


# ── Uso de ejemplo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from models import preparar_datos, entrenar_todos
    import pandas as pd

    df = pd.read_csv("data/processed/dataset_con_indice.csv")
    X_train, X_test, y_train, y_test, features = preparar_datos(df)
    modelos, cv_results = entrenar_todos(X_train, y_train)

    df_metricas = evaluar_todos(modelos, X_test, y_test)
    guardar_tabla_resultados(df_metricas, cv_results)

    plot_predicho_vs_real(modelos, X_test, y_test)
    plot_residuales(modelos, X_test, y_test)
    plot_comparacion_metricas(df_metricas)

    if "RandomForest" in modelos:
        plot_importancia_variables(modelos["RandomForest"], features, "RandomForest")