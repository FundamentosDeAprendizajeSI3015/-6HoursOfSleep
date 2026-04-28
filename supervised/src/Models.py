"""
models.py
=========
Implementación de modelos supervisados para predicción de deserción.
Responsable: Juanes | Rama: feature/supervised-models

Incluye:
- Modelos baseline (regresión lineal, Ridge, Lasso, ElasticNet)
- Modelos avanzados (Random Forest, Gradient Boosting, SVR, KNN)
- Función de entrenamiento con validación cruzada estandarizada
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ── Configuración global del pipeline ────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

FEATURE_COLS = [
    "desempleo", "pobreza", "inflacion", "ipc_educacion",
    "tasa_interes", "pib_departamental", "crecimiento_pib",
    "ingreso_real", "ratio_desempleo_pobreza",
    "indice_vulnerabilidad_pca", "indice_vulnerabilidad_teorico"
]
TARGET_COL = "desercion"


# ── Definición de modelos ─────────────────────────────────────────────────────

def get_modelos_baseline() -> dict:
    """
    Retorna modelos lineales básicos como pipelines con estandarización.
    Todos incluyen StandardScaler para comparación justa.
    """
    return {
        "OLS": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1, max_iter=5000))
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000))
        ])
    }


def get_modelos_avanzados() -> dict:
    """
    Retorna modelos avanzados. Los basados en árboles no necesitan
    estandarización, pero se incluye en pipeline para consistencia.
    """
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=5,
                random_state=RANDOM_STATE
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE
            ))
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5))
        ])
    }


def get_todos_los_modelos() -> dict:
    """Retorna el diccionario completo de todos los modelos."""
    return {**get_modelos_baseline(), **get_modelos_avanzados()}


# ── Split de datos ────────────────────────────────────────────────────────────

def preparar_datos(df: pd.DataFrame,
                   features: list = None,
                   target: str = TARGET_COL) -> tuple:
    """
    Prepara X e y, elimina nulos y hace el split train/test.

    Retorna
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    if features is None:
        features = [f for f in FEATURE_COLS if f in df.columns]

    # Usar solo filas con todos los datos completos
    cols_necesarias = features + [target]
    df_clean = df[cols_necesarias].dropna()

    if len(df_clean) < 30:
        print(f"⚠️  Solo {len(df_clean)} filas completas. El modelo puede no ser confiable.")

    X = df_clean[features].values
    y = df_clean[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"✅ Split: {len(X_train)} train | {len(X_test)} test")
    print(f"   Features usadas: {features}")

    return X_train, X_test, y_train, y_test, features


# ── Entrenamiento con CV ──────────────────────────────────────────────────────

def entrenar_con_cv(modelo, X_train: np.ndarray, y_train: np.ndarray,
                    nombre: str = "Modelo") -> dict:
    """
    Entrena un modelo con validación cruzada k-fold.

    Retorna métricas de CV para el notebook de comparación.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        modelo, X_train, y_train,
        cv=kf,
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
        return_train_score=True
    )

    rmse_cv = -cv_results["test_neg_root_mean_squared_error"]
    mae_cv  = -cv_results["test_neg_mean_absolute_error"]
    r2_cv   =  cv_results["test_r2"]

    print(f"  {nombre} CV ({CV_FOLDS}-fold):")
    print(f"    RMSE: {rmse_cv.mean():.4f} ± {rmse_cv.std():.4f}")
    print(f"    MAE:  {mae_cv.mean():.4f} ± {mae_cv.std():.4f}")
    print(f"    R²:   {r2_cv.mean():.4f} ± {r2_cv.std():.4f}")

    # Entrenar en todo el conjunto de entrenamiento
    modelo.fit(X_train, y_train)

    return {
        "modelo": nombre,
        "cv_rmse_mean": rmse_cv.mean(),
        "cv_rmse_std":  rmse_cv.std(),
        "cv_mae_mean":  mae_cv.mean(),
        "cv_r2_mean":   r2_cv.mean()
    }


# ── Entrenamiento completo de todos los modelos ───────────────────────────────

def entrenar_todos(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Entrena todos los modelos con validación cruzada.

    Retorna
    -------
    modelos_entrenados : dict con pipelines ya ajustados
    resultados_cv      : DataFrame con métricas de CV
    """
    modelos = get_todos_los_modelos()
    resultados = []
    modelos_entrenados = {}

    print(f"\n🔄 Entrenando {len(modelos)} modelos con {CV_FOLDS}-fold CV...\n")

    for nombre, modelo in modelos.items():
        metricas = entrenar_con_cv(modelo, X_train, y_train, nombre)
        resultados.append(metricas)
        modelos_entrenados[nombre] = modelo

    df_resultados = pd.DataFrame(resultados).set_index("modelo")
    return modelos_entrenados, df_resultados


# ── Uso de ejemplo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/processed/dataset_con_indice.csv")

    X_train, X_test, y_train, y_test, features = preparar_datos(df)
    modelos, cv_results = entrenar_todos(X_train, y_train)

    print("\n📊 Resultados de CV:")
    print(cv_results.sort_values("cv_rmse_mean"))