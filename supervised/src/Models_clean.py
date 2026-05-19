"""
Models.py
=========
Implementacion de modelos supervisados para prediccion de desercion estudiantil.
Responsable: Juanes | Rama: feature/supervised-models

Componentes principales:
- Modelos Baseline: Regresion lineal, Ridge, Lasso, ElasticNet
- Modelos Avanzados: Random Forest, Gradient Boosting, SVR, KNN
- Pipeline de entrenamiento: Validacion cruzada k-fold estandarizada

Flujo:
1. Cargar datos con data_loader.py
2. Preparar features y target
3. Entrenar todos los modelos con CV
4. Evaluar en test set
5. Generar reporte comparativo

Notas tecnicas:
- Todos los modelos incluyen StandardScaler (necesario para regularizacion)
- Split: 80% train / 20% test
- CV: 5-fold
- Random state: 42 (reproducibilidad)
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

# Desabilitar joblib parallel en Windows para evitar errores de subprocess
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"


# ============================================================================
# CONFIGURACION GLOBAL DEL PIPELINE
# ============================================================================

RANDOM_STATE = 42      # Para reproducibilidad
TEST_SIZE = 0.20       # 80% train, 20% test
CV_FOLDS = 5           # Validacion cruzada 5-fold
TARGET_COL = "outcome_tasa_desercion_snies"  # Variable objetivo


# ============================================================================
# MODELOS BASELINE - PUNTOS DE REFERENCIA
# ============================================================================
# Estos modelos son lineales e interpretables, sirven como benchmark para
# evaluar si los modelos mas complejos realmente mejoran la prediccion.

def get_modelos_baseline() -> dict:
    """
    Retorna modelos lineales basicos como pipelines con estandarizacion.
    
    JUSTIFICACION DE CADA MODELO:
    
    1. OLS (Ordinary Least Squares):
       - Baseline puro sin regularizacion
       - Interpretable (coeficientes directos)
       - Problema: Puede sufrir de overfitting con multicolinealidad
    
    2. Ridge (Regresion L2):
       - Agrega penalizacion L2 a la suma de cuadrados de coeficientes
       - Mantiene todas las variables (no las elimina)
       - Ventaja: Maneja multicolinealidad mejor que OLS
       - Usamos alpha=1.0 (penalizacion moderada)
    
    3. Lasso (Regresion L1):
       - Agrega penalizacion L1 basada en valor absoluto de coeficientes
       - SELECCIONA automaticamente variables (pone algunas a 0)
       - Ventaja: Feature selection automatica, modelos mas simples
       - Usamos alpha=0.1 (penalizacion menos agresiva que Ridge)
    
    4. ElasticNet (Combinacion L1+L2):
       - Combina ventajas de Ridge (multicolinealidad) y Lasso (sparsity)
       - Mas flexible y frecuentemente mejor que Ridge/Lasso solos
       - Usamos l1_ratio=0.5 (balance 50/50 entre L1 y L2)
    
    NOTA: Todos incluyen StandardScaler porque la regularizacion (L1, L2)
          depende de la magnitud de los coeficientes.
    
    Returns:
        dict: Diccionario {nombre: Pipeline}
    """
    return {
        "OLS": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1, max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, 
                                random_state=RANDOM_STATE))
        ])
    }


# ============================================================================
# MODELOS AVANZADOS - CAPTURAN RELACIONES NO-LINEALES
# ============================================================================
# Estos modelos pueden capturar patrones complejos que los modelos lineales
# no pueden detectar. Usan diferentes estrategias:
# - Tree-based: Division recursiva del espacio de features
# - Kernel-based: Transformacion implicita del espacio de features
# - Distance-based: Vecinos mas cercanos

def get_modelos_avanzados() -> dict:
    """
    Retorna modelos avanzados como pipelines.
    
    JUSTIFICACION DE CADA MODELO:
    
    1. Random Forest:
       - Ensemble de multiples arboles de decision
       - Cada arbol ve un subconjunto aleatorio de datos y features
       - Combina predicciones por promedio (reduce varianza)
       - Ventaja: Captura interacciones entre variables automaticamente
       - Desventaja: Menos interpretable que modelos lineales
       - Usamos n_estimators=100 arboles
    
    2. Gradient Boosting:
       - Ensemble que construye arboles secuencialmente
       - Cada nuevo arbol corrige errores del anterior (boosting)
       - Frecuentemente obtiene mejor rendimiento que Random Forest
       - Ventaja: Muy poderoso para datos complejos
       - Desventaja: Mas propenso a overfitting (necesita tuning)
       - Usamos n_estimators=100, learning_rate=0.1
    
    3. Support Vector Regressor (SVR):
       - Encuentra el hiperplano que maximiza el margen de prediccion
       - Usa kernel para transformar el espacio (kernel='rbf' es no-lineal)
       - Bueno con datos de alta dimension
       - Ventaja: Funciona bien con muchos features
       - Desventaja: Lento con muchas muestras (n=33 es pequeno, OK)
       - Usamos kernel='rbf' (radial basis function)
    
    4. K-Nearest Neighbors (KNN):
       - Prediccion basada en los k vecinos mas cercanos
       - Simple pero efectivo para patrones locales
       - Sensible a la escala (por eso StandardScaler es crucial)
       - Ventaja: No necesita ajuste explicito, bien con datos pequenos
       - Desventaja: Lento en prediccion, sensible a outliers
       - Usamos k=5 (vecinos), distancia Euclidiana
    
    NOTA: Todos usan StandardScaler porque el calculo de distancias
          depende de la magnitud de los features.
    
    Returns:
        dict: Diccionario {nombre: Pipeline}
    """
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                random_state=RANDOM_STATE
            ))
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
                gamma="scale"
            ))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=5,
                weights="distance",
                metric="euclidean"
            ))
        ])
    }


def get_todos_los_modelos() -> dict:
    """
    Retorna diccionario unificado de todos los modelos.
    
    Returns:
        dict: Baseline + Avanzados
    """
    return {**get_modelos_baseline(), **get_modelos_avanzados()}


# ============================================================================
# PREPARACION DE DATOS
# ============================================================================

def preparar_datos(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Realiza el split train/test del dataset.
    
    Usa imputacion (media) para manejar nulos en lugar de eliminar filas.
    
    Args:
        X: DataFrame de features (de data_loader)
        y: Series de target (de data_loader)
    
    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    # Crear DataFrame completo
    df_temp = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    
    # Contar nulos antes de imputar
    total_nulos = df_temp.isnull().sum().sum()
    if total_nulos > 0:
        print(f"\n[ADVERTENCIA] {total_nulos} valores nulos detectados")
        print(f"   Estrategia: Imputacion con media (evita eliminar filas)")
    
    # Imputar nulos solo en features (no en target)
    X_imputed = X.fillna(X.mean(numeric_only=True))
    
    # Eliminar filas donde el target es nulo
    mask_target_nulo = y.isnull()
    if mask_target_nulo.sum() > 0:
        print(f"\n[ADVERTENCIA] {mask_target_nulo.sum()} filas con target nulo - eliminadas")
        X_imputed = X_imputed[~mask_target_nulo]
        y = y[~mask_target_nulo]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    print(f"\nDatos preparados:")
    print(f"   Train: {len(X_train)} muestras ({100*len(X_train)/len(X_imputed):.0f}%)")
    print(f"   Test:  {len(X_test)} muestras ({100*len(X_test)/len(X_imputed):.0f}%)")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# ENTRENAMIENTO CON VALIDACION CRUZADA
# ============================================================================

def entrenar_con_cv(modelo: Pipeline, 
                   X_train: pd.DataFrame, 
                   y_train: pd.Series,
                   nombre: str = "Modelo") -> dict:
    """
    Entrena un modelo con validacion cruzada k-fold.
    
    Calcula metricas en el fold de validacion (no en entrenamiento) para
    evitar overfitting. Luego ajusta el modelo en TODO el set de entrenamiento
    para maxima capacidad predictiva en test.
    
    Args:
        modelo: Pipeline sklearn ya configurado
        X_train: Datos de entrenamiento
        y_train: Target de entrenamiento
        nombre: Nombre del modelo (para logging)
    
    Returns:
        dict: Metricas de CV para cada fold
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # Calcular CV con scoring negativo para que sea consistente
    cv_results = cross_validate(
        modelo, X_train, y_train,
        cv=kf,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2"
        },
        return_train_score=False,
        n_jobs=1
    )
    
    # Extraer y convertir metricas
    rmse_cv = np.sqrt(-cv_results["test_rmse"])
    mae_cv = -cv_results["test_mae"]
    r2_cv = cv_results["test_r2"]
    
    # Mostrar resultados
    print(f"\n  {nombre}:")
    print(f"    RMSE: {rmse_cv.mean():.4f} +/- {rmse_cv.std():.4f}")
    print(f"    MAE:  {mae_cv.mean():.4f} +/- {mae_cv.std():.4f}")
    print(f"    R2:   {r2_cv.mean():.4f} +/- {r2_cv.std():.4f}")
    
    # Entrenar en TODO el conjunto de entrenamiento
    modelo.fit(X_train, y_train)
    
    return {
        "modelo": nombre,
        "rmse_mean": rmse_cv.mean(),
        "rmse_std": rmse_cv.std(),
        "rmse_folds": rmse_cv,
        "mae_mean": mae_cv.mean(),
        "r2_mean": r2_cv.mean()
    }


# ============================================================================
# ENTRENAMIENTO COMPLETO
# ============================================================================

def entrenar_todos_modelos(X_train: pd.DataFrame, 
                          y_train: pd.Series) -> tuple:
    """
    Entrena TODOS los modelos (baseline + avanzados) con CV.
    
    Esta es la funcion principal para la fase de entrenamiento.
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Target de entrenamiento
    
    Returns:
        Tupla (modelos_entrenados, df_cv_results)
        - modelos_entrenados: dict {nombre: Pipeline.fit()}
        - df_cv_results: DataFrame con metricas de CV para comparacion
    """
    modelos_definidos = get_todos_los_modelos()
    resultados_cv = []
    modelos_entrenados = {}
    
    print(f"\n{'='*70}")
    print(f"Entrenando {len(modelos_definidos)} MODELOS CON {CV_FOLDS}-FOLD CV")
    print(f"{'='*70}")
    
    for nombre, modelo in modelos_definidos.items():
        metricas = entrenar_con_cv(modelo, X_train, y_train, nombre)
        resultados_cv.append({
            "modelo": nombre,
            "rmse_mean": metricas["rmse_mean"],
            "rmse_std": metricas["rmse_std"],
            "mae_mean": metricas["mae_mean"],
            "r2_mean": metricas["r2_mean"]
        })
        modelos_entrenados[nombre] = modelo
    
    # Crear DataFrame para facil comparacion
    df_resultados = pd.DataFrame(resultados_cv).sort_values("rmse_mean")
    
    print(f"\n{'='*70}")
    print(f"RANKING POR RMSE (Validacion Cruzada):")
    print(f"{'='*70}")
    for idx, row in df_resultados.iterrows():
        print(f"  {row['modelo']:20s} | RMSE: {row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f}")
    
    return modelos_entrenados, df_resultados


# ============================================================================
# PREDICCION EN TEST SET
# ============================================================================

def predecir_test(modelos_entrenados: dict,
                  X_test: pd.DataFrame) -> dict:
    """
    Realiza predicciones en test set con todos los modelos.
    
    Args:
        modelos_entrenados: Dict de modelos ya ajustados
        X_test: Datos de test
    
    Returns:
        dict: {nombre_modelo: y_pred}
    """
    predicciones = {}
    for nombre, modelo in modelos_entrenados.items():
        predicciones[nombre] = modelo.predict(X_test)
    return predicciones


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del modulo Models.py
    """
    from data_loader import load_and_prepare
    
    # Cargar datos
    X, y, loader = load_and_prepare()
    
    # Preparar split
    X_train, X_test, y_train, y_test = preparar_datos(X, y)
    
    # Entrenar todos los modelos
    modelos, cv_results = entrenar_todos_modelos(X_train, y_train)
    
    # Hacer predicciones
    predicciones = predecir_test(modelos, X_test)
    
    print(f"\nModelos listos para evaluacion con Evaluation.py")
