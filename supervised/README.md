# feature/supervised-models — Juanes

> Rama a cargo de: **Juanes**
> Objetivo: Implementar, comparar y evaluar modelos de aprendizaje supervisado para predecir la tasa de deserción estudiantil.

---

## 📁 Archivos de esta rama

```
notebooks/03_modelos_supervisados/
│
├── 01_baseline_regresion.ipynb      ← Modelos lineales (regresión)
├── 02_modelos_avanzados.ipynb       ← Random Forest, SVR, Gradient Boosting
├── 03_comparacion_modelos.ipynb     ← Tabla comparativa de todos los modelos
├── 04_interpretabilidad.ipynb       ← Feature importance, SHAP
└── README.md                        ← Este archivo

src/
├── models.py                        ← Implementación de modelos
└── evaluation.py                    ← Métricas y visualización de resultados
```

---

## Responsabilidades

### Variables

**Variable objetivo (regresión):**

* `desercion` → tasa continua (%)

**Features:**

* Todos los indicadores económicos + índice construido por Isabela
* `indice_vulnerabilidad_pca` (feature clave)
* Variables derivadas de Santi

---

### 1. Modelos baseline (`01_baseline_regresion.ipynb`)

Implementar como punto de comparación:

| Modelo            | Justificación                                |
| ----------------- | --------------------------------------------- |
| Regresión Lineal | Baseline interpretable                        |
| Ridge             | Regularización L2 (maneja multicolinealidad) |
| Lasso             | Regularización L1 (selección de variables)  |
| ElasticNet        | Combinación Ridge + Lasso                    |

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

modelos_baseline = {
    "OLS":        LinearRegression(),
    "Ridge":      Ridge(alpha=1.0),
    "Lasso":      Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}
```

### 2. Modelos avanzados (`02_modelos_avanzados.ipynb`)

| Modelo            | Hiperparámetros clave      |
| ----------------- | --------------------------- |
| Random Forest     | n_estimators, max_depth     |
| Gradient Boosting | n_estimators, learning_rate |
| SVR               | C, kernel, epsilon          |
| KNN Regressor     | n_neighbors                 |

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

modelos_avanzados = {
    "RF":  RandomForestRegressor(n_estimators=100, random_state=42),
    "GBM": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
    "SVR": SVR(kernel="rbf", C=1.0),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}
```

### 3. Comparación de modelos (`03_comparacion_modelos.ipynb`)

**Protocolo de evaluación estándar:**

* Split: 80% train / 20% test (`random_state=42`)
* Validación cruzada: 5-fold CV en conjunto de entrenamiento
* Reportar tanto CV como test final

---

## 📏 Métricas de evaluación

> **Estas métricas son el estándar del pipeline. Todos los modelos deben reportar exactamente estas.**

### Métricas de regresión

| Métrica       | Fórmula          | Interpretación                           |
| -------------- | ----------------- | ----------------------------------------- |
| **RMSE** | √(MSE)           | Error en unidades de la variable objetivo |
| **MAE**  | mean(             | y - ŷ                                    |
| **R²**  | 1 - SS_res/SS_tot | Varianza explicada (0 a 1)                |
| **MAPE** | mean(             | y-ŷ                                      |

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluar_modelo(y_true, y_pred, nombre="Modelo"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n📊 {nombre}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")
    print(f"   R²   : {r2:.4f}")
    print(f"   MAPE : {mape:.2f}%")

    return {"modelo": nombre, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
```

### Tabla comparativa final (formato estándar)

| Modelo        | RMSE | MAE | R² | MAPE | CV_RMSE_mean | CV_RMSE_std |
| ------------- | ---- | --- | --- | ---- | ------------ | ----------- |
| OLS           | ...  | ... | ... | ...  | ...          | ...         |
| Ridge         | ...  | ... | ... | ...  | ...          | ...         |
| Random Forest | ...  | ... | ... | ...  | ...          | ...         |
| GBM           | ...  | ... | ... | ...  | ...          | ...         |

### 4. Interpretabilidad (`04_interpretabilidad.ipynb`)

```python
# Feature importance para modelos de árbol
importancias = pd.Series(
    rf.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

# SHAP (opcional pero recomendado)
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

**Preguntas a responder:**

* ¿Cuáles son las 3 variables más importantes para predecir deserción?
* ¿El índice construido por Isabela mejora el modelo?
* ¿Hay diferencias por región/departamento?

---

## Visualizaciones requeridas

* [ ] Residuales vs. valores ajustados (diagnóstico)
* [ ] Gráfico de importancia de variables
* [ ] Scatter: predicho vs. real (todos los modelos)
* [ ] Tabla comparativa de métricas
* [ ] Curva de aprendizaje (train vs. validation score)

---

## ✅ Checklist antes de hacer merge

* [ ] Todos los modelos usan el mismo split (random_state=42)
* [ ] Tabla comparativa con RMSE, MAE, R², MAPE generada
* [ ] Al menos un modelo con validación cruzada 5-fold
* [ ] `models.py` y `evaluation.py` funcionan al importar
* [ ] El mejor modelo identificado y justificado
* [ ] Visualizaciones exportadas a `reports/figures/`

---

## 🔗 Dependencias

* **Requiere:** `data/processed/dataset_con_indice.csv` de Isabela
* **Provee:** Tabla final de métricas y modelo guardado en `reports/`
