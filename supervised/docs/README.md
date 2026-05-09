# feature/supervised-models — Modelos Supervisados

> **Responsable:** Juanes  
> **Rama:** `feature/supervised-models`  
> **Objetivo:** Implementar, entrenar, comparar y evaluar modelos supervisados para predicción de tasa de deserción estudiantil.

---

## Descripcion General

Este módulo implementa un **pipeline completo de modelado supervisado** con:

- [OK] **Carga y limpieza de datos** automática
- [OK] **8 modelos diferentes**: 4 baseline + 4 avanzados
- [OK] **Validación cruzada 5-fold** estandarizada
- [OK] **Evaluación en test set** con métricas estándar
- [OK] **Visualizaciones de diagnóstico** comprehensivas
- [OK] **Feature importance** para modelos de árbol
- [OK] **Reportes y exportación** de resultados

---

## Estructura de Archivos

```
supervised/
├── main.py                              ← Punto de entrada (ejecutar este)
│
├── src/
│   ├── data_loader.py                  ← Carga y limpieza de datos
│   ├── Models.py                       ← Definición y entrenamiento de 8 modelos
│   ├── Evaluation.py                   ← Métricas y visualizaciones
│   └── utils.py                        ← Funciones utilitarias
│
├── reports/                            ← Salidas generadas
│   ├── eda_figures/
│   │   ├── supervisado_predicho_vs_real.png
│   │   ├── supervisado_residuales.png
│   │   ├── supervisado_comparacion_metricas.png
│   │   ├── supervisado_importancia_RandomForest.png
│   │   └── supervisado_importancia_GradientBoosting.png
│   ├── tabla_comparativa_modelos_supervisados.csv
│   └── informe_supervisado.txt
│
└── README.md                           ← Este archivo
```

---

## Uso Rapido

### Ejecución Completa

```bash
# Desde la carpeta supervised/
python main.py
```

### Ejecución en Jupyter Notebook

```bash
# Desde la carpeta supervised/
jupyter notebook 01_pipeline_completo.ipynb
```

Este comando ejecuta el pipeline completo:
1. Carga datos desde `data/final/dataset_con_indice.csv`
2. Realiza limpieza y preparación
3. Entrena 8 modelos con validación cruzada
4. Evalúa en test set
5. Genera visualizaciones y reportes

---

## [DATOS] Modelos Implementados

### Baseline (4 modelos)

| Modelo | Parámetros | Justificación |
|--------|-----------|---------------|
| **OLS** | Ninguno | Baseline sin regularización, máximo interpretable |
| **Ridge** | α=1.0 | Regularización L2, maneja multicolinealidad |
| **Lasso** | α=0.1 | Regularización L1, selección automática de variables |
| **ElasticNet** | α=0.1, l1_ratio=0.5 | Combinación L1+L2, mejor en muchos casos |

**Utilidad:** Proporcionan puntos de referencia simples e interpretables.

### Avanzados (4 modelos)

| Modelo | Parámetros | Justificación |
|--------|-----------|---------------|
| **Random Forest** | n_estimators=100 | Ensemble robusto a outliers |
| **Gradient Boosting** | n_estimators=100, lr=0.1 | Boosting secuencial, típicamente superior |
| **SVR (RBF)** | C=1.0, ε=0.1 | Kernel no-lineal, bueno en espacios altos |
| **KNN** | n_neighbors=5 | Basado en instancias, captura patrones locales |

**Utilidad:** Capturan relaciones no-lineales complejas entre variables.

---

##  Métricas de Evaluación

Todos los modelos son evaluados con **4 métricas estándar**:

| Métrica | Fórmula | Interpretación | Unidad |
|---------|---------|----------------|--------|
| **RMSE** | √(MSE) | Error cuadrático medio | % |
| **MAE** | Mean(\|y - ŷ\|) | Error promedio absoluto | % |
| **R²** | 1 - (SS_res/SS_tot) | Varianza explicada | [0, 1] |
| **MAPE** | Mean(\|Δy/y\|) × 100 | Error porcentual medio | % |

**Métrica principal:** RMSE (usado para ranking final)

---

## [PROCESANDO] Protocolo de Validación

```
┌─────────────────────────────────────────────────────┐
│ DATOS COMPLETOS (66 obs, 2 años × 33 depart.)     │
└──────────┬──────────────────────────────────────────┘
           │ Eliminar nulos
           ↓
┌──────────────────────────┐
│ DATOS LIMPIOS (50-55 obs)│
└──────────┬───────────────┘
           │ Split 80/20
       ┌───┴───┐
       ↓       ↓
   TRAIN   TEST
   (40)    (10)
    │       │
    │   [EVALUACIÓN FINAL]
    │
    ↓
  CV 5-fold
  (métricas reportadas)
```

**Configuración:**
- Split: 80% train / 20% test
- Random state: 42 (reproducibilidad)
- CV: 5-fold, shuffle=True

---

## [RANKING] Ejemplo de Salida

```
=====================================================================
[DATOS] RANKING POR RMSE (Validación Cruzada)
=====================================================================
  GradientBoosting     | RMSE: 3.1234 ± 0.5678
  RandomForest         | RMSE: 3.4567 ± 0.6789
  SVR                  | RMSE: 4.1234 ± 0.7890
  KNN                  | RMSE: 4.5678 ± 0.8901
  Ridge                | RMSE: 5.1234 ± 0.9012
  ElasticNet           | RMSE: 5.3456 ± 0.9123
  Lasso                | RMSE: 5.5678 ± 0.9234
  OLS                  | RMSE: 5.6789 ± 0.9345

=====================================================================
[LISTA] TABLA COMPARATIVA FINAL (Test Set)
=====================================================================
              RMSE      MAE      R2    MAPE
modelo
GradientBoosting  3.0521   2.4567  0.7234  12.34%
RandomForest      3.2143   2.6789  0.7012  13.45%
SVR               4.0876   3.1234  0.6789  14.56%
...
```

---

##  Usar el Módulo en Código

```python
from src.data_loader import load_and_prepare
from src.models import preparar_datos, entrenar_todos_modelos
from src.evaluation import evaluar_todos, plot_predicho_vs_real

# 1. Cargar datos
X, y, loader = load_and_prepare()

# 2. Preparar split
X_train, X_test, y_train, y_test = preparar_datos(X, y)

# 3. Entrenar
modelos, cv_results = entrenar_todos_modelos(X_train, y_train)

# 4. Evaluar
metricas = evaluar_todos(modelos, X_test.values, y_test.values)

# 5. Visualizar
plot_predicho_vs_real(modelos, X_test.values, y_test.values)

# Ver modelo ganador
print(f"Mejor: {metricas.index[0]}")
```

---

## [CONFIGURACION] Configuración de Hiperparámetros

Los hiperparámetros pueden ajustarse en `src/models.py`:

```python
# Modelos baseline
Ridge(alpha=1.0)              # Aumentar para mayor regularización
Lasso(alpha=0.1)              # Aumentar para más sparsidad

# Modelos avanzados
RandomForestRegressor(
    n_estimators=100,         # Más árboles → mejor pero más lento
    max_depth=None,           # Limitar para evitar overfitting
    min_samples_split=5       # Mayor → árboles más simples
)

GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,        # Menor → entrenamiento más lento pero mejor
    max_depth=3               # Profundidad de cada árbol
)
```

---

## [DATOS] Interpretación de Visualizaciones

### Predicho vs. Real
- **Puntos sobre diagonal roja** → Predicción correcta
- **Dispersión grande** → Modelo con alta varianza
- **Sesgo sistemático** → Error consistente en dirección

### Residuales
- **Aleatorios alrededor de 0** → Buen modelo
- **Patrón en embudo** → Heteroscedasticidad
- **Puntos alejados** → Outliers problemáticos

### Importancia de Variables
- **Barras largas** → Variables clave para predicción
- **Top features** → Enfocarse en estas para interpretabilidad

---

##  Variable Objetivo y Features

### Target
```
outcome_tasa_desercion_snies    Tasa de deserción estudiantil (%)
```

### Features Clave
```
Macroeconómicas:
  • geih_td_nacional_media_anual          Tasa de desempleo nacional
  • ipc_nacional_total_var_mensual_media  Inflación nacional
  • pib_variacion_pct_anual_vs_anio_previo  Crecimiento PIB

Índices (Construcción Previa):
  • indice_vulnerabilidad_pca             PCA (44.8% varianza)
  • indice_vulnerabilidad_teorico         Teórico (literatura)

Educativas:
  • spadies_td_anual_tecnologico          Deserción técnica
  • total_matriculados                    Acceso a educación
  • proxy_pib_miles_mm_cop_por_matriculado  PIB por estudiante
```

---

## [OK] Checklist de Completitud

- [x] Carga automática de datos
- [x] Limpieza de data leakage conocido
- [x] 4 modelos baseline
- [x] 4 modelos avanzados
- [x] Validación cruzada 5-fold
- [x] Métricas estándar (RMSE, MAE, R², MAPE)
- [x] Split train/test 80/20
- [x] Visualizaciones (scatter, residuales, comparación)
- [x] Feature importance
- [x] Exportación de resultados CSV
- [x] Documentación completa
- [x] Código bien comentado

---

##  Troubleshooting

### Error: "No se encontró dataset_con_indice.csv"
```python
# Especificar ruta manualmente
loader = DataLoader("../../data/final/dataset_con_indice.csv")
```

### Pocos datos para CV (< 30 muestras)
- Normal en datasets pequeños
- Usar k-fold bajo (k=3) si es necesario
- Considerar leave-one-out CV

### Modelo con R² negativo
- Indica PEOR que baseline
- Revisar features y datos
- Considerar transformaciones

---

##  Referencias Teóricas

### Validación Cruzada
Estima performance real mediante folds múltiples, evita overfitting.

### RMSE vs MAE
- RMSE penaliza más errores grandes
- MAE es más resistente a outliers

### Regularización (Ridge/Lasso)
- Ridge: Reduce magnitud coeficientes (L2)
- Lasso: Selecciona variables importantes (L1)

### Ensemble Methods
- Random Forest: Paralelo, robusto
- Gradient Boosting: Secuencial, típicamente mejor

---

## [NOTA] Notas Finales

- **Reproducibilidad:** Random state = 42 en todos los splits
- **Datos limitados:** 50-55 muestras finales después de limpieza
- **Data leakage:** Ya eliminado en EDA (spadies_td_anual_universitario)
- **Multicolinealidad:** Índices sintéticos capturan información económica
- **Siguiente paso:** Tuning de hiperparámetros en modelo ganador

---

**Actualizado:** Mayo 5, 2026  
**Responsable:** Juanes  
**Rama:** `feature/supervised-models`
