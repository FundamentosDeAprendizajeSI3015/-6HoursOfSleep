# Pipeline Supervisado: Predicción de Deserción Estudiantil

**Responsable:** Juanes  
**Rama:** `feature/supervised-models`  
**Versión:** 2.0 (18,612 registros mensuales)  
**Fecha:** Mayo 18, 2026

---

## 🎯 Objetivo General

Implementar, entrenar, comparar y evaluar **8 modelos de regresión supervisada** para predecir la **tasa de deserción estudiantil** (`outcome_tasa_desercion_snies`) usando:

- **18,612 registros mensuales** (33 departamentos × 47 años × 12 meses)
- **27 features** predictoras después de limpieza
- **Indicadores económicos** (PIB, IPC, desempleo)
- **Índices sintéticos** de vulnerabilidad
- **Variables educativas** del sistema SPADIES

---

## 📊 Datos

### Fuente Principal
```
data_simulada/processed/data_simulado_1980_2026.csv
├── Registros: 18,612 (MENSUAL)
├── Período: 1980-2026 (47 años)
├── Departamentos: 33 (Colombia)
├── Meses: 1-12 (datos mensuales)
└── Columnas: 34 (incluye target y técnicas)
```

### Limpieza Realizada

| Paso | Eliminado | Motivo | Registros |
|------|-----------|--------|-----------|
| Original | - | - | 18,612 |
| Data leakage | `spadies_td_anual_universitario` | Correlación 1.0 con target | 18,612 |
| Técnicas | `outcome_merge_pendiente` | Columna técnica inútil | 18,612 |
| **Limpio** | **2 columnas** | - | **18,612** |

### Características Finales

```
Variables Demográficas:
  ✓ codigo_departamento         (excluida del modelado)
  ✓ departamento               (excluida del modelado)
  ✓ anio                       (excluida del modelado)
  ✓ mes                        (excluida del modelado - NUEVO)

Variables Macroeconómicas (9):
  ✓ pib_total_miles_millones_cop
  ✓ pib_variacion_pct_anual_vs_anio_previo
  ✓ ipc_nacional_total_var_mensual_media
  ✓ ipc_nacional_total_var_mensual_mediana
  ✓ ipc_nacional_total_var_mensual_std
  ✓ ipc_nacional_educacion_var_mensual_media
  ✓ ipc_nacional_educacion_var_mensual_mediana
  ✓ ipc_nacional_educacion_var_mensual_std
  ✓ ipc_nacional_alimentos_var_mensual_media
  + 6 más (IPC transporte, capital)

Variables Educativas (6):
  ✓ total_admitidos
  ✓ total_matriculados
  ✓ ratio_matriculados_sobre_admitidos
  ✓ var_pct_matriculados_vs_anio_previo
  ✓ proxy_pib_miles_mm_cop_por_matriculado
  ✓ SPADIES_td_anual_[tecnologico, tyt, tecnico_profesional]

Variables de Mercado Laboral (3):
  ✓ geih_td_nacional_media_anual
  ✓ geih_to_nacional_media_anual
  ✓ geih_tgp_nacional_media_anual

TARGET (1):
  ✓ outcome_tasa_desercion_snies     (% de deserción)
```

**Total Features: 27** (después de excluir demográficas y técnicas)

---

## 🤖 Modelos Implementados

### Baseline (4 modelos lineales)

| # | Modelo | Parámetros | Interpretabilidad | Complejidad |
|---|--------|-----------|------------------|------------|
| 1 | **OLS** | Ninguno | ⭐⭐⭐⭐⭐ | Muy baja |
| 2 | **Ridge** | α=1.0 (L2) | ⭐⭐⭐⭐ | Baja |
| 3 | **Lasso** | α=0.1 (L1) | ⭐⭐⭐⭐ | Baja |
| 4 | **ElasticNet** | α=0.1, L1_ratio=0.5 | ⭐⭐⭐ | Baja |

**Utilidad:** Proporcionan puntos de referencia simples, interpretables y eficientes.

### Avanzados (4 modelos no-lineales)

| # | Modelo | Parámetros | Interpretabilidad | Complejidad |
|---|--------|-----------|------------------|------------|
| 5 | **Random Forest** | n_trees=100 | ⭐⭐ | Media-Alta |
| 6 | **Gradient Boosting** | n_trees=100, lr=0.1 | ⭐⭐ | Media-Alta |
| 7 | **SVR (RBF)** | C=1.0, γ=auto | ⭐ | Alta |
| 8 | **KNN** | k=5 | ⭐⭐⭐ | Media |

**Utilidad:** Capturan relaciones no-lineales complejas entre variables.

---

## 📈 Resultados Principales

### Validación Cruzada (5-fold)

```
================================================================================
RANKING POR RMSE (Validación Cruzada - 5 Folds)
================================================================================
  1. RandomForest         | RMSE: 0.7552 ± 0.0063 | R²: 0.9233 ± 0.0029
  2. KNN                  | RMSE: 0.7730 ± 0.0027 | R²: 0.9159 ± 0.0019
  3. GradientBoosting     | RMSE: 0.7815 ± 0.0046 | R²: 0.9121 ± 0.0031
  4. SVR                  | RMSE: 0.7856 ± 0.0036 | R²: 0.9102 ± 0.0027
  5. Ridge                | RMSE: 0.7886 ± 0.0028 | R²: 0.9088 ± 0.0021
  6. OLS                  | RMSE: 0.7886 ± 0.0028 | R²: 0.9088 ± 0.0021
  7. ElasticNet           | RMSE: 0.7934 ± 0.0028 | R²: 0.9066 ± 0.0022
  8. Lasso                | RMSE: 0.7983 ± 0.0023 | R²: 0.9043 ± 0.0024
================================================================================
```

### Test Set Final (20% de datos)

```
================================================================================
RANKING POR RMSE (Test Set - Evaluación Final)
================================================================================
  1. RandomForest       | RMSE: 0.5566 | MAE: 0.4334 | R²: 0.9284 | MAPE: 4.32%
  2. KNN                | RMSE: 0.5751 | MAE: 0.4224 | R²: 0.9235 | MAPE: 4.16%
  3. GradientBoosting   | RMSE: 0.6002 | MAE: 0.4809 | R²: 0.9167 | MAPE: 4.78%
  4. SVR                | RMSE: 0.6060 | MAE: 0.4784 | R²: 0.9151 | MAPE: 4.75%
  5. OLS                | RMSE: 0.6146 | MAE: 0.4924 | R²: 0.9126 | MAPE: 4.89%
  6. Ridge              | RMSE: 0.6147 | MAE: 0.4924 | R²: 0.9126 | MAPE: 4.89%
  7. ElasticNet         | RMSE: 0.6220 | MAE: 0.4991 | R²: 0.9105 | MAPE: 5.02%
  8. Lasso              | RMSE: 0.6308 | MAE: 0.5058 | R²: 0.9080 | MAPE: 5.10%
================================================================================
```

### 🏆 Modelo Ganador: **Random Forest**

```
┌────────────────────────────────────────────────────────────────┐
│                  RANDOM FOREST - GANADOR                       │
├────────────────────────────────────────────────────────────────┤
│ RMSE:                    0.5566 (puntos porcentuales)           │
│ MAE:                     0.4334 (error promedio)                │
│ R²:                      0.9284 (92.84% varianza explicada)     │
│ MAPE:                    4.32% (error porcentual medio)         │
│                                                                 │
│ Mejora vs Baseline (OLS):                                       │
│   - RMSE: 0.0580 puntos (9.4% mejor)                            │
│   - R²: +0.0158 (mejor ajuste)                                  │
│                                                                 │
│ Top 5 Features:                                                 │
│   1. spadies_td_anual_tecnico_profesional     (imp: 0.884)     │
│   2. spadies_td_anual_tecnologico             (imp: 0.874)     │
│   3. spadies_td_anual_tyt                     (imp: 0.864)     │
│   4. pib_total_miles_millones_cop             (imp: 0.459)     │
│   5. ipc_capital_total_var_mensual_media      (imp: 0.424)     │
└────────────────────────────────────────────────────────────────┘
```

---

## 📊 Métricas de Evaluación

| Métrica | Fórmula | Rango | Interpretación |
|---------|---------|-------|----------------|
| **RMSE** | √(MSE) | [0, ∞) | Error cuadrático medio - **Métrica principal** |
| **MAE** | Mean(\|y - ŷ\|) | [0, ∞) | Error absoluto promedio - Resistente a outliers |
| **R²** | 1 - (SS_res/SS_tot) | [0, 1] | Proporción de varianza explicada |
| **MAPE** | Mean(\|dy/y\|) × 100 | [0, ∞) | Error porcentual medio - Interpretable |

---

## 🔄 Protocolo de Validación

```
┌─────────────────────────────────────────────────────────────┐
│ DATOS ORIGINALES (18,612 registros mensuales)              │
├─────────────────────────────────────────────────────────────┤
│                   ↓ Limpieza                                │
├─────────────────────────────────────────────────────────────┤
│ DATOS LIMPIOS (18,612 registros - sin cambios)             │
├──────────────────────┬──────────────────────────────────────┤
│ (80% Train)          │ (20% Test)                           │
│ 14,889 muestras      │ 3,723 muestras                       │
│ ├─ Fold 1            │                                      │
│ ├─ Fold 2            │ [EVALUACIÓN FINAL]                   │
│ ├─ Fold 3 (5-Fold CV)│ Métricas de prueba                   │
│ ├─ Fold 4            │ RMSE, MAE, R², MAPE                 │
│ └─ Fold 5            │                                      │
└──────────────────────┴──────────────────────────────────────┘
```

**Configuración:**
- Split: 80% train / 20% test
- CV: 5-fold estratificado
- Random state: 42 (reproducibilidad)
- Imputación: Media (792 valores nulos encontrados)

---

## 📁 Estructura de Archivos

```
supervised/
│
├── 01_pipeline_completo.ipynb        ← Notebook interactivo (recomendado)
├── main.py                           ← Ejecutar: python main.py
├── config.py                         ← Configuración global
│
├── src/
│   ├── data_loader.py               ← Carga 18,612 registros mensuales
│   ├── Models.py                    ← 8 modelos de regresión
│   ├── Evaluation.py                ← Métricas y visualizaciones
│   └── utils.py                     ← Funciones auxiliares
│
├── reports/
│   ├── tabla_comparativa_modelos_supervisados.csv
│   └── eda_figures/
│       ├── supervisado_predicho_vs_real.png
│       ├── supervisado_residuales.png
│       ├── supervisado_comparacion_metricas.png
│       ├── supervisado_importancia_randomforest.png
│       └── supervisado_importancia_gradientboosting.png
│
├── results/
│   ├── pipeline_results.json        ← Resultados en JSON
│   ├── resumen_ejecutivo.json       ← Resumen ejecutivo
│   └── README.md                    ← Documentación de resultados
│
└── docs/
    ├── README_CONSOLIDATED.md       ← Este archivo (versión actualizada)
    ├── CONCLUSIONES.md              ← Análisis y conclusiones
    ├── INDEX.md                     ← Índice de documentación
    └── QUICK_REFERENCE.md           ← Referencia rápida
```

---

## 🚀 Uso Rápido

### Opción 1: Jupyter Notebook (Recomendado)
```bash
# Desde la carpeta supervised/
jupyter notebook 01_pipeline_completo.ipynb
```
✅ Interactivo  
✅ Visualizaciones integradas  
✅ Ejecutar celda por celda  

### Opción 2: Script Python Completo
```bash
# Desde la carpeta supervised/
python main.py
```
✅ Rápido  
✅ Automatizado  
✅ Genera todos los reportes  

### Opción 3: Test Simple
```bash
# Desde la carpeta supervised/
python test_pipeline_simple.py
```
✅ Verifica configuración  
✅ Cargas rápidas  
✅ Detecta problemas  

---

## 💻 Usar en Código Python

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("./src")))

# 1. Cargar datos (18,612 registros)
from data_loader import load_and_prepare
X, y, loader = load_and_prepare()
print(f"Datos: {X.shape[0]:,} muestras × {X.shape[1]} features")

# 2. Preparar split train/test
from Models import preparar_datos
X_train, X_test, y_train, y_test = preparar_datos(X, y)

# 3. Entrenar 8 modelos con CV 5-fold
from Models import entrenar_todos_modelos
modelos, cv_results = entrenar_todos_modelos(X_train, y_train)

# 4. Evaluar en test set
from Evaluation import evaluar_todos
metricas = evaluar_todos(modelos, X_test.values, y_test.values)

# 5. Obtener ganador
ganador = metricas.index[0]
print(f"Mejor modelo: {ganador}")
print(metricas.loc[ganador])

# 6. Visualizar predicciones
from Evaluation import plot_predicho_vs_real
plot_predicho_vs_real(modelos, X_test.values, y_test.values, guardar=True)
```

---

## 📊 Interpretación de Visualizaciones

### 1. Predicho vs Real
- **Diagonal roja:** Predicción perfecta
- **Puntos sobre la línea:** Modelo sobreestima
- **Puntos bajo la línea:** Modelo subestima
- **Dispersión:** Variabilidad del modelo

### 2. Residuales
- **Aleatorios alrededor de 0:** ✅ Buen modelo
- **Patrón de embudo:** ⚠️ Heteroscedasticidad
- **Puntos lejanos:** ⚠️ Outliers influyentes

### 3. Comparación de Métricas
- **Barras: Altura = RMSE**
- **Menor altura = Mejor modelo**
- **Errores mostrados:** CV vs Test

### 4. Feature Importance
- **Barras largas:** Variables clave para predicción
- **Random Forest:** Importancia por "mean decrease impurity"
- **Gradient Boosting:** Importancia por contribución predictiva

---

## 🎓 Metodología

### Data Quality Checks
✅ 18,612 registros sin cambios  
✅ 27 features seleccionadas (excluidas demográficas/técnicas)  
✅ 792 valores nulos imputados con media  
✅ 0 duplicados exactos  
✅ Data leakage eliminado  

### Cross-Validation Strategy
✅ 5-fold CV en train set  
✅ Métricas reportadas: media ± std  
✅ Test set virgen (20%) para evaluación final  
✅ Random state = 42 (reproducibilidad)  

### Model Selection
✅ 8 modelos diversos (baseline + avanzados)  
✅ Ranking por RMSE  
✅ Comparación de todas las métricas  
✅ Feature importance extraído  

---

## 🔍 Hallazgos Clave

### 1. **Random Forest es el mejor modelo**
- RMSE: 0.5566 (9.4% mejor que baseline)
- R²: 0.9284 (explica 92.84% de la varianza)
- Robusto a outliers, captura no-linealidades

### 2. **Variables SPADIES son predictoras clave**
- `spadies_td_anual_tecnico_profesional` (imp: 0.884)
- `spadies_td_anual_tecnologico` (imp: 0.874)
- `spadies_td_anual_tyt` (imp: 0.864)
- **Conclusión:** La deserción en otros programas predice bien la deserción SNIES

### 3. **Variables económicas tienen importancia secundaria**
- PIB: correlación 0.459
- IPC: correlación 0.424
- Desempleo: correlación 0.235
- **Conclusión:** Contexto macroeconómico influye pero no determina

### 4. **Datos mensuales mejoran predicción**
- Capturan variabilidad estacional
- 18,612 registros vs 1,551 anuales (12x más datos)
- Mejor generalización del modelo

---

## ⚙️ Configuración de Hiperparámetros

### Random Forest (Ganador)
```python
RandomForestRegressor(
    n_estimators=100,       # 100 árboles (mejor complejidad-precisión)
    max_depth=None,         # Profundidad ilimitada
    min_samples_split=2,    # Mínimo 2 muestras para split
    min_samples_leaf=1,     # Mínimo 1 muestra en hoja
    random_state=42,        # Reproducibilidad
    n_jobs=-1               # Usar todos los CPUs
)
```

### Gradient Boosting
```python
GradientBoostingRegressor(
    n_estimators=100,       # 100 boosts secuenciales
    learning_rate=0.1,      # Shrinkage rate
    max_depth=3,            # Profundidad limitada
    subsample=1.0,          # Usar todos los datos
    random_state=42,        # Reproducibilidad
)
```

### Otros Modelos
Ver `src/Models.py` para detalles completos de configuración.

---

## 📋 Checklist de Completitud

Versión 2.0 (18,612 registros mensuales):

- [x] Carga automatizada de **18,612 registros mensuales**
- [x] Limpieza de data leakage (`spadies_td_anual_universitario`)
- [x] **27 features** seleccionados (excluidas demográficas/técnicas)
- [x] 4 modelos baseline (OLS, Ridge, Lasso, ElasticNet)
- [x] 4 modelos avanzados (RF, GB, SVR, KNN)
- [x] Validación cruzada 5-fold estandarizada
- [x] Métricas múltiples (RMSE, MAE, R², MAPE)
- [x] Split train/test 80/20 reproducible
- [x] Visualizaciones comprehensivas (scatter, residuales, comparación, feature importance)
- [x] Exportación de resultados (CSV, JSON)
- [x] Documentación completa en Markdown
- [x] Código comentado y modular

---

## 🚨 Troubleshooting

### Error: "No se encontró data_simulado_1980_2026.csv"
```python
# Solución: Especificar ruta absoluta
from data_loader import DataLoader
loader = DataLoader(r"c:\...\data_simulada\processed\data_simulado_1980_2026.csv")
```

### Menos features que esperado
```python
# Verificar exclusiones en prepare_features()
# Se excluyen: departamento, anio, mes, codigo_departamento, outcome_merge_pendiente
# + columnas con >50% nulos
```

### Resultados diferentes en segunda ejecución
```python
# Verificar random_state
# Debe ser 42 en split y en todos los modelos
# Diferentes semillas generan modelos diferentes (normal)
```

---

## 📚 Referencias Teóricas

### Random Forests
- Bagging + múltiples árboles de decisión
- Robusto a outliers y multicolinealidad
- Captura interacciones no-lineales
- Feature importance bien definida

### Gradient Boosting
- Boosting secuencial: cada árbol corrige errores del anterior
- Mejor en muchos benchmark competitions
- Mayor riesgo de overfitting que RF
- Learning rate controla velocidad de aprendizaje

### Cross-Validation
- Estima performance real evitando overfitting
- 5-fold: balance entre sesgo y varianza
- Reporta media ± std para estabilidad

### RMSE vs MAE
- RMSE: Penaliza errores grandes (recomendado para outliers)
- MAE: Error promedio en unidades originales (más interpretable)

---

## 🔗 Documentación Relacionada

- **[CONCLUSIONES.md](CONCLUSIONES.md)** — Análisis detallado de resultados
- **[INDEX.md](INDEX.md)** — Índice completo de documentación
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — Referencia rápida de comandos
- **[../results/README.md](../results/README.md)** — Documentación de resultados JSON

---

## 👤 Información del Proyecto

**Proyecto:** Predicción de Tasa de Deserción Estudiantil en Colombia  
**Rama:** `feature/supervised-models`  
**Responsable:** Juanes  
**Versión:** 2.0 (18,612 registros mensuales)  
**Última actualización:** Mayo 18, 2026  
**Estado:** ✅ Completado

---

## 📞 Contacto y Soporte

Para preguntas o problemas:
1. Revisar [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. Consultar [CONCLUSIONES.md](CONCLUSIONES.md)
3. Ver logs en `supervised/results/pipeline_results.json`
4. Ejecutar `python test_pipeline_simple.py` para diagnóstico

---

**[↑ Volver al inicio](#pipeline-supervisado-predicción-de-deserción-estudiantil)**
