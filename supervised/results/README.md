# 📊 Resultados del Pipeline Supervisado

Esta carpeta contiene todos los resultados, métricas y análisis del pipeline de modelado supervisado para la predicción de deserción estudiantil en Colombia.

## 📁 Estructura

```
results/
├── pipeline_results.json          # Resultados completos en formato JSON
├── README.md                      # Este archivo (guía de contenidos)
```

## 📄 Archivo Principal: `pipeline_results.json`

Este archivo JSON contiene toda la información del pipeline organizada en secciones:

### 🎯 Secciones Principales

#### 1. **metadata**
   - Información general sobre la ejecución
   - Dataset utilizado: `data_simulado_1980_2026.csv` (datos mensuales)
   - Fecha de ejecución: 2026-05-18

#### 2. **datos_cargados**
   - **Descripción:** Información sobre el dataset procesado
   - **Registros:** 18,612 (mensual)
   - **Dimensión:** 33 departamentos × 47 años (1980-2026) × 12 meses
   - **Columnas originales:** 34 → **Columnas limpias:** 32

#### 3. **limpieza_datos**
   - Columnas eliminadas por data leakage y técnicas
   - Estrategia de manejo de valores nulos
   - Total de registros después de limpieza

#### 4. **analisis_exploratorio**
   - **Target:** `outcome_tasa_desercion_snies`
     - Media: 10.25%
     - Desviación estándar: 2.06%
   - **Top 5 features más correlacionados:**
     1. spadies_td_anual_tecnico_profesional (0.8839) ← Fuerte
     2. spadies_td_anual_tecnologico (0.8736)
     3. spadies_td_anual_tyt (0.8635)
     4. pib_total_miles_millones_cop (0.4590)
     5. ipc_capital_total_var_mensual_media (0.4241)

#### 5. **preparacion_datos**
   - Split: **80% Training (14,889) / 20% Test (3,723)**
   - Features utilizadas: 27

#### 6. **entrenamiento_modelos**
   - **Método:** 5-Fold Cross Validation
   - **Modelos:** 8 algoritmos evaluados
   - **Resultados CV:**
     - 🥇 RandomForest: RMSE = 0.7552
     - 🥈 KNN: RMSE = 0.7730
     - 🥉 GradientBoosting: RMSE = 0.7815

#### 7. **evaluacion_test_set**
   - Evaluación final en 20% de datos de prueba (3,723 muestras)
   - Resultados de todos los 8 modelos

#### 8. **modelo_ganador: RandomForest**
   - **RMSE:** 0.5566
   - **MAE:** 0.4334
   - **R²:** 0.9284 (92.84% varianza explicada)
   - **MAPE:** 4.32%
   - **Mejora vs OLS:** 9.4%

#### 9. **visualizaciones_generadas**
   - Ubicación: `supervised/reports/eda_figures/`
   - 5 gráficos generados:
     - Predicho vs Real
     - Análisis de Residuales
     - Comparación de Métricas
     - Feature Importance RandomForest
     - Feature Importance GradientBoosting

#### 10. **conclusiones**
   - Hallazgos clave
   - Próximos pasos recomendados
   - Consideraciones técnicas

## 📊 Métricas Explicadas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **RMSE** | √(Σ(y_pred - y_real)² / n) | Promedio del error (en unidades del target) |
| **MAE** | Σ\|y_pred - y_real\| / n | Error absoluto medio |
| **R²** | 1 - (SS_res / SS_tot) | Porcentaje de varianza explicada (0-1) |
| **MAPE** | (Σ\|error\| / \|y_real\|) × 100 | Error porcentual medio |

## 🎯 Recomendaciones

### ✅ Modelo en Producción
- **Usar:** RandomForest
- **Razón:** Mejor RMSE (0.5566) y R² (0.9284)
- **Error esperado:** ~4.32% en nuevas predicciones

### 🔄 Próximos Pasos
1. Tuning de hiperparámetros de RandomForest
2. Crear ensemble con top 3 modelos
3. Análisis de SHAP values para interpretabilidad
4. Validación estratificada por departamento
5. Monitoreo con nuevos datos mensuales

## 📈 Comparación de Modelos (Test Set)

```
Ranking Final (por RMSE):
1. RandomForest      | RMSE: 0.5566 | R²: 0.9284 ⭐
2. KNN               | RMSE: 0.5751 | R²: 0.9235
3. GradientBoosting  | RMSE: 0.6002 | R²: 0.9167
4. SVR               | RMSE: 0.6060 | R²: 0.9151
5. OLS (Baseline)    | RMSE: 0.6146 | R²: 0.9126
6. Ridge             | RMSE: 0.6147 | R²: 0.9126
7. ElasticNet        | RMSE: 0.6220 | R²: 0.9105
8. Lasso             | RMSE: 0.6308 | R²: 0.9080
```

## 🔍 Cómo Usar Este JSON

### Python
```python
import json

with open('results/pipeline_results.json', 'r') as f:
    resultados = json.load(f)

# Acceder a información específica
print(resultados['modelo_ganador']['nombre'])  # RandomForest
print(resultados['modelo_ganador']['metricas_test']['r2'])  # 0.9284
```

### Consultas Útiles
```python
# Top features correlacionados
top_features = resultados['analisis_exploratorio']['top_5_features_correlacionados']

# Comparación de modelos
modelos_test = resultados['evaluacion_test_set']['metricas']

# Próximos pasos
pasos = resultados['conclusiones']['proximos_pasos']
```

## 📝 Notas Técnicas

- **Dataset:** 18,612 registros mensuales (no agregado)
- **Período:** 1980-2026 (47 años)
- **Cobertura espacial:** 33 departamentos de Colombia
- **Granularidad:** Mensual (12 meses × años × departamentos)
- **Variable objetivo:** Tasa de deserción SNIES
- **Random state:** 42 (para reproducibilidad)

## 🔗 Archivos Relacionados

- Gráficos: `supervised/reports/eda_figures/supervisado_*.png`
- Tabla CSV: `supervised/reports/tabla_comparativa_modelos_supervisados.csv`
- Notebook: `supervised/01_pipeline_completo.ipynb`
- Script principal: `supervised/main.py`

---

**Última actualización:** 2026-05-18  
**Estado:** ✅ Completado  
**Responsable:** Juanes
