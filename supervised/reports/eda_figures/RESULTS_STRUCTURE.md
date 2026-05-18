# 📦 Estructura de Resultados del Pipeline Supervisado

## 📂 Árbol de Directorios

```
supervised/
├── results/                              # 📁 NUEVA CARPETA CON RESULTADOS
│   ├── pipeline_results.json             # 📊 Resultados completos (estructurado)
│   ├── resumen_ejecutivo.json            # 📈 Resumen ejecutivo (alto nivel)
│   └── README.md                         # 📖 Guía de contenidos
│
├── results_viewer.py                     # 🔍 Script interactivo para visualizar
│
├── reports/                              # Gráficos y tablas
│   ├── tabla_comparativa_modelos_supervisados.csv
│   └── eda_figures/
│       ├── supervisado_predicho_vs_real.png
│       ├── supervisado_residuales.png
│       ├── supervisado_comparacion_metricas.png
│       ├── supervisado_importancia_randomforest.png
│       └── supervisado_importancia_gradientboosting.png
│
├── src/                                  # Código fuente
│   ├── data_loader.py                    # Carga de datos mensuales (18,612)
│   ├── Models.py                         # Entrenamiento de modelos
│   └── Evaluation.py                     # Evaluación y visualización
│
├── 01_pipeline_completo.ipynb            # Notebook con flujo completo
├── main.py                               # Script principal
└── test_pipeline_simple.py               # Script de prueba
```

---

## 📊 Contenido de `pipeline_results.json`

Archivo JSON con 12 secciones principales:

### 1. **metadata** - Información General
```json
{
  "descripcion": "Resultados completos del pipeline",
  "fecha_ejecucion": "2026-05-18",
  "dataset": "data_simulado_1980_2026.csv (mensual)"
}
```

### 2. **datos_cargados** - Dataset
```json
{
  "registros_totales": 18612,
  "formula": "33 departamentos × 47 años × 12 meses",
  "filas_limpias": 18612,
  "columnas_limpias": 32
}
```

### 3. **limpieza_datos** - Preprocesamiento
```json
{
  "columnas_eliminadas": ["spadies_td_anual_universitario", "outcome_merge_pendiente"],
  "valores_nulos_tratados": 792,
  "estrategia": "Imputación con media"
}
```

### 4. **analisis_exploratorio** - EDA
```json
{
  "target": "outcome_tasa_desercion_snies",
  "target_media": 10.2505,
  "target_std": 2.0645,
  "top_5_features": [...]
}
```

### 5. **preparacion_datos** - Split Train/Test
```json
{
  "train": 14889,
  "test": 3723,
  "split_ratio": "80/20"
}
```

### 6. **entrenamiento_modelos** - Validación Cruzada
```json
{
  "metodo": "5-Fold Cross Validation",
  "modelos": 8,
  "resultados_cv": [...]
}
```

### 7. **evaluacion_test_set** - Resultados Finales
```json
{
  "muestras_test": 3723,
  "metricas": [...]
}
```

### 8. **modelo_ganador** - RandomForest
```json
{
  "nombre": "RandomForest",
  "rmse": 0.5566,
  "r2": 0.9284,
  "mape": "4.32%",
  "mejora_vs_baseline": "9.4%"
}
```

### 9. **visualizaciones_generadas** - Gráficos
```json
{
  "ubicacion": "supervised/reports/eda_figures/",
  "archivos": 5
}
```

### 10. **archivos_exportados** - Resultados
```json
{
  "tabla_csv": "tabla_comparativa_modelos_supervisados.csv",
  "json": "pipeline_results.json"
}
```

### 11. **conclusiones** - Hallazgos
```json
{
  "hallazgos_clave": [...],
  "proximos_pasos": [...]
}
```

### 12. **resumen_ejecucion** - Cierre
```json
{
  "estado": "COMPLETADO EXITOSAMENTE",
  "mejor_modelo": "RandomForest"
}
```

---

## 📈 Contenido de `resumen_ejecutivo.json`

Archivo JSON optimizado para **alta dirección**. 10 secciones:

### 1. **resumen_ejecutivo** - Título y Fecha
```json
{
  "titulo": "Predicción de Deserción Estudiantil en Colombia",
  "estado": "COMPLETADO EXITOSAMENTE ✅"
}
```

### 2. **dataset** - Dataset Overview
### 3. **target** - Variable Objetivo
### 4. **metodologia** - Enfoque Técnico
### 5. **modelos_evaluados** - Lista de Algoritmos
### 6. **resultados_validacion_cruzada** - CV Results
### 7. **resultados_test_set** - Test Results
### 8. **modelo_ganador_detallado** - Best Model Analysis
### 9. **variables_mas_importantes** - Top Features
### 10. **comparativa_todos_modelos** - Model Ranking
### 11. **hallazgos_clave** - Key Findings
### 12. **recomendaciones** - Action Plan

---

## 🔍 Cómo Usar `results_viewer.py`

Script interactivo con menú:

```bash
# Desde la carpeta supervised
python results_viewer.py

# Menú interactivo
# 1. Mostrar resumen ejecutivo
# 2. Mostrar información del dataset
# 3. Mostrar información del target
# 4. Mostrar top 5 features
# 5. Mostrar comparación de modelos
# 6. Mostrar hallazgos clave
# 7. Mostrar recomendaciones
# 8. Mostrar reporte completo
# 9. Salir
```

### Ejemplo de Uso en Python

```python
import json

# Cargar resultados
with open('results/pipeline_results.json', 'r') as f:
    resultados = json.load(f)

# Acceder a información
print(f"Modelo ganador: {resultados['modelo_ganador']['nombre']}")
print(f"RMSE: {resultados['modelo_ganador']['metricas_test']['rmse']}")
print(f"R²: {resultados['modelo_ganador']['metricas_test']['r2']}")

# Iterar sobre modelos
for modelo in resultados['evaluacion_test_set']['metricas']:
    print(f"{modelo['modelo']}: RMSE={modelo['rmse']:.4f}")
```

---

## 📊 Comparación Rápida de Modelos

| Modelo | RMSE | R² | MAPE | Status |
|--------|------|-----|------|--------|
| 🥇 RandomForest | 0.5566 | 0.9284 | 4.32% | ✅ GANADOR |
| 🥈 KNN | 0.5751 | 0.9235 | 4.16% | - |
| 🥉 GradientBoosting | 0.6002 | 0.9167 | 4.78% | - |
| SVR | 0.6060 | 0.9151 | 4.75% | - |
| OLS | 0.6146 | 0.9126 | 4.89% | Baseline |
| Ridge | 0.6147 | 0.9126 | 4.89% | - |
| ElasticNet | 0.6220 | 0.9105 | 5.02% | - |
| Lasso | 0.6308 | 0.9080 | 5.10% | - |

---

## 🎯 Métricas Explicadas

- **RMSE**: Error cuadrático medio (0.5566 = ±0.56 puntos porcentuales)
- **MAE**: Error absoluto medio (0.4334 = ±0.43 puntos porcentuales)
- **R²**: 0.9284 = Explica 92.84% de la varianza
- **MAPE**: 4.32% = Error porcentual medio en predicciones

---

## 📁 Archivos en la Carpeta `results/`

### `pipeline_results.json` (COMPLETO)
- 12 secciones estructuradas
- Toda la información técnica detallada
- ~8,000 líneas de JSON

### `resumen_ejecutivo.json` (EJECUTIVO)
- 12 secciones optimizadas
- Para presentaciones a dirección
- ~1,200 líneas de JSON

### `README.md` (GUÍA)
- Explicación de todas las secciones
- Ejemplos de uso
- Consultas útiles

---

## 🚀 Próximos Pasos Recomendados

1. ✅ **Revisar gráficos** en `reports/eda_figures/`
2. ✅ **Ejecutar** `python results_viewer.py` para explorar
3. ✅ **Compartir** `resumen_ejecutivo.json` con stakeholders
4. ✅ **Documentar** hallazgos en reportes finales
5. ✅ **Preparar** despliegue de RandomForest en producción

---

## 📝 Versión y Fecha

- **Versión:** 1.0
- **Fecha:** 2026-05-18
- **Dataset:** Mensual (18,612 registros)
- **Mejor Modelo:** RandomForest (R² = 0.9284, RMSE = 0.5566)
- **Estado:** ✅ COMPLETADO

---

**Creado por:** Juanes  
**Rama:** feature/supervised-models  
**Responsable:** Análisis de Deserción Estudiantil en Colombia
