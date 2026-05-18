# Pipeline Supervisado: Predicción de Deserción Estudiantil

**⚠️ ACTUALIZACIÓN (Mayo 18, 2026):** Este archivo ha sido consolidado en **[README_CONSOLIDATED.md](README_CONSOLIDATED.md)** con información actualizada sobre los **18,612 registros mensuales**.

**Responsable:** Juanes  
**Rama:** `feature/supervised-models`  
**Versión:** 2.0 (18,612 registros mensuales)  
**Última actualización:** Mayo 18, 2026

---

## ⚡ Resumen Ejecutivo

**Dataset:** 18,612 registros mensuales (33 departamentos × 47 años × 12 meses)  
**Features:** 27 variables predictoras (después de limpieza)  
**Modelos:** 8 modelos (4 baseline + 4 avanzados)  
**Ganador:** Random Forest (RMSE: 0.5566, R²: 0.9284)  
**Mejora:** 9.4% vs baseline OLS  

### Resultados Principales

| Modelo | RMSE | MAE | R² | MAPE |
|--------|------|-----|-----|------|
| 🏆 **Random Forest** | **0.5566** | **0.4334** | **0.9284** | **4.32%** |
| KNN | 0.5751 | 0.4224 | 0.9235 | 4.16% |
| Gradient Boosting | 0.6002 | 0.4809 | 0.9167 | 4.78% |
| OLS (Baseline) | 0.6146 | 0.4924 | 0.9126 | 4.89% |

---

## Descripción General

Este módulo implementa un pipeline completo de modelado supervisado con:

- ✅ **18,612 registros mensuales** (12x más que versión anterior)
- ✅ Carga y limpieza de datos automatizada
- ✅ 8 modelos diferentes: 4 baseline + 4 avanzados
- ✅ Validación cruzada 5-fold estandarizada
- ✅ Evaluación en test set con métricas estándar
- ✅ Visualizaciones comprehensivas de diagnóstico
- ✅ Feature importance para modelos de árbol
- ✅ Reportes y exportación de resultados (JSON, CSV)

---

## 🚀 Inicio Rápido

### Opción 1: Jupyter Notebook (Recomendado)
```bash
jupyter notebook ../01_pipeline_completo.ipynb
```

### Opción 2: Script Python Completo
```bash
python ../main.py
```

### Opción 3: Test Simple
```bash
python ../test_pipeline_simple.py
```

---

## 📖 Documentación Completa

📌 **[README_CONSOLIDATED.md](README_CONSOLIDATED.md)** ← Ir aquí para documentación completa

Este archivo consolidado incluye:
- ✅ Descripción completa de datos (18,612 registros mensuales)
- ✅ Metodología de modelado
- ✅ Resultados detallados de los 8 modelos
- ✅ Interpretación de visualizaciones
- ✅ Guía de uso en código Python
- ✅ Troubleshooting
- ✅ Referencias teóricas

---

## Estructura de Archivos

```
supervised/
├── 01_pipeline_completo.ipynb        ← Notebook interactivo
├── main.py                           ← Script principal
│
├── src/
│   ├── data_loader.py               ← Carga 18,612 registros
│   ├── Models.py                    ← 8 modelos
│   └── Evaluation.py                ← Métricas y visualizaciones
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
│   └── resumen_ejecutivo.json
│
└── docs/
    ├── README_CONSOLIDATED.md       ← Documentación completa
    ├── README.md                    ← Este archivo
    ├── CONCLUSIONES.md
    └── QUICK_REFERENCE.md
```

---

## 🎯 Modelos

### Baseline (4 modelos lineales)
1. **OLS** - Regresión lineal simple
2. **Ridge** - Regresión con regularización L2
3. **Lasso** - Regresión con regularización L1
4. **ElasticNet** - Combinación L1 + L2

### Avanzados (4 modelos)
5. **Random Forest** 🏆 - RMSE: 0.5566 (9.4% mejor que OLS)
6. **Gradient Boosting** - RMSE: 0.6002
7. **SVR** - Support Vector Regression
8. **KNN** - K-Nearest Neighbors

---

## 📊 Datos

**Fuente:** `data_simulada/processed/data_simulado_1980_2026.csv`
- Registros: **18,612** (mensual)
- Período: 1980-2026 (47 años)
- Departamentos: 33
- Features: 27 (después de limpieza)
- Target: `outcome_tasa_desercion_snies`

---

## 🔗 Documentación Relacionada

- **[README_CONSOLIDATED.md](README_CONSOLIDATED.md)** — Documentación completa (→ empezar aquí)
- **[CONCLUSIONES.md](CONCLUSIONES.md)** — Análisis detallado
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — Comandos rápidos
- **[INDEX.md](INDEX.md)** — Índice completo

---

**Última actualización:** Mayo 18, 2026  
**Versión:** 2.0 (18,612 registros mensuales)  
**Responsable:** Juanes

→ **[Ir a documentación completa](README_CONSOLIDATED.md)**
