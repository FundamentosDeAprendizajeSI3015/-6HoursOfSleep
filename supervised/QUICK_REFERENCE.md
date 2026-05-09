# QUICK REFERENCE - PIPELINE SUPERVISADO

## Ejecución Rápida

```bash
# Opcion 1: Script completo
cd supervised
python main.py

# Opcion 2: Notebook interactivo
jupyter notebook 01_pipeline_completo.ipynb

# Opcion 3: Quick execution
python quickstart.py
```

## Estructura de Codigo

```
src/
├── data_loader.py      # Carga datos desde dataset_con_indice.csv
├── Models.py           # Define 8 modelos (4 baseline + 4 avanzados)
├── Evaluation.py       # Calcula metricas y genera visualizaciones
└── utils.py            # Funciones auxiliares
```

## Los 8 Modelos

### Baseline (4)
| Modelo | Tipo | Hiperparametros |
|--------|------|-----------------|
| OLS | Lineal sin regularizacion | - |
| Ridge | Lineal L2 | alpha=1.0 |
| Lasso | Lineal L1 | alpha=0.1 |
| ElasticNet | Lineal L1+L2 | alpha=0.1, l1_ratio=0.5 |

### Avanzados (4)
| Modelo | Tipo | Hiperparametros |
|--------|------|-----------------|
| RandomForest | Ensemble arbol | n_estimators=100 |
| GradientBoosting | Ensemble boosting | n_estimators=100, lr=0.1 |
| SVR | Kernel RBF | C=1.0, epsilon=0.1 |
| KNN | Distancia basada | n_neighbors=5 |

## Metricas Principales

```
MODELO GANADOR: GradientBoosting

Test Set:
  RMSE: 0.0517  (Error cuadratico medio)
  MAE:  0.0293  (Error absoluto medio)
  R²:   0.4123  (Varianza explicada)
  MAPE: 20.41%  (Error porcentual)

vs Baseline OLS:
  Mejora: 16.75% en RMSE
```

## Archivos de Salida

| Archivo | Tipo | Ubicacion |
|---------|------|-----------|
| tabla_comparativa_modelos_supervisados.csv | CSV | reports/ |
| informe_supervisado.txt | TXT | reports/ |
| supervisado_predicho_vs_real.png | PNG | reports/eda_figures/ |
| supervisado_residuales.png | PNG | reports/eda_figures/ |
| supervisado_comparacion_metricas.png | PNG | reports/eda_figures/ |
| supervisado_importancia_randomforest.png | PNG | reports/eda_figures/ |
| supervisado_importancia_gradientboosting.png | PNG | reports/eda_figures/ |

## Datos

```
Dataset: dataset_con_indice.csv
Casos limpios: 33 (de 66 originales)
Features: 32 (de 37 originales)
Target: outcome_tasa_desercion_snies
Rango: [0.047, 0.263] (tasa de desercion)
```

## Configuracion CV

```
Train/Test split: 80/20
Cross-Validation: 5-fold
Random state: 42 (reproducibilidad)
Scaler: StandardScaler (todos los modelos)
```

## Workflow Completo

1. **Load** → data_loader.py
   - Carga datos
   - Limpia nulos
   - Selecciona features

2. **Prepare** → Models.py
   - Split train/test 80/20
   - Imputa nulos con media
   - Estandariza features

3. **Train** → Models.py
   - CV 5-fold para cada modelo
   - Entrena en set completo
   - Calcula metricas CV

4. **Evaluate** → Evaluation.py
   - Predice en test set
   - Calcula metricas test
   - Genera visualizaciones

5. **Export** → utils.py
   - Tabla CSV
   - Reporte TXT
   - Graficos PNG

## Archivo de Configuracion

```python
# config.py
RANDOM_STATE = 42      # Reproducibilidad
TEST_SIZE = 0.20       # 80% train, 20% test
CV_FOLDS = 5           # Validacion cruzada
TARGET_COL = "outcome_tasa_desercion_snies"
```

## Cambios Recientes

- [✓] Emojis removidos de codigo fuente
- [✓] Comentarios explicativos agregados a Models.py
- [✓] Joblib parallel deshabilitado en Windows
- [✓] Indentacion corregida
- [✓] Pipeline ejecutado exitosamente
- [✓] Todas las salidas generadas

## Para Más Información

- `README.md` - Documentacion completa
- `RESUMEN_FINAL.md` - Estado del proyecto
- `CONCLUSIONES.md` - Analisis detallado
- `01_pipeline_completo.ipynb` - Notebook interactivo

---

*Ultima actualizacion: Mayo 5, 2026*
