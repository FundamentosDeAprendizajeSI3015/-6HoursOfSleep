# INDICE DE DOCUMENTACION - PIPELINE SUPERVISADO

## Documentos Principales

### 1. README.md [LEER PRIMERO]

- Descripcion general del proyecto
- Estructura de archivos
- Instrucciones de uso
- Modelos explicados
- Metricas y evaluacion
- Troubleshooting

### 2. QUICK_REFERENCE.md [GUIA RAPIDA]

- Como ejecutar el pipeline (3 opciones)
- Tabla de los 8 modelos
- Configuracion CV
- Datos de entrada/salida
- Atajos y comandos utiles

### 3. CONCLUSIONES.md [ANALISIS DETALLADO]

- Desempeño de cada modelo
- Interpretacion de resultados
- Limitaciones observadas
- Recomendaciones accionables
- Proximos pasos sugeridos

## Codigo Fuente

### src/data_loader.py

```python
# Carga dataset_con_indice.csv
# Limpia datos (nulos, duplicados)
# Selecciona features
# Retorna (X, y, loader)
```

### src/Models.py [CON COMENTARIOS EXPLICATIVOS]

```python
# 8 modelos en pipelines sklearn:
# - OLS, Ridge, Lasso, ElasticNet (baseline)
# - RandomForest, GradientBoosting, SVR, KNN (avanzados)
# Incluye CV 5-fold y entrenamiento
```

### src/Evaluation.py

```python
# Evaluacion en test set
# Calcula: RMSE, MAE, R2, MAPE
# Genera visualizaciones (incluyendo comparativa regional)
# Exporta resultados
```

### src/utils.py

```python
# Funciones auxiliares
# Reportes formateados
# Exportacion CSV/TXT
```

## Scripts Principales

### main.py

```bash
python main.py  # Ejecucion completa del pipeline
```

### 01_pipeline_completo.ipynb

```bash
jupyter notebook 01_pipeline_completo.ipynb  # Notebook interactivo
```

## Salidas Generadas

### Reportes

- reports/tabla_comparativa_modelos_supervisados.csv
- reports/informe_supervisado.txt

### Visualizaciones

- reports/eda_figures/supervisado_predicho_vs_real.png
- reports/eda_figures/supervisado_residuales.png
- reports/eda_figures/supervisado_comparacion_metricas.png
- reports/eda_figures/supervisado_comparacion_departamentos.png
- reports/eda_figures/supervisado_importancia_randomforest.png
- reports/eda_figures/supervisado_importancia_gradientboosting.png

## Modelos - Resumen Rapido

| Modelo           | Tipo              | Mejor Para                      | RMSE   |
| ---------------- | ----------------- | ------------------------------- | ------ |
| GradientBoosting | Ensemble Boosting | Rendimiento maximo              | 0.0517 |
| RandomForest     | Ensemble Arbol    | Interpretabilidad + rendimiento | 0.0545 |
| OLS              | Lineal            | Baseline/Interpretacion         | 0.0621 |
| Ridge            | Lineal L2         | Multicolinealidad               | 0.0651 |
| ElasticNet       | Lineal L1+L2      | Balance regularizacion          | 0.0698 |
| Lasso            | Lineal L1         | Feature selection               | 0.0698 |
| SVR              | Kernel            | Datos no-lineales               | 0.0713 |
| KNN              | Distancia         | Patrones locales                | 0.0750 |

## Flujo de Trabajo

```
1. LOAD (data_loader.py)
   -> Carga datos desde CSV
   -> Limpia nulos y duplicados
   -> Selecciona 32 features

2. PREPARE (Models.py)
   -> Split 80/20 train/test
   -> Imputa con media
   -> Estandariza features

3. TRAIN (Models.py)
   -> CV 5-fold para cada modelo
   -> Calcula metricas CV
   -> Entrena en set completo

4. EVALUATE (Evaluation.py)
   -> Predice en test
   -> Calcula metricas finales
   -> Genera visualizaciones

5. EXPORT (utils.py + main.py)
   -> CSV con tabla comparativa
   -> TXT con reporte
   -> PNG con graficos
```

## Preguntas Frecuentes

**P: Cual es el mejor modelo?**
R: GradientBoosting (RMSE: 0.0517, 16.75% mejor que OLS)

**P: Como ejecuto el pipeline?**
R: python main.py desde la carpeta supervised/

**P: Donde veo los resultados?**
R: Carpeta reports/ con CSV, TXT y PNG

**P: Como cambio los hiperparametros?**
R: Edit config.py o modifica directamente en Models.py

**P: Como entiendo cada modelo?**
R: Lee comentarios explicativos en Models.py

## Contacto y Cambios

- Responsable: Juanes
- Rama: feature/supervised-models
- Ultima actualizacion: Mayo 11, 2026
- Estado: COMPLETADO Y LISTO PARA PRODUCCION

---

## Proximas Acciones Sugeridas

1. Revisar CONCLUSIONES.md para recomendaciones
2. Ejecutar python main.py para reproducir resultados
3. Explorar visualizaciones en reports/eda_figures/
4. Considerar ensemble de top 3 modelos
5. Recolectar mas datos (target: 100+ muestras)

**Inicio Recomendado:**

1. Leer: README.md
2. Referencia: QUICK_REFERENCE.md
3. Ejecutar: python main.py
4. Explorar: Archivos en reports/
5. Analizar: CONCLUSIONES.md
