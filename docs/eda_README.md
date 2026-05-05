# 🔵 Deserción Estudiantil EDA — Análisis Exploratorio de Datos

> **Rama/Enfoque:** Análisis Exploratorio de Datos (EDA)
> **Objetivo:** Realizar un análisis detallado de los datos disponibles para comprender las variables, evaluar la calidad de los datos, identificar patrones, descubrir relaciones y detectar posibles problemas como data leakage, con el fin de preparar el dataset para el modelado predictivo.

---

## 📁 Archivos de esta rama
```
│
├── -6HoursOfSleep/
│ ├── docs/
│ │ └── eda_README.md
│ │
│ ├── data/
│ │ └── processed/
│ │ └── panel_desercion_socioeconomico_completo_1.csv
│ │
│ ├── reports/
│ │ └── eda_figures/
│ │ ├── 01_nulos_por_columna.png
│ │ ├── 02_distribucion_target.png
│ │ ├── 03_distribuciones_variables.png
│ │ ├── 04_boxplots_outliers.png
│ │ ├── 05_correlacion_con_target.png
│ │ ├── 06_heatmap_correlaciones.png
│ │ ├── 07_leakage_analysis.png
│ │ └── 08_evolucion_matricula.png
│ │
│ └── src/
│ └── eda.py
```
---

## 🎯 Propósito y Hallazgos Principales del EDA

El análisis exploratorio de datos (EDA) se llevó a cabo para obtener una comprensión profunda del dataset de deserción estudiantil, cubriendo aspectos desde la carga inicial hasta la identificación de patrones de data leakage y la estructura temporal del panel.

### Propósito:

1.  **Carga e Inspección:** Verificación de dimensiones, años cubiertos, departamentos únicos y listado de columnas.
2.  **Calidad de Datos:** Detección de duplicados, cálculo de valores nulos por columna, y análisis de cardinalidad para identificar variables con baja o alta variabilidad.
3.  **Estadísticas Descriptivas:** Resumen numérico de variables (media, desviación estándar, percentiles) y cálculo del coeficiente de variación (CV).
4.  **Estructura del Panel:** Análisis de la cobertura de datos por departamento y año, especialmente para la variable target.
5.  **Distribución del Target:** Examen de la distribución de la tasa de deserción estudiantil (`outcome_tasa_desercion_snies`) y su ranking por departamento.
6.  **Distribuciones Numéricas:** Visualización de histogramas para variables numéricas clave.
7.  **Detección de Outliers:** Uso del método IQR para identificar valores atípicos en variables numéricas.
8.  **Análisis de Correlaciones:** Evaluación de la relación entre variables numéricas (Pearson) y, crucialmente, con la variable target.
9.  **Detección de Data Leakage:** Identificación explícita de variables que contienen información futura o son idénticas al target, lo cual es crítico para el modelado.
10. **Análisis Temporal:** Estudio de la evolución de variables clave (matrícula, PIB, desempleo) entre 2023 y 2024.

### Hallazgos Clave:

*   **Data Leakage Crítico:** La variable `spadies_td_anual_universitario` es idéntica a la variable target `outcome_tasa_desercion_snies`. Esta variable **debe ser excluida** del conjunto de entrenamiento para evitar sobrestimar el rendimiento del modelo.
*   **Correlaciones Altas pero No Perfectas:** `spadies_td_anual_tecnologico` (r=0.61) y `proxy_pib_miles_mm_cop_por_matriculado` (r=0.86) muestran correlaciones significativas con el target, pero no son data leakage directo.
*   **Variables Macroeconómicas de Baja Utilidad Cross-Seccional:** Las variables del IPC nacional y GEIH (desempleo) tienen baja discriminación entre departamentos en un mismo año, siendo más informativas a nivel nacional o como feature del año.
*   **Calidad de Datos:** Se identifican altas proporciones de valores nulos en `var_pct_matriculados_vs_anio_previo` (57.6%) y variables SPADIES (~50%), lo que requerirá estrategias de imputación o exclusión.
*   **Estructura del Panel Limitada:** La variable target (`outcome_tasa_desercion_snies`) solo está disponible para el año 2023 (33 observaciones), lo que limita el análisis temporal directo para entrenamiento y sugiere que el año 2024 podría usarse para predicción futura.
*   **Concentración de Outliers:** Variables como PIB total y número de matriculados presentan alta concentración en departamentos específicos (Bogotá, Antioquia), lo que genera outliers legítimos que reflejan desigualdad regional.
*   **Recomendaciones para Modelado:** Se sugiere excluir las variables con data leakage, evaluar cuidadosamente otras variables SPADIES, gestionar los nulos y considerar modelos más simples o regularizados dada la baja dimensionalidad efectiva (33 observaciones con target).

---

## 📊 Visualizaciones Generadas

Los siguientes gráficos fueron generados y guardados en `reports/eda_figures/`:

*   `01_nulos_por_columna.png`: Porcentaje de valores nulos por columna.
*   `02_distribucion_target.png`: Distribución de la tasa de deserción y ranking departamental.
*   `03_distribuciones_variables.png`: Histogramas de variables numéricas.
*   `04_boxplots_outliers.png`: Boxplots para identificar outliers por grupos de variables.
*   `05_correlacion_con_target.png`: Correlación de Pearson de cada variable con la tasa de deserción SNIES.
*   `06_heatmap_correlaciones.png`: Matriz de correlaciones completa entre variables numéricas.
*   `07_leakage_analysis.png`: Análisis visual de data leakage y correlaciones clave.
*   `08_evolucion_matricula.png`: Evolución de la matrícula en los top 10 departamentos entre 2023 y 2024.