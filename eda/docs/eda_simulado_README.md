# Análisis Exploratorio de Datos - Dataset Simulado

> **Rama/Enfoque:** Análisis Exploratorio de Datos (EDA) - Dataset Simulado con Cobertura Temporal Extendida

> **Objetivo:** Realizar un análisis detallado del dataset simulado de deserción estudiantil con cobertura histórica ampliada (1980-2026) para comprender las variables, evaluar la calidad de los datos, identificar patrones, descubrir relaciones y detectar posibles problemas como data leakage, con el fin de preparar el dataset para el modelado predictivo con mayor dimensionalidad temporal.

---

## Archivos de esta rama
```
│
├── -6HoursOfSleep/
│
│ ├── data_simulada/
│ │ └── processed/
│ │     └── data_simulado_1980_2026.csv
│ │
│ ├── eda/
│ │
│ │ ├── docs/
│ │ │   ├── eda_README.md
│ │ │   └── eda_simulado_README.md
│ │ │
│ │ ├── reports/
│ │ │   ├── dataset_original/
│ │ │   │   ├── 01_nulos_por_columna.png
│ │ │   │   ├── 02_distribucion_target.png
│ │ │   │   ├── 03_distribuciones_variables.png
│ │ │   │   ├── 04_boxplots_outliers.png
│ │ │   │   ├── 05_correlacion_con_target.png
│ │ │   │   ├── 06_heatmap_correlaciones.png
│ │ │   │   ├── 07_leakage_analysis.png
│ │ │   │   └── 08_evolucion_matricula.png
│ │ │   │
│ │ │   └── dataset_simulado/
│ │ │       ├── 01_nulos_por_columna.png
│ │ │       ├── 02_distribucion_target.png
│ │ │       ├── 03_distribuciones_variables.png
│ │ │       ├── 04_boxplots_outliers.png
│ │ │       ├── 05_correlacion_con_target.png
│ │ │       ├── 06_heatmap_correlaciones.png
│ │ │       ├── 07_leakage_analysis.png
│ │ │       └── 08_evolucion_matricula.png
│ │ │
│ │ └── src/
│ │     └── eda.py
```
---

## Propósito y Hallazgos Principales del EDA - Dataset Simulado

El análisis exploratorio de datos (EDA) del dataset simulado se llevó a cabo para obtener una comprensión profunda del panel de deserción estudiantil ampliado temporalmente (1980-2026), con cobertura de 47 años y 18,612 observaciones mensuales, cubriendo aspectos desde la carga inicial hasta la identificación de patrones de data leakage y la estructura temporal del panel.

### Propósito:
Se conservan los mismos pasos que se utilizaron para el EDA del dataser original:

1.  Carga e Inspección.
2.  Calidad de Datos.
3.  Estadísticas Descriptivas.
4.  Estructura del Panel.
5.  Distribución del Target.
6.  Distribuciones Numéricas.
7.  Detección de Outliers.
8.  Análisis de Correlaciones.
9.  Detección de Data Leakage.
10. Análisis Temporal.

### Hallazgos Clave:

*   **Mejora Sustancial de Datos:** El dataset simulado amplía significativamente la cobertura temporal del dataset original. Con 18,612 observaciones y 47 años de datos (vs. solo 33 observaciones con target en 2023 en el original), se habilita modelado temporal robusto y validación cruzada confiable.

*   **Calidad de Datos Excelente:** Solo 2.1% de valores nulos en dos variables (`pib_variacion_pct_anual_vs_anio_previo` y `var_pct_matriculados_vs_anio_previo`), lo que representa una mejora significativa respecto al dataset original (~57.6% en estas mismas variables). No hay duplicados detectados.

*   **Data Leakage Crítico Confirmado:** La variable `spadies_td_anual_universitario` mantiene correlación perfecta (r=0.8648) con el target `outcome_tasa_desercion_snies`. Esta variable **debe ser excluida** del conjunto de entrenamiento para evitar sobrestimar el rendimiento del modelo.

*   **Outliers Moderados pero Legítimos:** Se detectan outliers en 3.2-4.8% de variables económicas (PIB, matrícula, ratios) y hasta 4.7% en variables IPC. Estos reflejan legítimas variaciones económicas regionales y concentración en departamentos como Bogotá y Antioquia. No requieren remoción sino tratamiento cuidadoso.

*   **Alta Dimensionalidad Temporal:** La disponibilidad de target para 18,612 observaciones (vs. 33 en el original) permite:
      - Validación cruzada temporal robusta
      - Detección de tendencias y ciclos a largo plazo (1980-2026)
      - Modelos más complejos sin riesgo extremo de overfitting
      - Evaluación de efectos estacionales y económicos históricos

---

## Estadísticas Detalladas

### Dimensiones del Dataset
- **Total de Filas:** 18,612 observaciones
- **Total de Columnas:** 34 variables
- **Años Cubiertos:** 1980 – 2026 (47 años)
- **Departamentos Únicos:** 33
- **Periodicidad:** Mensual (12 meses × 33 departamentos × 47 años ≈ 18,612)

#### Duplicados
- **Filas duplicadas:** 0 ✓

---

## Análisis de Correlaciones

### Correlación de Pearson con `outcome_tasa_desercion_snies`

#### Correlaciones Altas (r > 0.80)
| Variable | Correlación | Riesgo |
|----------|------------|---------|
| `spadies_td_anual_tecnico_profesional` | **0.8839** ◆ | **CRÍTICO - DATA LEAKAGE** |
| `spadies_td_anual_tecnologico` | **0.8736** ◆ | **CRÍTICO - DATA LEAKAGE** |
| `spadies_td_anual_universitario` | **0.8648** ◆ | **CRÍTICO - DATA LEAKAGE** |
| `spadies_td_anual_tyt` | **0.8635** ◆ | **CRÍTICO - DATA LEAKAGE** |

#### Correlaciones Moderadas (0.30 < r < 0.50)
| Variable | Correlación | Interpretación |
|----------|------------|----------------|
| `pib_total_miles_millones_cop` | -0.4590 | Mayor PIB → Menor deserción |
| `ipc_capital_total_var_mensual_media` | 0.4241 | Mayor inflación capital → Mayor deserción |
| `ipc_nacional_total_var_mensual_media` | 0.4239 | Mayor inflación nacional → Mayor deserción |
| `ipc_nacional_educacion_var_mensual_media` | 0.4222 | Mayor inflación educativa → Mayor deserción |
| `proxy_pib_miles_mm_cop_por_matriculado` | -0.3969 | Mayor PIB/matriculado → Menor deserción |
| `geih_tgp_nacional_media_anual` | -0.3915 | Mayor participación laboral → Menor deserción |

#### Correlaciones Débiles (|r| < 0.30)
- Desempleo (GEIH): r = -0.27 a 0.14 (débil relación)
- Ratios educativos (`ratio_matriculados_sobre_admitidos`): r = 0.0998
- Variaciones porcentuales anuales: r ≈ 0.06 (prácticamente nulas)

---

## Comparativa: Dataset Original vs. Dataset Simulado

| Aspecto | Dataset Original | Dataset Simulado | Mejora |
|---------|-----------------|-----------------|--------|
| **Observaciones totales** | 33 | 18,612 | **564× más datos** |
| **Cobertura temporal** | 1 año (2023) | 47 años (1980-2026) | **47× más años** |
| **Posible validación cruzada** | No viable | Robusto | **Habilitado** |
| **Análisis temporal** | No posible | Completo | **Habilitado** |
| **Riesgo overfitting** | Extremo | Manejable | **Reducido** |
