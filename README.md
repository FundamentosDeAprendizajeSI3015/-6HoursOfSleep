# Deserción Estudiantil e Indicadores Económicos en Colombia

## Informe Técnico 

> **Pipeline de Aprendizaje Automático para Predicción de Deserción Estudiantil Universitaria**

**Fecha:** Mayo 2026  
**Equipo:** Santiago Gómez, Isabella Camacho, Juan Esteban Alzate

---

## Tabla de Contenidos

1. [Descripción del Problema](#descripción-del-problema)
2. [Metodología](#metodología)
3. [Datos](#datos)
4. [Fuentes](#fuentes)
5. [Análisis Exploratorio de Datos](#análisis-exploratorio-de-datos)
6. [Simulación de Datos](#simulación-de-datos)
7. [Modelado](#modelado)
8. [Aprendizaje Supervisado](#aprendizaje-supervisado)
9. [Aprendizaje No Supervisado](#aprendizaje-no-supervisado)
10. [Índices y PCA](#índices-y-pca)
11. [Visualización y Dashboard](#visualización-y-dashboard)
12. [Retos Identificados](#retos-identificados)
13. [Conclusiones](#conclusiones)
14. [Instrucciones para Correr el Código](#instrucciones-para-correr-el-código)

---

## Descripción del Problema

La **deserción estudiantil** es un desafío crítico en la educación superior colombiana que afecta tanto la estabilidad financiera de las instituciones como el capital humano del país. A diferencia de un enfoque mas tradicional, que analiza la deserción desde aspectos académicos o personales en la vida de los estudiantes, este proyecto busca explicar la deserción desde **variables económicas y sociales agregadas**, reconociendo que:

- La deserción no es solo un problema individual, sino **sistémico**
- Factores macroeconómicos (desempleo, inflación, ciclos económicos) afectan **masivamente** las decisiones de permanencia
- Los departamentos con mayor pobreza y desempleo tienden a tener mayores tasas de deserción
- Las crisis económicas nacionales se correlacionan con picos de abandono estudiantil

Por otro lado, la relación causal y predictiva entre los diversos indicadores económicos y la tasa de deserción no ha sido sistematizada mediante técnicas modernas de machine learning. Este proyecto busca cerrar esa brecha a través de:

1. Caracterización de la deserción estudiantil como función de variables macroeconómicas, microeconómicas y educativas
2. Identificación patrones regionales de vulnerabilidad económica mediante clustering
3. Construcción modelos predictivos robustos que permitan anticipar crisis educativas anticipándose a ciclos económicos desfavorables

---

## Metodología

El proyecto sigue un enfoque **data-driven** estructurado en 5 fases:

1. Preparación de Datos
2. Análisis Exploratorio
3. Análisis No Supervisado
4. Construcción de Índices
5. Modelado Supervisado

Cada fase del proyecto se encuentra organizada en una carpeta independiente correspondiente a su rama:

- `load-data`
- `eda`
- `indexes-scores`
- `unsupervised`
- `supervised`
- `dashboard`

Las etapas comparten la misma estructura interna:

- `src/` → Código fuente y scripts principales.
- `docs/` → Documentación y README específicos de la etapa.
- `reports/` → Gráficas, resultados y salidas generadas.

Las carpetas globales `data/` y `data_simulada/`centralizan los datasets utilizados en todo el proyecto.
Las carpeta  `dashboard/` contiene una interfaz interactiva para ver los resultados obtenidos a lo largo del pipeline.

---

## Datos

### Características del Dataset Principal

```
Período:               1980-2026 (47 años)
Granularidad:         Mensual
Departamentos:        33 (Colombia)
Registros totales:    18,612 (33 × 47 años × 12 meses)
Variables:            39 columnas (32 features + target)
```

### Limpieza Realizada

| Paso | Acción | Columnas Afectadas | Justificación |
|------|--------|-------------------|--------------|
| **Data Leakage** | Eliminación | `spadies_td_anual_universitario` | r=1.0 con target (idéntica variable) |
| **Técnicas** | Eliminación | `outcome_merge_pendiente` | Columna de control sin valor predictivo |
| **Valores Nulos** | Imputación media | 792 valores (~0.38%) | Estrategia estándar para pequeños gaps |
| **Escalado** | StandardScaler | Todas las features | Normalización para modelos sensibles a escala |

### Composición de Variables Finales (27 features)

#### Variables Macroeconómicas (9)
- `pib_total_miles_millones_cop` — PIB nacional
- `pib_variacion_pct_anual_vs_anio_previo` — Crecimiento interanual
- `ipc_nacional_total_var_mensual_*` (media/mediana/std) — Inflación general
- `ipc_nacional_educacion_var_mensual_*` (media/mediana/std) — Inflación en educación
- `ipc_nacional_alimentos_var_mensual_*` — Inflación en alimentos

#### Variables de Mercado Laboral (3)
- `geih_td_nacional_media_anual` — Tasa de desempleo nacional
- `geih_to_nacional_media_anual` — Tasa de ocupación
- `geih_tgp_nacional_media_anual` — Tasa global de participación

#### Variables Educativas (6)
- `total_admitidos` — Nuevos estudiantes admitidos
- `total_matriculados` — Matrícula activa total
- `ratio_matriculados_sobre_admitidos` — Tasa de aceptación
- `var_pct_matriculados_vs_anio_previo` — Crecimiento de matrícula
- `proxy_pib_miles_mm_cop_por_matriculado` — PIB por estudiante
- `spadies_td_anual_tecnologico` — Deserción tecnológica

#### Variable Objetivo (Target)
- **`outcome_tasa_desercion_snies`** — Tasa de deserción estudiantil (%)

---

## Fuentes

| Fuente | Datos | URL | Variables |
|--------|-------|-----|-----------|
| **DANE** | PIB departamental, empleo | https://www.dane.gov.co | PIB, desempleo, ocupación, participación |
| **BanRep** | Inflación, tasas de interés | https://www.banrep.gov.co | IPC general, IPC por categoría |
| **MEN/SPADIES** | Matrícula y deserción | https://www.mineducacion.gov.co | Matrícula, deserción por nivel |
| **Banco Mundial** | Indicadores complementarios | https://data.worldbank.org | Datos macroeconómicos históricos |

---

## Análisis Exploratorio de Datos

### Propósito del EDA

Obtener una comprensión profunda del dataset mediante:

1. **Inspección básica** — Dimensiones, períodos, cobertura por departamento
2. **Calidad de datos** — Duplicados, valores nulos, cardinalidad
3. **Estadísticas descriptivas** — Media, desv. est., percentiles, coeficiente de variación
4. **Distribuciones** — Histogramas, boxplots, identificación de outliers
5. **Correlaciones** — Análisis Pearson con variable target
6. **Data leakage** — Detección de variables idénticas o correlacionadas perfectamente

### Hallazgos Clave

#### 🔴 Data Leakage Crítico
**Variable:** `spadies_td_anual_universitario`  
**Correlación con target:** r = 1.0 (perfecta)  
**Acción:** Eliminada del entrenamiento  
**Impacto:** Previene sobrestimación del rendimiento del modelo

#### 🟡 Correlaciones Moderadas Útiles
- `spadies_td_anual_tecnologico`: r = 0.61 (deserción técnica)
- `proxy_pib_miles_mm_cop_por_matriculado`: r = 0.86 (PIB por estudiante)

#### 🟡 Calidad de Datos
- Valores nulos altos en `var_pct_matriculados_vs_anio_previo` (57.6%)
- Valores nulos ~50% en variables SPADIES secundarias
- **Estrategia:** Imputación con media o exclusión selectiva

#### 🟢 Estructura del Panel
- Target disponible para múltiples años y departamentos
- Cobertura heterogénea por región (mayor en departamentos centrales)
- Variables económicas nacionales de baja discriminación cross-seccional

#### 📊 Outliers Legítimos
- PIB y matrícula concentrados en Bogotá y Antioquia (27-35% del total)
- Estos outliers reflejan desigualdad regional real, no errores

---

## Simulación de Datos

### Objetivo

Ampliar la cobertura temporal del dataset de ~33 observaciones (solo 2023 para target) a **18,612 registros mensuales** (1980-2026, 33 departamentos × 47 años × 12 meses) manteniendo realismo estadístico.

### Componentes Simulados

#### 1. **Variables Macroeconómicas**

**PIB (Lognormal + Tendencias)**
```
PIB(t) = Base × [Logística(t) × 2 + Tendencia(t) + Ciclo(t) + Crisis(t)] × Ruido_LN(σ=0.1)
```
- **Logística:** Crecimiento en forma de S (1980-2024)
- **Tendencia:** Crecimiento lineal con inflexión en 2005
- **Ciclo:** Oscilación económica cada 8 años
- **Crisis:** Shocks simulados en 1999 (crisis financiera), 2008 (crisis global), 2020 (COVID-19)

**Inflación (Normal por Régimen)**
```
1980-1990: Media=2.5%, Std=0.8%  (Inflación Alta)
1991-2000: Media=1.2%, Std=0.4%  (Inflación Moderada)
2001-2024: Media=0.6%, Std=0.2%  (Inflación Baja)
```

#### 2. **Variables de Mercado Laboral**

- **Tasa de desempleo:** Distribución normal con ciclos de 8 años (4% - 22%)
- **Tasa de ocupación:** Normal inversa (45% - 70%)
- **Tasa global de participación:** Normal con tendencia creciente (52% - 68%)

#### 3. **Variables Educativas**

- **Matrícula total:** Distribución Poisson (nuevos estudiantes)
- **Ratio aceptación:** Derivado como función de capacidad institucional
- **Variación interanual:** Normal calibrada con datos observados

#### 4. **Deserción (SPADIES)**

Simulada mediante distribución Beta (acotada en [0,1]) con correlaciones realistas:
- `spadies_td_anual_universitario` (universitaria)
- `spadies_td_anual_tyt` (técnico y tecnológico)
- `spadies_td_anual_tecnico_profesional` (técnico profesional)
- `spadies_td_anual_tecnologico` (tecnológica)

### Validación de Simulación

- ✅ Correlación de variables preservada según literatura
- ✅ Eventos históricos incorporados (crisis 1999, 2008, 2020)
- ✅ Calibración con valores observados 2023-2024
- ✅ Heterogeneidad regional reflejada en distribuciones por departamento

---

## Modelado

### Enfoque General

El modelado supervisado se divide en dos estrategias complementarias:

#### **Baseline (4 modelos lineales)**
Proporcionan interpretabilidad máxima y puntos de referencia:
1. **OLS** — Regresión lineal sin regularización
2. **Ridge** — Regularización L2 (α=1.0)
3. **Lasso** — Regularización L1 (α=0.1)
4. **ElasticNet** — Híbrido L1+L2 (α=0.1, L1_ratio=0.5)

#### **Avanzados (4 modelos no-lineales)**
Capturan interacciones complejas:

5. **Random Forest** — Ensemble de árboles (100 estimadores)
6. **Gradient Boosting** — Boosting secuencial (100 estimadores, lr=0.1)
7. **SVR (RBF)** — Support Vector Regression con kernel RBF
8. **KNN** — K-Nearest Neighbors (k=5)

### Estrategia de Validación

```
Dataset (18,612 registros)
    ↓
Train (80%): 14,889 registros
    ├── 5-Fold Cross Validation
    └── Métricas: RMSE, MAE, R², MAPE
    ↓
Test (20%): 3,723 registros
    └── Evaluación final del ganador
```

---

## Aprendizaje Supervisado

### Resultados de Validación Cruzada (5-Fold)

```
RANKING POR RMSE (Validación Cruzada)
═══════════════════════════════════════════════════════════════════
  1. RandomForest      │ RMSE: 0.7552 ± 0.0063 │ R²: 0.9233 ± 0.0029
  2. KNN               │ RMSE: 0.7730 ± 0.0027 │ R²: 0.9159 ± 0.0019
  3. GradientBoosting  │ RMSE: 0.7815 ± 0.0046 │ R²: 0.9121 ± 0.0031
  4. SVR               │ RMSE: 0.7856 ± 0.0036 │ R²: 0.9102 ± 0.0027
  5. Ridge             │ RMSE: 0.7886 ± 0.0028 │ R²: 0.9088 ± 0.0021
  6. OLS               │ RMSE: 0.7886 ± 0.0028 │ R²: 0.9088 ± 0.0021
  7. ElasticNet        │ RMSE: 0.7934 ± 0.0028 │ R²: 0.9066 ± 0.0022
  8. Lasso             │ RMSE: 0.7983 ± 0.0023 │ R²: 0.9043 ± 0.0024
═══════════════════════════════════════════════════════════════════
```

### 🏆 Modelo Ganador: Random Forest (Test Set)

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **RMSE** | 0.5566 | Error cuadrático medio (puntos porcentuales) |
| **MAE** | 0.4334 | Error absoluto medio (pp) |
| **R²** | 0.9284 | Explica 92.84% de varianza en test |
| **MAPE** | 4.32% | Error porcentual promedio |
| **Mejora vs OLS** | +9.4% | Ganancia respecto al baseline |

### Comparativa Test Set (20% datos)

```
RANKING POR RMSE (Test Set Final)
═══════════════════════════════════════════════════════════════════
  1. RandomForest      │ RMSE: 0.5566 │ MAE: 0.4334 │ R²: 0.9284
  2. KNN               │ RMSE: 0.5751 │ MAE: 0.4224 │ R²: 0.9235
  3. GradientBoosting  │ RMSE: 0.6002 │ MAE: 0.4809 │ R²: 0.9167
  4. SVR               │ RMSE: 0.6060 │ MAE: 0.4784 │ R²: 0.9151
  5. OLS (Baseline)    │ RMSE: 0.6146 │ MAE: 0.4924 │ R²: 0.9126
  6. Ridge             │ RMSE: 0.6147 │ MAE: 0.4924 │ R²: 0.9126
  7. ElasticNet        │ RMSE: 0.6220 │ MAE: 0.4991 │ R²: 0.9105
  8. Lasso             │ RMSE: 0.6308 │ MAE: 0.5058 │ R²: 0.9080
═══════════════════════════════════════════════════════════════════
```

### Importancia de Variables (Random Forest)

Las variables más influyentes según feature importance:

1. **`spadies_td_anual_tecnologico`** — Deserción tecnológica (~15%)
2. **`pib_variacion_pct_anual_vs_anio_previo`** — Crecimiento del PIB (~12%)
3. **`proxy_pib_miles_mm_cop_por_matriculado`** — PIB per cápita estudiantil (~11%)
4. **`total_matriculados`** — Tamaño del sistema educativo (~10%)
5. **`geih_td_nacional_media_anual`** — Desempleo nacional (~9%)

### Análisis de Desempeño

#### ✅ Fortalezas
- **R² alto (0.93):** El modelo explica >92% de la varianza en predicciones
- **Estabilidad en CV:** Baja desviación estándar (±0.006) indica robustez
- **MAPE bajo (4.32%):** Errores porcentuales manejables para políticas
- **Ensemble efectivo:** Random Forest supera modelos lineales en 9.4%

#### ⚠️ Limitaciones
- **Varianza residual:** ~7% de varianza no explicada sugiere factores externos
- **Heterogeneidad regional:** Errores mayores en departamentos periféricos
- **Sesgo conservador:** Modelo tiende a subestimar en regiones de alta deserción

---

## Aprendizaje No Supervisado

### Objetivo

Explorar estructura latente en los indicadores económico-educativos **sin usar la variable target** para:
- Identificar grupos de departamentos con perfiles similares
- Validar si clusters naturales se alinean con níveis de deserción
- Generar variables de apoyo para el modelado supervisado

### Algoritmos Implementados

#### 1. **K-Means (k=2 a 7)**

**Resultados óptimos (k=3):**

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Silhouette Score** | 0.4306 | Separación moderada entre clusters |
| **Calinski-Harabasz** | 9154.36 | Alta ratio densidad/separación |
| **Davies-Bouldin** | 0.9179 | Baja ratio (mejor compactidad) |

**Distribución de clusters:**
- Cluster 0: 32 observaciones (grupo mayoritario)
- Cluster 1: 32 observaciones (segundo grupo)
- Cluster 2: 2 observaciones (outliers)

#### 2. **Agglomerative Clustering (Jerárquico)**

Resultados **prácticamente idénticos** a K-Means (ARI ≈ 0.999), confirmando estabilidad de la estructura de clusters.

#### 3. **DBSCAN**

- Identifica solo 2 clusters densos (7 observaciones totales)
- Clasifica 59 observaciones como ruido
- **Conclusión:** DBSCAN inapropiado para este dataset (mala separabilidad)

### Relación con Tasa de Deserción

| Algoritmo | ARI (Adjusted Rand Index) | Interpretación |
|-----------|--------------------------|----------------|
| **K-Means** | 0.0014 | Muy baja coincidencia con clusters de deserción |
| **Agglomerative** | 0.0014 | Igual resultado que K-Means |
| **DBSCAN** | 1.0000 | Perfecto, pero solo en subconjunto no-ruido (7 obs) |

### Conclusión Principal

**Los clusters naturales NO se alinean directamente con la tasa de deserción.** Esto indica:

1. La estructura de variables económicas es **ortogonal** a la estructura de deserción
2. Se requiere un enfoque **supervisado** que use explícitamente el target
3. Los clusters de K-Means son útiles como variables auxiliares pero NO como predictores directos

---

## Índices y PCA

### Objetivo

Construir **índices de vulnerabilidad económica-educativa** como combinación lineal de variables para sintetizar la complejidad multidimensional en métricas interpretables.

### Metodología

#### **Fase 1: Preprocesamiento**

Selección de 7 variables clave (de 32 totales):
- `proxy_pib_miles_mm_cop_por_matriculado` — Inversión por estudiante
- `spadies_td_anual_tecnologico` — Deserción técnica
- `total_matriculados` — Tamaño del sistema
- `geih_td_nacional_media_anual` — Desempleo
- `ipc_nacional_total_var_mensual_media` — Inflación
- 2 variables adicionales según correlación con target

#### **Fase 2: PCA (Reducción de Dimensiones)**

| Dataset | PC1 varianza explicada | PC2 varianza explicada | Varianza Total | Registros |
|--------|--------------------------|---------|---------------|-----------|
| **Original** | 44.8%  | 27.6% | 72.4%  | 66 |
| **Simulado** | 31.24% | 19.96% | 51.20% | 18,612 |

Cargas Principales Dataset Original(PC1):
- `proxy_pib_miles_mm_cop_por_matriculado`: 0.705 (inversión educativa)
- `spadies_td_anual_tecnologico`: 0.699 (vulnerabilidad técnica)
- `total_matriculados`: -0.106 (escala del sistema)

Cargas Principales Dataset Simulado(PC1):
- `ipc_nacional_total_var_mensual_media`: 0.542 (inflación)
- `geih_to_nacional_media_anual`: 0.699 (tasa de ocupación - empleo)

#### **Fase 3: Construcción de Índices**

**Índice PCA (Automático)**
- Basado en cargas de PC1
- Normalizado [0,1]
- Correlación con outcome: ρ=0.387 (moderada)

**Índice Teórico (Basado en Expertos)**
- Pesos asignados según literatura económica
- Mayor peso a inversión por estudiante y desempleo
- Correlación con outcome: ρ=0.412 (ligeramente superior)

**Métricas de Correlación:**

| Índice | Correlación Spearman (ρ) | P-valor | Observaciones |
|--------|--------------------------|---------|---------------|
| **Índice PCA** | 0.6902 | 0.000 | 18,216 |
| **Índice Teórico** | 0.6287 | 0.000 | 18,612 |

*Nota: Correlaciones significativamente más altas en dataset simulado (ρ>0.62) comparado con dataset original (ρ≈0.39), sugiriendo mejor poder predictivo con cobertura temporal extendida.*

---

## Visualización y Dashboard

### Propósito del Dashboard

Proporcionar a stakeholders (rectores, formuladores de política) una interfaz interactiva para explorar:

1. **Resultados del modelado supervisado** — Métricas de desempeño, importancia de variables
2. **Análisis no supervisado** — Clusters y segmentación departamental
3. **Índices de vulnerabilidad** — Mapeo de riesgo por región
4. **Predicciones futuras** — Escenarios de deserción bajo diferentes condiciones macroeconómicas

### Tecnología

**Framework:** Dash + Plotly (interactivo)  
**Backend:** Python + pandas/scikit-learn  
**Ubicación:** `dashboard/app.py`

### Cómo Usar el Dashboard

```bash
# 1. Navegarr a la carpeta del dashboard
cd dashboard

# 2. Ejecutar la aplicación
python app.py

# 3. Abrir en navegador
# Acceder a: http://localhost:8050
```

### Componentes Principales

#### **Sección 1: Modelos Supervisados**
- Tabla comparativa de 8 modelos (RMSE, MAE, R², MAPE)
- Gráfico de validación cruzada (media ± std)
- Feature importance del modelo ganador (Random Forest)
- Curva de predicho vs real

#### **Sección 2: Análisis No Supervisado**
- Scatter plot: K-Means clusters (PC1 vs PC2)
- Métricas de validación (Silhouette, Calinski-Harabasz)
- Comparativa K-Means vs Agglomerative

#### **Sección 3: Índices y PCA**
- Biplot PCA con cargas de variables
- Índice PCA vs Índice Teórico (scatter)
- Mapa de vulnerabilidad por departamento

#### **Sección 4: Exploración Interactiva**
- Filtros por año y departamento
- Comparativa de indicadores seleccionados
- Descarga de resultados en CSV

---

## Retos Identificados

### 🔴 Retos Críticos

#### 1. **Tamaño de Muestra Inicial Limitado**
- **Problema:** Solo 33 observaciones originales con target disponible (2023)
- **Impacto:** Varianza alta, alto riesgo de overfitting, validación cruzada inestable
- **Solución adoptada:** Simulación de datos ampliando a 18,612 registros
- **Limitación:** Simulaciones introducen dependencia estadística artificial

#### 2. **Data Leakage Crítico**
- **Problema:** Variable `spadies_td_anual_universitario` idéntica al target
- **Impacto:** Sin detección, hubiera inflado métricas de desempeño
- **Solución:** Exclusión explícita durante limpieza
- **Lección:** Necesario EDA exhaustivo antes de modelado

#### 3. **Heterogeneidad Espacial No Capturada**
- **Problema:** Variables macroeconómicas (PIB, desempleo) son nacionales, no departamentales
- **Impacto:** Baja discriminación entre departamentos en mismo año
- **Solución parcial:** Proxies como PIB por matriculado
- **Necesario:** Datos departamentales granulares de DANE

### 🟡 Retos Moderados

#### 4. **Multicolinealidad entre Variables**
- **Problema:** Variables SPADIES correlacionadas entre sí (r>0.6)
- **Impacto:** Inestabilidad en modelos lineales
- **Solución:** Ridge/Lasso regularización; modelos de árbol más robustos

#### 5. **Desalineamiento Clustering-Target**
- **Problema:** Clusters K-Means (ARI=0.0014) no predicen deserción
- **Impacto:** Clustering no sirve como predictor directo
- **Implicación:** Estructura económica ≠ estructura de deserción

#### 6. **Valores Nulos Heterogéneos**
- **Problema:** Variables SPADIES tienen ~50% nulos; otras <5%
- **Impacto:** Estrategias de imputación pueden introducir sesgos
- **Solución:** Análisis de sensibilidad con múltiples estrategias de imputación

### 🟢 Retos Menores

#### 7. **Interpretabilidad vs Precisión**
- **Trade-off:** Random Forest (R²=0.93) vs OLS (R²=0.91, interpretable)
- **Solución:** Combinar modelos para diferentes audiencias

#### 8. **Generalización Geográfica**
- **Problema:** Modelo entrenado en 33 departamentos, limitada validación externa
- **Solución:** Validación cruzada estratificada por región

---

## Conclusiones

### 🎯 Hallazgos Principales

1. **Random Forest es el modelo óptimo** con R²=0.9284, capturando >92% de varianza en predicciones de deserción
   - Mejora de **+9.4%** respecto a baseline OLS
   - Robusto en validación cruzada (std=0.006)
   - MAPE de 4.32% es práctico para política educativa

2. **La estructura de variables económicas es ortogonal a clusters de deserción**
   - ARI=0.0014 indica desalineamiento entre clustering automático y target
   - Implica que la deserción es impulsada por factores complejos no capturados por clustering no supervisado
   - Modelado supervisado es imprescindible

3. **Índices de vulnerabilidad sintetizan información efectivamente**
   - Índice PCA explica 72.4% de varianza con 2 componentes (de 7 variables)
   - Correlación moderada con target (ρ=0.387-0.412) sugiere capacidad predictiva
   - Útiles para comunicación a stakeholders

4. **Las variables más influyentes reflejan intuición económica**
   - Deserción tecnológica (r=0.61) es fuerte predictor
   - Inversión por estudiante (PIB/matrícula) correlaciona con outcome
   - Desempleo nacional tiene impacto significativo

### ⚠️ Limitaciones Reconocidas

1. **Simulación introduce dependencias artificiales** — Resultados optimistas respecto a datos completamente reales
2. **Cobertura temporal limitada de target** — Solo 2023 disponible, reduce validación temporal
3. **Factores cualitativos no capturados** — Conflicto armado, infraestructura específica, calidad pedagógica
4. **Tamaño de muestra original muy pequeño** — n=33 departamentos × 1 año = 33 observaciones

---

## Instrucciones para Correr el Código

### Requisitos Previos

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd Desercion_Estudiantil/-6HoursOfSleep

# 2. Crear ambiente virtual
python -m venv venv

# 3. Activar ambiente
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

### Ejecución por Etapa

#### **Etapa 1: Generación de Datos Simulados**

```bash
cd data_simulada
python init_data.py

# Genera:
# - data_simulada/processed/data_simulado_1980_2026.csv (18,612 registros)
# - data_simulada/processed/resumen_simulacion.txt (documentación)
```

#### **Etapa 2: Análisis Exploratorio de Datos (EDA)**

```bash
# Análisis del dataset simulado
cd eda
python src/eda.py

# Genera visualizaciones en:
# - eda/reports/dataset_simulado/*.png (8 gráficos)
```

#### **Etapa 3: Construcción de Índices**

```bash
cd indexes-scores
python src/index_builder.py

# Genera:
# - data/final/dataset_con_indice.csv (dataset enriquecido)
# - indexes-scores/reports/dataset_simulado/*.png (6 visualizaciones)
```

#### **Etapa 4: Modelado Supervisado (Completo)**

```bash
# Opción A: Script Python completo
cd supervised
python main.py

# Opción B: Notebook interactivo (recomendado)
jupyter notebook 01_pipeline_completo.ipynb

# Opción C: Quick execution
python quickstart.py

# Genera:
# - supervised/results/pipeline_results.json
# - supervised/results/resumen_ejecutivo.json
# - supervised/reports/eda_figures/*.png (5 visualizaciones)
```

#### **Etapa 5: Modelado No Supervisado (Clustering)**

```bash
cd unsupervised
python main.py

# Genera:
# - unsupervised/reports/cluster_assignments_*.csv (K-Means, DBSCAN, Agglomerative)
# - unsupervised/reports/unsupervised_validation_summary.csv
# - unsupervised/docs/unsupervised_results.json
```

#### **Etapa 6: Dashboard Interactivo**

```bash
cd dashboard
python app.py

# Salida esperada:
# Running on http://127.0.0.1:8050/
# Abrir navegador en esa dirección
```

### Archivos Clave de Referencia

| Archivo | Propósito | Ejecución |
|---------|----------|-----------|
| `data_simulada/init_data.py` | Generar 18,612 registros | `python init_data.py` |
| `eda/src/eda.py` | Análisis exploratorio | `python eda.py` |
| `indexes-scores/src/index_builder.py` | Crear índices PCA | `python index_builder.py` |
| `supervised/main.py` | Entrenar 8 modelos | `python main.py` |
| `unsupervised/main.py` | Clustering K-Means/DBSCAN | `python main.py` |
| `supervised/01_pipeline_completo.ipynb` | Notebook interactivo | `jupyter notebook` |
| `dashboard/app.py` | Visualización interactiva | `python app.py` |

---

## Equipo y Responsabilidades

| Integrante | Responsabilidad | Etapas |
|------------|-----------------|--------|
| **Santi** | Carga y limpieza de datos, modelos no supervisados y visualizaciones | load-data, unsupervised, dashboard|
| **Isabella** | EDA, análisis no supervisado + índices | eda, indexes-scores |
| **Juanes** | Simulación de datos, modelos supervisados + evaluación | supervised |

---

**Última actualización:** Mayo 2026  
**Estado del Proyecto:** ✅ **COMPLETADO EXITOSAMENTE**
