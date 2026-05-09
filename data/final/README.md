# 📊 Dataset con Índice de Vulnerabilidad - Resumen del Análisis

**Archivo:** `dataset_con_indice.csv`  
**Dimensiones:** 66 filas × 37 columnas  
**Fecha generación:** 5 de mayo de 2026

---

## 🔄 Pipeline Ejecutado

El análisis completo fue generado por `src/index_builder.py` e incluye:

### 1. **Limpieza del Dataset** (según conclusiones del EDA)
- ✅ **Data Leakage Detectado:** `spadies_td_anual_universitario` era idéntico a `outcome_tasa_desercion_snies` → **ELIMINADO**
- ✅ **Columnas técnicas eliminadas:** `outcome_merge_pendiente`
- ✅ **Columnas con >55% nulos excluidas del análisis:**
  - `var_pct_matriculados_vs_anio_previo` (57.6%)
  - `spadies_td_anual_tecnico_profesional` (57.6%)
- ✅ **Identificadas 15 variables macroeconómicas nacionales** con baja discriminación entre departamentos

---

## 📐 Análisis de Dimensionalidad (PCA)

### Varianza Explicada:
- **PC1:** 44.8%
- **PC2:** 27.6%
- **Acumulada (2 componentes):** 72.4%

### Variables Utilizadas (7 en total):
1. `geih_td_nacional_media_anual` - Tasa de desempleo
2. `ipc_nacional_total_var_mensual_media` - Inflación
3. `proxy_pib_miles_mm_cop_por_matriculado` - Ingreso relativo
4. `total_matriculados` - Acceso a educación
5. `spadies_td_anual_tecnologico` - Deserción técnica
6. `pib_variacion_pct_anual_vs_anio_previo` - Crecimiento económico
7. `geih_to_nacional_media_anual` - Tasa de ocupación

### Cargas (Importancia de variables en PC1):
| Variable | Carga |
|----------|-------|
| proxy_pib_miles_mm_cop_por_matriculado | 0.7050 |
| spadies_td_anual_tecnologico | 0.6986 |
| total_matriculados | -0.1061 |
| pib_variacion_pct_anual_vs_anio_previo | -0.0608 |
| Otras (macroeconómicas) | ≈ 0 |

---

## 🔨 Índices Construidos

### 1. **Índice de Vulnerabilidad PCA**
- **Método:** Cargas de la primera componente principal (PC1)
- **Signo:** Ajustado para correlación positiva con outcome
- **Interpretación:** Valores altos = mayor vulnerabilidad
- **Componente:** 44.8% de la varianza explicada

### 2. **Índice de Vulnerabilidad Teórico**
- **Método:** Suma ponderada con pesos definidos por literatura económica
- **Pesos normalizados:**
  - Desempleo (+0.250): Aumenta vulnerabilidad
  - Inflación (+0.150): Aumenta vulnerabilidad
  - Ingreso relativo (-0.200): Disminuye vulnerabilidad
  - Acceso a educación (-0.100): Disminuye vulnerabilidad
  - Deserción técnica (+0.150): Correlación positiva
  - Crecimiento económico (-0.100): Disminuye vulnerabilidad
  - Ocupación (-0.150): Disminuye vulnerabilidad

- **Correlación con Outcome:** ρ=0.164 (p=0.3625) - Débil pero en dirección positiva

---

## 🎯 Clustering (Análisis no supervisado)

### K-Means
| k | Silhouette | Calinski-Harabasz | Inercia |
|---|-----------|-------------------|---------|
| 2 | **0.722** | 16.66 | 64.81 |
| 3 | 0.682 | 21.37 | 38.84 |
| 4 | 0.602 | 47.75 | 14.94 |
| 5 | 0.550 | 51.54 | 10.41 |
| 6 | 0.530 | 65.66 | 6.49 |
| 7 | 0.304 | 71.67 | 4.80 |

**k óptimo seleccionado: 2** (máximo silhouette score = 0.722)

### Clustering Jerárquico
- **Método:** Aglomerativo con linkage Ward
- **Número de clusters:** 2 (igual a k óptimo de K-Means)
- **Métrica:** Distancia euclidiana

---

## ✔️ Validación de Etiquetas (Adjusted Rand Index)

### Resultados:
- **ARI K-Means:** -0.002 → ❌ **Poca correspondencia entre clusters y outcome**
- **ARI Jerárquico:** -0.002 → ❌ **Poca correspondencia entre clusters y outcome**

### Interpretación:
Los clusters generados por el análisis no supervisado **NO coinciden bien** con los niveles de deserción observados. Esto sugiere:
- La estructura de vulnerabilidad económica no separa naturalmente a los departamentos según su tasa de deserción
- Pueden existir otros factores no capturados en el índice que expliquen mejor la deserción
- La relación entre variables económicas y deserción es más compleja que la capturada por clustering simple

---

## 📊 Columnas Nuevas Agregadas

| Columna | Descripción | Rango | Notas |
|---------|-------------|-------|-------|
| `indice_vulnerabilidad_pca` | Índice PCA (Primera componente) | [-2.5, 2.5] | Normalizado |
| `indice_vulnerabilidad_teorico` | Índice teórico (Suma ponderada) | [-1.5, 1.5] | Estandarizado |
| `cluster_km` | Etiqueta K-Means | {0, 1} | 2 clusters |
| `cluster_hierarchical` | Etiqueta jerárquica | {0, 1} | 2 clusters |
| `pc1` | Componente Principal 1 | [-0.64, 0.88] | Normalizado |
| `pc2` | Componente Principal 2 | [-0.62, 0.82] | Normalizado |

---

## 📈 Visualizaciones Generadas

Guardadas en `reports/figures/`:

1. **01_scree_plot.png**
   - Varianza explicada acumulativa por componentes PCA
   - Muestra que 2 componentes capturan 72.4% de la varianza

2. **01b_biplot_pca.png**
   - PC1 vs PC2 con observaciones coloreadas por outcome
   - Flechas indican contribución de variables al biplot

3. **02_indice_pca_vs_outcome.png**
   - Scatter: Índice PCA (eje X) vs Tasa de Deserción (eje Y)
   - Departamentos etiquetados

4. **03_indice_teorico_vs_outcome.png**
   - Scatter: Índice Teórico (eje X) vs Tasa de Deserción (eje Y)
   - Departamentos etiquetados

5. **03b_comparacion_indices.png**
   - Scatter: Índice PCA (eje X) vs Índice Teórico (eje Y)
   - Relación entre ambos enfoques

6. **04_clusters_vs_outcome.png**
   - Scatter: Cluster (eje X) vs Tasa de Deserción (eje Y)
   - Puntos coloreados según cluster asignado

---

## ✅ Checklist de Entregables (README)

- [x] **Varianza explicada por PCA documentada** → 44.8% (PC1), 72.4% (acumulada)
- [x] **Al menos 2 algoritmos de clustering comparados** → K-Means y Jerárquico
- [x] **Métricas de clustering documentadas** → Silhouette, Calinski-Harabasz, Inercia
- [x] **Validación de etiquetas (ARI)** → Ambos algoritmos con ARI = -0.002
- [x] **Índice construido y correlacionado** → PCA e Índice Teórico
- [x] **index_builder.py funcional** → Ejecutable y completo
- [x] **Visualizaciones exportadas** → 6 gráficas en `reports/figures/`
- [x] **Columnas requeridas agregadas** → `indice_*`, `cluster_*`, `pc1`, `pc2`

---

## 🚀 Próximos Pasos

1. **Supervisado (Juanes):** Usar `dataset_con_indice.csv` con:
   - Features: Todas las variables (excepto las eliminadas por leakage/nulos)
   - Target: `outcome_tasa_desercion_snies`
   - Índices construidos como features adicionales opcionales

2. **Mejoras futuras:**
   - Explorar técnicas de imputación para variables con nulos (MICE, KNN)
   - Incluir características no numéricas (región geográfica, tipo de institución)
   - Modelos de regresión con regularización L2 (dado n=33)
   - Análisis temporal si se agrega más años de datos

---

## 📝 Notas Finales

- **Limitación:** Solo 33 observaciones con target (año 2023)
- **Recomendación EDA:** Excluir variables con >50% nulos del modelado supervisado
- **Data Leakage:** Variable `spadies_td_anual_universitario` eliminada correctamente
- **Estructura:** Panel de 66 observaciones (33 depts × 2 años) con target solo para 2023
