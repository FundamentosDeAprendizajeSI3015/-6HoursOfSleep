# Creación de Índices - Dataset Simulado

> **Rama/Enfoque:** Creación de índices a partir de análisis no supervisado y análisis teórico - Dataset Simulado
> **Objetivo:** Análisis no supervisado para validar etiquetas + construcción del índice de vulnerabilidad económica-educativa utilizando el dataset simulado 1980-2026.

---

## Proceso Implementado

Se aplicó el **mismo pipeline descrito en [ind_README.md](ind_README.md)** al dataset simulado:

- [x] Reducción de dimensiones (PCA)
- [x] Clustering (K-Means + Jerárquico)
- [x] Validación de etiquetas (ARI)
- [x] Construcción de índices (PCA + Teórico)
- [x] Visualizaciones exhaustivas
- [x] Exportación a `data/final/`

Por detalles completos del metodología, limpieza de variables, y arquitectura del código, consultar [ind_README.md](ind_README.md).

---

## Hallazgos Principales - Dataset Simulado

### **1. Dimensiones del Dataset Procesado**

| Métrica | Valor |
|---------|-------|
| **Filas originales** | 18,612 |
| **Filas finales** | 18,612 |
| **Columnas finales** | 39 |
| **Retención de datos** | 100.0% |
| **Porcentaje nulos** | 0.38% |

El dataset simulado amplía significativamente la cobertura temporal respecto al original, manteniendo integridad completa de datos.

---

### **2. Análisis PCA - Reducción de Dimensiones**

| Métrica | Valor |
|---------|-------|
| **Componentes analizadas** | 2 |
| **Varianza explicada PC1** | 31.24% |
| **Varianza explicada PC2** | 19.96% |
| **Varianza acumulada (PC1+PC2)** | **51.20%** |

**Interpretación:** Con 2 componentes principales se captura el 51.20% de la varianza en 7 dimensiones de vulnerabilidad. Este es un poder explicativo moderado que sugiere la existencia de múltiples factores de riesgo independientes.

---

### **3. Variables Utilizadas en el Índice**

Se seleccionaron **7 variables** para la construcción de índices:

1. `geih_td_nacional_media_anual` - Tasa de desempleo nacional
2. `ipc_nacional_total_var_mensual_media` - Inflación nacional
3. `proxy_pib_miles_mm_cop_por_matriculado` - PIB por estudiante
4. `total_matriculados` - Volumen de matrícula
5. `spadies_td_anual_tecnologico` - Deserción técnica
6. `pib_variacion_pct_anual_vs_anio_previo` - Crecimiento económico
7. `geih_to_nacional_media_anual` - Tasa de ocupación nacional

---

### **4. Índices Construidos - Correlaciones con Target**

#### **Índice PCA** (Automático)

| Métrica | Valor |
|---------|-------|
| **Correlación Spearman (ρ)** | **0.6902** |
| **P-valor** | < 0.000001 ✓ |
| **Observaciones** | 18,612 |
| **Significancia** | **Altamente significativo** |

**Interpretación:** El índice PCA muestra una correlación **moderada-fuerte** y estadísticamente significativa con la tasa de deserción. Esto sugiere que la vulnerabilidad económica-educativa capturada por PCA es un **predictor útil** de deserción.

#### **Índice Teórico** (Basado en Literatura)

| Métrica | Valor |
|---------|-------|
| **Correlación Spearman (ρ)** | **0.6287** |
| **P-valor** | < 0.000001 ✓ |
| **Observaciones** | 18,612 |
| **Significancia** | **Altamente significativo** |

**Interpretación:** El índice teórico, construido con pesos justificados en la literatura económica, también presenta correlación **moderada-fuerte** y significativa. Aunque menor que el PCA, valida la consistencia del enfoque teórico.

---

### **5. Clustering - Validación de Estructura**

#### **K-Means**

| Métrica | Valor |
|---------|-------|
| **K óptimo** | 3 |
| **Rango de k probado** | 2-7 |
| **Silhouette Score** | 0.2566 |
| **Calinski-Harabasz** | 4654.92 |
| **Inercia** | 84,380.45 |
| **ARI (vs outcome)** | 0.1195 |

#### **Clustering Jerárquico**

| Métrica | Valor |
|---------|-------|
| **ARI (vs outcome)** | 0.0465 |

**Interpretación:**
- K-Means obtiene k=3 como óptimo, con silhouette moderado (0.2566)
- La **coincidencia con outcome es baja** (ARI = 0.12 para K-Means)
- Los clusters económicos NO separan naturalmente por nivel de deserción
- **Implicación:** La estructura económica es insuficiente para predecir deserción; se requieren features adicionales

---

## Comparativa: Dataset Original vs. Dataset Simulado

| Aspecto | Original (n=66) | Simulado (n=18,612) | Cambio |
|---------|-----------------|-------------------|--------|
| **Filas para índice** | 66 | 18,612 | **282× más** |
| **Varianza PCA (2 comps)** | Comparable* | 51.20% | Mayor capacidad explicativa |
| **Correlación índice-outcome** | No reportada | 0.6902 (PCA) | **Altamente significativa** |
| **K óptimo** | 2 | 3 | Estructura más compleja |
| **ARI** | -0.002 | 0.1195 | Mejora marginal con más datos |

*En el original, n era muy pequeño; en simulado se obtiene estimación más robusta

---

## Entregables

### **Dataset Procesado**
```
data/final/dataset_con_indice_simulado.csv
├── 18,612 filas (100% retención)
├── 39 columnas (32 originales + 7 nuevas)
└── Variables originales limpias según EDA

Columnas nuevas:
  • indice_vulnerabilidad_pca          ← Índice automático (r=0.6902 con outcome)
  • indice_vulnerabilidad_teorico      ← Índice teórico (r=0.6287 con outcome)
  • cluster_km                         ← Etiquetas K-Means (k=3)
  • cluster_hierarchical               ← Etiquetas Jerárquico
  • pc1, pc2                          ← Componentes principales
```

### **Visualizaciones** (en `reports/dataset_simulado/`)
```
01_scree_plot.png                     ← Varianza explicada acumulada
01b_biplot_pca.png                   ← PC1 vs PC2 con cargas de variables
02_indice_pca_vs_outcome.png          ← Índice PCA vs tasa de deserción
03_indice_teorico_vs_outcome.png      ← Índice teórico vs outcome
03b_comparacion_indices.png           ← Comparación PCA vs Teórico
04_clusters_vs_outcome.png            ← Clusters vs tasa de deserción
```

---

## Conclusiones Principales

### **✅ Mejoras vs. Dataset Original**

1. **Robustez de estimaciones:** 282× más observaciones permiten estimaciones confiables de correlaciones y componentes
2. **Significancia estadística:** Ambos índices alcanzan p < 0.000001, muy por debajo del threshold de 0.05
3. **Poder predictivo:** Correlaciones de 0.629-0.690 indican que vulnerabilidad económica **sí predice** deserción
4. **Complejidad estructural:** K óptimo = 3 (vs. 2 en original) sugiere mayor heterogeneidad en datos

### **⚠️ Limitaciones Persistentes**

1. **Bajo acuerdo entre clusters y outcome:** ARI = 0.12 indica que clusters económicos no alinean bien con deserción
2. **Varianza parcialmente explicada:** PC1+PC2 solo capturan 51.2%, sugiriendo múltiples dimensiones de vulnerabilidad
3. **Índices como proxy:** Aunque significativos, estos índices explican ~ 48% de la varianza en deserción (R² ≈ 0.476 para PCA)