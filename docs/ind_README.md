# Creación de índices — Isabella

> **Rama/Enfoque:** Creación de índices a partir de análisis no supervisado y análisis teórico.
> **Objetivo:** Análisis no supervisado para validar etiquetas + construcción del índice de vulnerabilidad económica-educativa como combinación lineal de variables.

---

## 🎯 Objetivo Alcanzado

Implementar el pipeline completo descrito en [docs/ind_README.md](../docs/ind_README.md):

- [x] Reducción de dimensiones (PCA)
- [x] Clustering (K-Means + Jerárquico)
- [x] Validación de etiquetas (ARI)
- [x] Construcción de índices (PCA + Teórico)
- [x] Visualizaciones exhaustivas
- [x] Exportación a `data/final/`

---

## 📦 Entregables

### 1. **Dataset Procesado**
```
data/final/dataset_con_indice.csv
├── 66 filas (33 departamentos × 2 años)
├── 37 columnas (31 originales + 6 nuevas)
└── Variables originales limpias según EDA

Columnas nuevas agregadas:
  • indice_vulnerabilidad_pca          ← Índice automático (PCA)
  • indice_vulnerabilidad_teorico      ← Índice por pesos teóricos
  • cluster_km                         ← Etiquetas K-Means (k=2)
  • cluster_hierarchical               ← Etiquetas Jerárquico (k=2)
  • pc1, pc2                          ← Componentes principales
```

### 2. **Visualizaciones** (en `reports/figures/`)
```
01_scree_plot.png                     ← Varianza explicada acumulada
01b_biplot_pca.png                   ← PC1 vs PC2 con cargas de variables
02_indice_pca_vs_outcome.png          ← Índice PCA vs tasa de deserción
03_indice_teorico_vs_outcome.png      ← Índice teórico vs outcome
03b_comparacion_indices.png           ← Comparación PCA vs Teórico
04_clusters_vs_outcome.png            ← Clusters vs tasa de deserción
```

### 3. **Código Mejorado**
```
src/index_builder.py
├── ✅ Limpieza automática según EDA
├── ✅ Identificación dinámica de variables
├── ✅ PCA con varianza documentada
├── ✅ Dos algoritmos de clustering
├── ✅ Validación de etiquetas (ARI)
├── ✅ Índices teórico y automático
├── ✅ 6 visualizaciones automáticas
└── ✅ Pipeline ejecutable y modular
```

### 4. **Documentación**
```
data/final/README.md                  ← Resumen técnico del análisis
                                        (este archivo)
```

---

## 📊 Resultados Clave

### **Limpieza del Dataset**

| Acción | Detalles |
|--------|----------|
| **Data Leakage** | `spadies_td_anual_universitario` ≡ `outcome_tasa_desercion_snies` → **ELIMINADO** |
| **Columnas técnicas** | `outcome_merge_pendiente` → **ELIMINADO** |
| **Nulos altos (>55%)** | 2 variables excluidas del análisis de índice |
| **Macroeconómicas** | 15 variables identificadas como de baja discriminación |
| **Resultado** | 31 columnas limpias, 7 para construcción de índice |

### **PCA - Reducción de Dimensiones**

```
Componentes principales (7 variables → 2 PCs):
  PC1:  44.8% de varianza
  PC2:  27.6% de varianza
  Total: 72.4% ✅

Cargas principales en PC1:
  • proxy_pib_miles_mm_cop_por_matriculado:  0.705 (PIB por estudiante)
  • spadies_td_anual_tecnologico:           0.699 (Deserción técnica)
  • total_matriculados:                    -0.106 (Acceso a educación)
```

### **Índices Construidos**

#### **Índice PCA** (Automático)
- Basado en cargas de PC1
- Captura 44.8% de la varianza en 7 dimensiones
- Correlación con outcome: `ρ=0.387` (promedio de índices)

#### **Índice Teórico** (Literatura Económica)
- Suma ponderada con pesos justificados
- Pesos: Desempleo (0.25) → Ingreso (-0.20) → Ocupación (-0.15)
- Correlación Spearman con outcome: `ρ=0.164` (p=0.3625)

### **Clustering: Validación**

| Métrica | K-Means | Jerárquico |
|---------|---------|-----------|
| **k óptimo** | 2 | 2 |
| **Silhouette** | 0.722 ⭐ | - |
| **Calinski-Harabasz** | 16.66 | - |
| **ARI vs outcome** | -0.002 ❌ | -0.002 ❌ |
| **Conclusión** | Poca correspondencia con deserción | Poca correspondencia |

### **Interpretación**
✅ El clustering no supervisado obtiene **excelentes métricas internas** (silhouette=0.722)  
❌ PERO **NO coincide** con los niveles de deserción (ARI ≈ 0)

**Implicación:** La estructura económica de vulnerabilidad no separa naturalmente a los departamentos según su tasa de deserción. Otros factores pueden ser más importantes.

---

## 🔧 Funciones Disponibles (Para uso posterior)

```python
# Importar funciones individuales
from src.index_builder import (
    limpiar_dataset,                    # Limpieza automática
    identificar_variables_indice,       # Selección de variables
    analizar_pca,                       # PCA
    construir_indice_pca,               # Índice automático
    construir_indice_teorico,           # Índice teórico
    clustering_kmeans,                  # K-Means
    clustering_jerarquico,              # Clustering jerárquico
    validar_etiquetas,                  # ARI
    generar_visualizaciones,            # Gráficas
    pipeline_completo                   # Ejecutar todo
)

# O ejecutar el pipeline completo
df_final = pipeline_completo()
```

---

## ✅ Checklist del README Completado

| Item | Estado | Detalles |
|------|--------|----------|
| 1. Varianza explicada documentada | ✅ | PC1: 44.8%, PC2: 27.6%, Total: 72.4% |
| 2. 2+ algoritmos de clustering | ✅ | K-Means (k=2) + Jerárquico (k=2) |
| 3. Validación de etiquetas (ARI) | ✅ | ARI = -0.002 para ambos |
| 4. Índice construido | ✅ | PCA + Teórico + Correlación documentada |
| 5. index_builder.py importable | ✅ | 19 funciones disponibles |
| 6. Visualizaciones en reports/ | ✅ | 6 gráficas (scree, biplot, índices, clusters) |
| 7. Columnas requeridas agregadas | ✅ | indice_*, cluster_*, pc1, pc2 |
| 8. Dataset exportado a final/ | ✅ | dataset_con_indice.csv (66×37) |

---

## 📈 Paso Siguiente (Para Juanes - Modelado Supervisado)

El dataset está listo para modelado:

```python
# Cargar
df = pd.read_csv('data/final/dataset_con_indice.csv')

# Features disponibles:
# • Todas las variables económicas limpias
# • Índices construidos (indice_vulnerabilidad_pca, indice_vulnerabilidad_teorico)
# • Componentes PCA (pc1, pc2)
# • Clusters (cluster_km, cluster_hierarchical)

# Target:
# • outcome_tasa_desercion_snies (solo 33 observaciones con valor)

# ⚠️ Cuidado:
# • n=33 observaciones con target → Considerar regularización (Ridge/Lasso)
# • Variables con nulos → Estrategia de imputación
# • Desbalance: n<<p potencial → Feature selection
```

---

## 📝 Notas Técnicas

- **Variables de entrada:** 7 seleccionadas (descartadas 24 por alta cardinalidad o nulos)
- **Escalado:** StandardScaler en todas las etapas
- **Random state:** Fijo (42) para reproducibilidad
- **Métodos de clustering:** Ward linkage para jerárquico, k-means++ init
- **Validación:** Silhouette, Calinski-Harabasz, Inercia, ARI

---

## 🎯 Conclusiones Principales

1. **Data Leakage:** Correctamente identificado y eliminado ✅
2. **Dimensionalidad:** 7 variables → 2 PCs capturan 72.4% ✅
3. **Clustering interno:** Excelente (silhouette=0.722) ✅
4. **Correspondencia con target:** Débil (ARI≈0) ⚠️
5. **Índices construidos:** Listos para supervisado ✅

La **baja correspondencia de clusters con outcome** sugiere que:
- La vulnerabilidad económica de un departamento **no es el único predictor** de deserción
- Factores adicionales (institucionales, sociales, culturales) probablemente importan más
- Apropiado pasar a **modelado supervisado** donde el outcome guía el aprendizaje

---

**Documento generado automáticamente por `src/index_builder.py`**
