# Conclusiones — Análisis No Supervisado

## Resumen del pipeline

La etapa no supervisada del proyecto aplicó clustering sobre el dataset final con tres algoritmos:

- `KMeans` con selección de `k` óptimo
- `DBSCAN` con búsqueda de `eps` y `min_samples`
- `AgglomerativeClustering` con el mismo número de clusters de K-Means

Los resultados se guardaron en:

- `unsupervised/reports/eda_figures/unsupervised_kmeans_pca.png`
- `unsupervised/reports/eda_figures/unsupervised_dbscan_pca.png`
- `unsupervised/reports/eda_figures/unsupervised_agglomerative_pca.png`
- `unsupervised/reports/cluster_assignments_KMeans.csv`
- `unsupervised/reports/cluster_assignments_DBSCAN.csv`
- `unsupervised/reports/cluster_assignments_Agglomerative.csv`
- `unsupervised/reports/unsupervised_validation_summary.csv`

## Resultados principales

### K-Means

- Número de clusters: 3
- Distribución de clusters:
  - Cluster 0: 32 observaciones
  - Cluster 1: 32 observaciones
  - Cluster 2: 2 observaciones
- Interpretación: K-Means encontró dos grupos grandes y un pequeño grupo residual.

### Agglomerative

- Número de clusters: 3
- Distribución de clusters:
  - Cluster 0: 32 observaciones
  - Cluster 1: 32 observaciones
  - Cluster 2: 2 observaciones
- Interpretación: El clustering jerárquico produjo una segmentación casi idéntica a K-Means, lo que sugiere una estructura estable de tres grupos en el espacio de features escaladas.

### DBSCAN

- Número de clusters identificados: 2
- Etiquetas de ruido: 59 observaciones
- Asignaciones de cluster:
  - Cluster 0: 3 observaciones
  - Cluster 1: 4 observaciones
  - Ruido (-1): 59 observaciones
- Interpretación: El modelo de densidad detectó muy pocos grupos densos y clasificó la mayoría del dataset como ruido, lo que indica que la distribución actual de las variables no es fácilmente separable con el umbral de densidad usado.

## Validación y correspondencia con la variable objetivo

- `KMeans` ARI: 0.0014
- `Agglomerative` ARI: 0.0014
- `DBSCAN` ARI: 1.0000

### Observaciones de validación

- Los valores ARI muy bajos para K-Means y Agglomerative muestran que los grupos encontrados no coinciden significativamente con las categorías derivadas del target.
- El ARI perfecto para DBSCAN sugiere que el conjunto reducido de observaciones no-ruido coincide exactamente con las etiquetas cuando se compara con la variable objetivo, pero este resultado debe interpretarse con cuidado debido al alto porcentaje de puntos tratados como ruido.

## Interpretación general

- La presencia de dos clusters grandes y uno pequeño en K-Means y Agglomerative sugiere que existen patrones globales fuertes en el dataset.
- DBSCAN evidencia que la estructura local de densidad es débil, ya que muchas observaciones no parecen pertenecer a grupos densos bien definidos.
- Es probable que la variabilidad de los indicadores macro y socioeconómicos genere grupos amplios más que clusters densos y bien separados.

## Limitaciones

1. **Alta proporción de ruido en DBSCAN**
   - 59 de 66 observaciones fueron etiquetadas como ruido.
   - Esto limita el valor de DBSCAN en el contexto actual.

2. **Cluster pequeño en K-Means/Agglomerative**
   - Solo 2 observaciones forman el tercer grupo.
   - Puede ser un outlier o un segmento muy específico.

3. **ARI bajo para los modelos de partición**
   - Los clusters no reflejan el target de deserción en términos de categorías derivadas.

## Recomendaciones

- Usar `KMeans` y `Agglomerative` como herramientas exploratorias iniciales, pero no como segmentación final sin más refinamiento.
- Explorar nuevas combinaciones de features y transformaciones antes de repetir DBSCAN.
- Evaluar si el clustering es útil para análisis de grupos de departamentos o si el dataset requiere más variables relevantes.
- Considerar técnicas de reducción de dimensionalidad más avanzadas (como UMAP) para analizar visualmente la separación de clusters.

