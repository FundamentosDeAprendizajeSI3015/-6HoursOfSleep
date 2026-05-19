# Conclusiones — Análisis No Supervisado

## Objetivo del módulo

El objetivo de esta etapa no supervisada es apoyar la predicción de la tasa de deserción estudiantil en Colombia. Este análisis busca explorar patrones en los indicadores socioeconómicos y educativos, identificar grupos de departamentos con características similares y validar si esa estructura de datos es consistente con la variable objetivo de deserción.

## Resultados principales

### K-Means

- Número de clusters: 3
- Distribución de clusters:
  - Cluster 0: 32 observaciones
  - Cluster 1: 32 observaciones
  - Cluster 2: 2 observaciones
- Interpretación: K-Means encontró dos grupos grandes y un grupo muy reducido. Esto sugiere una segmentación global en los datos, pero no implica una predicción directa de la tasa de deserción.

### Agglomerative Clustering

- Número de clusters: 3
- Distribución de clusters:
  - Cluster 0: 32 observaciones
  - Cluster 1: 32 observaciones
  - Cluster 2: 2 observaciones
- Interpretación: El clustering jerárquico coincide casi exactamente con K-Means, lo que refuerza la existencia de una estructura estable de tres grupos en el espacio de features escaladas.

### DBSCAN

- Número de clusters identificados: 2
- Etiquetas de ruido: 59 observaciones
- Asignaciones de cluster:
  - Cluster 0: 3 observaciones
  - Cluster 1: 4 observaciones
  - Ruido (-1): 59 observaciones
- Interpretación: DBSCAN detecta muy pocos núcleos densos y clasifica la mayoría de las observaciones como ruido. En este caso, DBSCAN no es una buena opción para segmentar el dataset completo.

## Relación con la tasa de deserción

- `KMeans` ARI: 0.0014
- `Agglomerative` ARI: 0.0014
- `DBSCAN` ARI: 1.0000

### Qué indican estos valores

- KMeans y Agglomerative muestran una coincidencia muy baja con las categorías derivadas del target de deserción. Esto significa que los grupos encontrados no están alineados directamente con la tasa de deserción estudiantil.
- DBSCAN tiene un ARI perfecto solo sobre el subconjunto no-ruido (7 observaciones), por lo que ese resultado no respalda una relación general entre los clusters y la variable objetivo.

## Conclusión principal

El análisis no supervisado aporta un diagnóstico útil, pero no sustituye el modelado supervisado para predecir la tasa de deserción.

- KMeans y Agglomerative son valiosos para explorar grupos de departamentos similares en términos de indicadores.
- Sin embargo, la baja correspondencia con el target sugiere que, para predecir deserción, se requiere un enfoque supervisado que use explícitamente la variable objetivo.
- DBSCAN no es apropiado como método de segmentación general en este conjunto de datos debido a la alta proporción de ruido.

## Lecciones claves

1. **Clustering como herramienta exploratoria**
   - El clustering ayuda a encontrar patrones estructurales, no a predecir directamente la deserción.

2. **Estructura de datos vs. objetivo**
   - La estructura que mejor capturan KMeans y Agglomerative no está estrechamente ligada a la tasa de deserción.

3. **DBSCAN debe usarse con precaución**
   - En este dataset, la mayoría de los puntos se consideran ruido, por lo que el método no aporta segmentaciones útiles para el objetivo principal.

## Recomendaciones para el proyecto

- Mantener el pipeline supervisado como la solución principal para predecir tasa de deserción.
- Usar los clusters de KMeans/Agglomerative como variables de apoyo o para análisis de grupos similares.
- No basar decisiones de predicción únicamente en DBSCAN.
- Evaluar el valor de los grupos detectados para la ingeniería de features y la segmentación de departamentos.

## Próximos pasos en el contexto de la predicción de deserción

1. Usar los clusters como nuevas variables en el modelo supervisado.
2. Probar si la etiqueta de cluster mejora el rendimiento del modelo de deserción.
3. Generar variables derivadas adicionales que capturen mejor la heterogeneidad regional.
4. Revisar los departamentos del cluster pequeño para ver si representan casos atípicos con alto riesgo.
5. Combinar este análisis exploratorio con el pipeline supervisado para obtener predicciones más robustas.

---

*Documento actualizado con base en los resultados del pipeline no supervisado y el objetivo de predecir la tasa de deserción estudiantil en Colombia.*
