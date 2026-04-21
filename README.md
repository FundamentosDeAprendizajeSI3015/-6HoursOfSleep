# 🟣 feature/unsupervised-index — Isabela

> Rama a cargo de: **Isabela**
> Objetivo: Análisis no supervisado para validar etiquetas + construcción del índice de vulnerabilidad económica-educativa como combinación lineal de variables.

---

## 📁 Archivos de esta rama

```
notebooks/02_unsupervised_indice/
│
├── 01_reduccion_dimensiones.ipynb   ← PCA, t-SNE, UMAP
├── 02_clustering.ipynb              ← K-Means, DBSCAN, jerárquico
├── 03_validacion_etiquetas.ipynb    ← Evaluar si la deserción es una buena etiqueta
├── 04_construccion_indice.ipynb     ← Índice compuesto (combinación lineal)
└── README.md                        ← Este archivo

src/
└── index_builder.py                 ← Funciones para construir el índice
```

---

## 🎯 Responsabilidades

### 1. Reducción de dimensiones (`01_reduccion_dimensiones.ipynb`)

Técnicas a aplicar:

| Técnica | Uso | Librería |
|---------|-----|----------|
| **PCA** | Dimensiones altas → bajas, interpretable | `sklearn` |
| **t-SNE** | Visualización 2D de clusters | `sklearn` |
| **UMAP** | Visualización + preserva estructura global | `umap-learn` |

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=0.95)  # Retener 95% de varianza
X_pca = pca.fit_transform(X)
print(f"Componentes necesarios: {pca.n_components_}")
print(f"Varianza explicada: {pca.explained_variance_ratio_.cumsum()}")
```

**Reportar:**
- Varianza explicada por componente (scree plot)
- Biplot de las dos primeras componentes
- Qué variables tienen más peso en cada componente

### 2. Clustering (`02_clustering.ipynb`)

Algoritmos a implementar:

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# K-Means con selección de k óptimo
inertias = []
silhouettes = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
```

**Métricas de evaluación de clustering:**
- Silhouette Score (entre -1 y 1, mayor = mejor)
- Calinski-Harabasz Index
- Davies-Bouldin Score
- Método del codo (elbow method)

### 3. Validación de etiquetas (`03_validacion_etiquetas.ipynb`)

> ⚠️ **Esto es crítico para justificar el enfoque supervisado**

Preguntas a responder:
- ¿Los clusters no supervisados coinciden con niveles de deserción?
- ¿La variable `desercion` separa bien los grupos?
- ¿Hay grupos naturales que no estamos capturando?

```python
# Comparar clusters con niveles de deserción
from sklearn.metrics import adjusted_rand_score

# Discretizar deserción en cuartiles
df["desercion_cat"] = pd.qcut(df["desercion"], q=4,
                               labels=["Baja", "Media-Baja", "Media-Alta", "Alta"])

# Calcular ARI entre clusters y categorías de deserción
ari = adjusted_rand_score(df["desercion_cat"], df["cluster"])
print(f"Adjusted Rand Index: {ari:.3f}")
# ARI > 0.3 sugiere que la etiqueta es informativa
```

### 4. Construcción del índice (`04_construccion_indice.ipynb`)

**Objetivo:** Crear `indice_vulnerabilidad` como combinación lineal de variables económicas.

Dos enfoques a comparar:

#### Enfoque A — Pesos por PCA (automático)
```python
# Los pesos son las cargas de la primera componente principal
pca = PCA(n_components=1)
pca.fit(X_scaled)
pesos = pca.components_[0]
indice_pca = X_scaled @ pesos
```

#### Enfoque B — Pesos teóricos (manual, justificado)
```python
# Pesos definidos por teoría económica y revisión de literatura
pesos_teoricos = {
    "desempleo":    0.30,
    "pobreza":      0.25,
    "inflacion":    0.15,
    "tasa_interes": 0.15,
    "ingreso_real": -0.15  # negativo: más ingreso = menos vulnerabilidad
}
```

**Comparar ambos índices** con la variable `desercion` usando correlación de Spearman.

---

## 📊 Visualizaciones requeridas

- [ ] Scree plot de PCA
- [ ] Biplot PC1 vs PC2 coloreado por nivel de deserción
- [ ] Mapa de clusters por departamento (si hay datos geográficos)
- [ ] Dendrograma del clustering jerárquico
- [ ] Scatter: índice construido vs. tasa de deserción real
- [ ] Comparación de índice PCA vs. índice teórico

---

## 📤 Entregable al pipeline

Al final de esta rama, agregar al dataset:

```
data/processed/dataset_con_indice.csv
```

Con las columnas adicionales:
- `indice_vulnerabilidad_pca` — índice por cargas de PCA
- `indice_vulnerabilidad_teorico` — índice por pesos teóricos
- `cluster_km` — etiqueta de cluster K-Means
- `pc1`, `pc2` — primeras dos componentes principales

---

## ✅ Checklist antes de hacer merge

- [ ] Varianza explicada por PCA documentada
- [ ] Al menos 2 algoritmos de clustering comparados con métricas
- [ ] Validación de etiquetas con ARI o métrica equivalente
- [ ] Índice construido y correlacionado con `desercion`
- [ ] `index_builder.py` funciona al importar
- [ ] Visualizaciones exportadas a `reports/figures/`

---

## 🔗 Dependencias

- **Requiere:** `data/processed/dataset_final.csv` de Santi
- **Provee a Juanes:** `data/processed/dataset_con_indice.csv` con el índice como variable adicional
