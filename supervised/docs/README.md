# feature/supervised-models -- Modelos Supervisados

Responsable: Juanes  
Rama: feature/supervised-models  
Objetivo: Implementar, entrenar, comparar y evaluar modelos supervisados para prediccion de tasa de desercion estudiantil.

---

## Descripcion General

Este modulo implementa un pipeline completo de modelado supervisado con:

- [SI] Carga y limpieza de datos automatica
- [SI] 8 modelos diferentes: 4 baseline + 4 avanzados
- [SI] Validacion cruzada 5-fold estandarizada
- [SI] Evaluacion en test set con metricas estandar
- [SI] Visualizaciones de diagnostico comprehensivas
- [SI] Comparacion detallada por departamento
- [SI] Feature importance para modelos de arbol
- [SI] Reportes y exportacion de resultados

---

## Estructura de Archivos

```
supervised/
├── main.py                              <- Punto de entrada (ejecutar este)
│
├── src/
│   ├── data_loader.py                  <- Carga y limpieza de datos
│   ├── Models.py                       <- Definicion y entrenamiento de 8 modelos
│   ├── Evaluation.py                   <- Metricas y visualizaciones
│   └── utils.py                        <- Funciones utilitarias
│
├── reports/                            <- Salidas generadas
│   ├── eda_figures/
│   │   ├── supervisado_predicho_vs_real.png
│   │   ├── supervisado_residuales.png
│   │   ├── supervisado_comparacion_metricas.png
│   │   ├── supervisado_comparacion_departamentos.png
│   │   ├── supervisado_importancia_RandomForest.png
│   │   └── supervisado_importancia_GradientBoosting.png
│   ├── tabla_comparativa_modelos_supervisados.csv
│   └── informe_supervisado.txt
│
└── docs/
    └── README.md                       <- Este archivo
```

---

## Uso Rapido

### Ejecucion Completa

```bash
# Desde la carpeta supervised/
python main.py
```

### Ejecucion en Jupyter Notebook

```bash
# Desde la carpeta supervised/
jupyter notebook 01_pipeline_completo.ipynb
```

Este comando ejecuta el pipeline completo:
1. Carga datos desde data/final/dataset_con_indice.csv
2. Realiza limpieza y preparacion
3. Entrena 8 modelos con validacion cruzada
4. Evalua en test set
5. Genera visualizaciones y reportes

---

## Modelos Implementados

### Baseline (4 modelos)

| Modelo | Parametros | Justificacion |
|--------|-----------|---------------|
| OLS | Ninguno | Baseline sin regularizacion, maximo interpretable |
| Ridge | a=1.0 | Regularizacion L2, maneja multicolinealidad |
| Lasso | a=0.1 | Regularizacion L1, seleccion automatica de variables |
| ElasticNet | a=0.1, l1_ratio=0.5 | Combinacion L1+L2, mejor en muchos casos |

Utilidad: Proporcionan puntos de referencia simples e interpretables.

### Avanzados (4 modelos)

| Modelo | Parametros | Justificacion |
|--------|-----------|---------------|
| Random Forest | n_estimators=100 | Ensemble robusto a outliers |
| Gradient Boosting | n_estimators=100, lr=0.1 | Boosting secuencial, tipicamente superior |
| SVR (RBF) | C=1.0, e=0.1 | Kernel no-lineal, bueno en espacios altos |
| KNN | n_neighbors=5 | Basado en instancias, captura patrones locales |

Utilidad: Capturan relaciones no-lineales complejas entre variables.

---

## Metricas de Evaluacion

Todos los modelos son evaluados con 4 metricas estandar:

| Metrica | Formula | Interpretacion | Unidad |
|---------|---------|----------------|--------|
| RMSE | sqrt(MSE) | Error cuadratico medio | % |
| MAE | Mean(|y - y_hat|) | Error promedio absoluto | % |
| R2 | 1 - (SS_res/SS_tot) | Varianza explicada | [0, 1] |
| MAPE | Mean(|dy/y|) x 100 | Error porcentual medio | % |

Metrica principal: RMSE (usado para ranking final)

---

## Protocolo de Validacion

```
+-----------------------------------------------------+
| DATOS COMPLETOS (66 obs, 2 años x 33 depart.)     |
+----------+------------------------------------------+
           | Eliminar nulos
           v
+--------------------------+
| DATOS LIMPIOS (50-55 obs)|
+----------+---------------+
           | Split 80/20
       +---+---+
       v       v
   TRAIN   TEST
   (40)    (10)
    |       |
    |   [EVALUACION FINAL]
    |
    v
  CV 5-fold
  (metricas reportadas)
```

Configuracion:
- Split: 80% train / 20% test
- Random state: 42 (reproducibilidad)
- CV: 5-fold, shuffle=True

---

## Ejemplo de Salida

```
=====================================================================
RANKING POR RMSE (Validacion Cruzada)
=====================================================================
  GradientBoosting     | RMSE: 3.1234 +/- 0.5678
  RandomForest         | RMSE: 3.4567 +/- 0.6789
  SVR                  | RMSE: 4.1234 +/- 0.7890
  KNN                  | RMSE: 4.5678 +/- 0.8901
  Ridge                | RMSE: 5.1234 +/- 0.9012
  ElasticNet           | RMSE: 5.3456 +/- 0.9123
  Lasso                | RMSE: 5.5678 +/- 0.9234
  OLS                  | RMSE: 5.6789 +/- 0.9345

=====================================================================
TABLA COMPARATIVA FINAL (Test Set)
=====================================================================
              RMSE      MAE      R2    MAPE
modelo
GradientBoosting  3.0521   2.4567  0.7234  12.34%
RandomForest      3.2143   2.6789  0.7012  13.45%
SVR               4.0876   3.1234  0.6789  14.56%
...
```

---

## Usar el Modulo en Codigo

```python
from src.data_loader import load_and_prepare
from src.Models import preparar_datos, entrenar_todos_modelos
from src.Evaluation import evaluar_todos, plot_predicho_vs_real

# 1. Cargar datos
X, y, loader = load_and_prepare()

# 2. Preparar split
X_train, X_test, y_train, y_test = preparar_datos(X, y)

# 3. Entrenar
modelos, cv_results = entrenar_todos_modelos(X_train, y_train)

# 4. Evaluar
metricas = evaluar_todos(modelos, X_test.values, y_test.values)

# 5. Visualizar
plot_predicho_vs_real(modelos, X_test.values, y_test.values)

# Ver modelo ganador
print(f"Mejor: {metricas.index[0]}")
```

---

## Configuracion de Hiperparametros

Los hiperparametros pueden ajustarse en src/Models.py:

```python
# Modelos baseline
Ridge(alpha=1.0)              # Aumentar para mayor regularizacion
Lasso(alpha=0.1)              # Aumentar para mas sparsidad

# Modelos avanzados
RandomForestRegressor(
    n_estimators=100,         # Mas arboles -> mejor pero mas lento
    max_depth=None,           # Limitar para evitar overfitting
    min_samples_split=5       # Mayor -> arboles mas simples
)

GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,        # Menor -> entrenamiento mas lento pero mejor
    max_depth=3               # Profundidad de cada arbol
)
```

---

## Interpretacion de Visualizaciones

### Predicho vs. Real
- Puntos sobre diagonal roja -> Prediccion correcta
- Dispersion grande -> Modelo con alta varianza
- Sesgo sistematico -> Error consistente en direccion

### Residuales
- Aleatorios alrededor de 0 -> Buen modelo
- Patron en embudo -> Heteroscedasticidad
- Puntos alejados -> Outliers problematicos

### Comparacion por Departamento
- Barras horizontales comparan Real vs Predicho por region.
- Permite identificar departamentos donde el modelo subestima o sobreestima la desercion.
- Util para focalizar politicas publicas regionales.

### Importancia de Variables
- Barras largas -> Variables clave para prediccion
- Top features -> Enfocarse en estas para interpretabilidad

---

## Variable Objetivo y Features

### Target
```
outcome_tasa_desercion_snies    Tasa de desercion estudiantil (%)
```

### Features Clave
```
Macroeconomicas:
  * geih_td_nacional_media_anual          Tasa de desempleo nacional
  * ipc_nacional_total_var_mensual_media  Inflacion nacional
  * pib_variacion_pct_anual_vs_anio_previo  Crecimiento PIB

Indices (Construccion Previa):
  * indice_vulnerabilidad_pca             PCA (44.8% varianza)
  * indice_vulnerabilidad_teorico         Teorico (literatura)

Educativas:
  * spadies_td_anual_tecnologico          Desercion tecnica
  * total_matriculados                    Acceso a educacion
  * proxy_pib_miles_mm_cop_por_matriculado  PIB por estudiante
```

---

## Checklist de Completitud

- [x] Carga automatica de datos
- [x] Limpieza de data leakage conocido
- [x] 4 modelos baseline
- [x] 4 modelos avanzados
- [x] Validacion cruzada 5-fold
- [x] Metricas estandar (RMSE, MAE, R2, MAPE)
- [x] Split train/test 80/20
- [x] Visualizaciones (scatter, residuales, comparacion por departamento)
- [x] Feature importance
- [x] Exportacion de resultados CSV
- [x] Documentacion completa
- [x] Codigo bien comentado

---

## Troubleshooting

### Error: "No se encontro dataset_con_indice.csv"
```python
# Especificar ruta manualmente
loader = DataLoader("../../data/final/dataset_con_indice.csv")
```

### Pocos datos para CV (< 30 muestras)
- Normal en datasets pequeños
- Usar k-fold bajo (k=3) si es necesario
- Considerar leave-one-out CV

### Modelo con R2 negativo
- Indica PEOR que baseline
- Revisar features y datos
- Considerar transformaciones

---

## Referencias Teoricas

### Validacion Cruzada
Estima performance real mediante folds multiples, evita overfitting.

### RMSE vs MAE
- RMSE penaliza mas errores grandes
- MAE es mas resistente a outliers

### Regularizacion (Ridge/Lasso)
- Ridge: Reduce magnitud coeficientes (L2)
- Lasso: Selecciona variables importantes (L1)

### Ensemble Methods
- Random Forest: Paralelo, robusto
- Gradient Boosting: Secuencial, tipicamente mejor

---

## Notas Finales

- Reproducibilidad: Random state = 42 en todos los splits
- Datos limitados: 50-55 muestras finales despues de limpieza
- Data leakage: Ya eliminado en EDA (spadies_td_anual_universitario)
- Multicolinealidad: Indices sinteticos capturan informacion economica
- Siguiente paso: Tuning de hiperparametros en modelo ganador

---

Actualizado: Mayo 11, 2026  
Responsable: Juanes  
Rama: feature/supervised-models
