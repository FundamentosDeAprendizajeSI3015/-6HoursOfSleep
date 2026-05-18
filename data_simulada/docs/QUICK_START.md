"""
QUICK_START.md - Guía Rápida del Módulo de Datos
==================================================
"""

# 🚀 Quick Start - Generación de Datos

## ⚡ En 3 Pasos

### 1️⃣ Generar Dataset

```bash
cd supervised/data
python init_data.py
```

**Resultado:**
- `data_simulado_1980_2024.csv` (1,485 registros)
- `resumen_simulacion.txt` (documentación)

### 2️⃣ Cargar en Python

```python
import pandas as pd

# Cargar dataset completo
df = pd.read_csv('supervised/data/data_simulado_1980_2024.csv')

# Ver información
print(f"Registros: {len(df)}")
print(f"Período: {df['anio'].min()}-{df['anio'].max()}")
print(f"Departamentos: {df['codigo_departamento'].nunique()}")

# Filtrar año específico
df_2024 = df[df['anio'] == 2024]
df_reciente = df[df['anio'] >= 2010]
```

### 3️⃣ Usar en Pipeline

```python
# En supervised/main.py
from pathlib import Path

# Cargar datos simulados
data_path = Path(__file__).parent / "data" / "data_simulado_1980_2024.csv"
df = pd.read_csv(data_path)

# Usar en el pipeline
X = df.drop(['outcome_tasa_desercion_snies', 'outcome_merge_pendiente'], axis=1)
y = df['outcome_tasa_desercion_snies']
```

## 📊 Variables Principales

### Target (Variable a Predecir)
```python
df['outcome_tasa_desercion_snies']  # Tasa de deserción (0-100%)
```

### Features Económicas
```python
df['pib_total_miles_millones_cop']           # PIB departamental
df['ipc_nacional_total_var_mensual_media']   # Inflación
df['geih_td_nacional_media_anual']           # Desempleo
df['geih_to_nacional_media_anual']           # Ocupación
df['geih_tgp_nacional_media_anual']          # Participación
```

### Features Educativas
```python
df['total_admitidos']      # Nuevos estudiantes admitidos
df['total_matriculados']   # Estudiantes activos
df['ratio_matriculados_sobre_admitidos']  # Tasa de aceptación
```

### Features de Deserción por Nivel
```python
df['spadies_td_anual_universitario']         # Deserción universitaria
df['spadies_td_anual_tyt']                   # Deserción técnico/tecnológico
df['spadies_td_anual_tecnico_profesional']   # Deserción técnico profesional
df['spadies_td_anual_tecnologico']           # Deserción tecnológica
```

## 📈 Ejemplos de Análisis

### Evolución del PIB
```python
import matplotlib.pyplot as plt

# PIB por departamento en 2024
pib_2024 = df[df['anio'] == 2024].nlargest(10, 'pib_total_miles_millones_cop')
pib_2024.plot(x='departamento', y='pib_total_miles_millones_cop', kind='barh')
plt.xlabel('PIB (miles de millones COP)')
plt.title('Top 10 Departamentos por PIB - 2024')
plt.show()
```

### Tendencia de Deserción
```python
# Promedio nacional por año
desercion_por_año = df.groupby('anio')['outcome_tasa_desercion_snies'].mean()

plt.figure(figsize=(12, 6))
plt.plot(desercion_por_año.index, desercion_por_año.values, linewidth=2)
plt.xlabel('Año')
plt.ylabel('Tasa de Deserción (%)')
plt.title('Evolución de la Tasa de Deserción (1980-2024)')
plt.grid(True, alpha=0.3)
plt.show()
```

### Correlaciones
```python
# Variables clave correlacionadas con deserción
cols = ['outcome_tasa_desercion_snies', 'pib_total_miles_millones_cop', 
        'total_matriculados', 'geih_td_nacional_media_anual', 
        'ipc_nacional_total_var_mensual_media']

correlaciones = df[cols].corr()['outcome_tasa_desercion_snies'].sort_values()
print(correlaciones)
```

## 🔍 Información de Calidad

### Datos Faltantes
```python
# Columnas con nulos (solo variaciones interanuales en primer año)
print(df.isnull().sum()[df.isnull().sum() > 0])
```

**Esperado:**
- `pib_variacion_pct_anual_vs_anio_previo`: 33 nulos (año 1980)
- `var_pct_matriculados_vs_anio_previo`: 33 nulos (año 1980)

### Distribuciones

| Variable | Min | Media | Max | Distribución |
|----------|-----|-------|-----|--------------|
| PIB (M COP) | 24K | 185K | 1.5M | Lognormal |
| Matriculados | 8K | 34K | 344K | Normal |
| Deserción (%) | 3 | 10.3 | 17.4 | Beta |
| IPC (%mensual) | 0.21 | 1.28 | 3.5 | Normal |
| Desempleo (%) | 9.1 | 13.4 | 18.8 | Normal |

## 💡 Tips

### Filtrar por Período
```python
# Datos recientes
df_2010_2024 = df[df['anio'] >= 2010]

# Datos históricos
df_1980_2000 = df[df['anio'] <= 2000]

# Específico
df_2023 = df[df['anio'] == 2023]
```

### Filtrar por Departamento
```python
# Bogotá
df_bogota = df[df['codigo_departamento'] == 11]

# Región
ciudades_mayores = [5, 8, 11, 68, 76]
df_ciudades = df[df['codigo_departamento'].isin(ciudades_mayores)]
```

### Normalizar Variables
```python
from sklearn.preprocessing import StandardScaler

# Normalizar features numéricos
scaler = StandardScaler()
cols_numericas = df.select_dtypes(include=['float64', 'int64']).columns
df_scaled = df.copy()
df_scaled[cols_numericas] = scaler.fit_transform(df[cols_numericas])
```

## 🎯 Casos de Uso

### 1. Predecir Deserción por Departamento
```python
# Datos 2023-2024
df_train = df[df['anio'] < 2023]
df_test = df[df['anio'] >= 2023]

# Entrenar modelo
from sklearn.ensemble import RandomForestRegressor

X_train = df_train.drop(['outcome_tasa_desercion_snies', ...], axis=1)
y_train = df_train['outcome_tasa_desercion_snies']

model = RandomForestRegressor()
model.fit(X_train, y_train)
```

### 2. Análisis Temporal
```python
# Impacto de crisis en deserción
crisis_1999 = df[(df['anio'] >= 1997) & (df['anio'] <= 2001)]
crisis_2008 = df[(df['anio'] >= 2006) & (df['anio'] <= 2010)]

print(f"Promedio deserción 1997-1999: {crisis_1999['outcome_tasa_desercion_snies'].mean():.2f}%")
print(f"Promedio deserción 2006-2008: {crisis_2008['outcome_tasa_desercion_snies'].mean():.2f}%")
```

### 3. Comparación Departamental
```python
# Deserción por departamento (2024)
df_2024 = df[df['anio'] == 2024]
desercion_dept = df_2024.groupby('departamento')['outcome_tasa_desercion_snies'].mean().sort_values()

print("Departamentos con MENOR deserción:")
print(desercion_dept.head(10))

print("\nDepartamentos con MAYOR deserción:")
print(desercion_dept.tail(10))
```

## 📚 Documentación Completa

Para más detalles ver:
- `data/README.md` - Documentación completa
- `data/resumen_simulacion.txt` - Estadísticas detalladas
- `data/data_generator.py` - Código fuente comentado

## ❓ Preguntas Frecuentes

**P: ¿Por qué hay valores faltantes en variaciones?**
A: Son NaN en el año 1980 (primer año) porque no hay año anterior para comparar.

**P: ¿Qué pasa si ejecuto el generador múltiples veces?**
A: Genera nuevos números aleatorios cada vez. Usa `seed=42` para reproducibilidad.

**P: ¿Puedo usar datos posteriores a 2024?**
A: Sí, pero son proyecciones basadas en tendencias históricas, no reales observadas.

**P: ¿Cómo calibro variables específicas?**
A: Edita `VALORES_BASE` en `data_generator.py` con datos reales observados.

---

**Última actualización**: 2024
**Versión**: 1.0 ✅
