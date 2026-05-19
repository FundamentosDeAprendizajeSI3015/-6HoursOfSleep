"""
README.md - Data Generation Module
===================================

Módulo de generación y validación de dataset simulado.
"""

# 📊 Data Generation Module

Generación de dataset simulado de **variables socioeconómicas y educativas de Colombia (1980-2024)**.

## 📂 Estructura

```
data/
├── data_generator.py      # Generador principal
├── data_validator.py      # Validador de calidad
├── README.md              # Este archivo
├── data_simulado_1980_2024.csv  # Dataset generado (después de ejecutar)
└── resumen_simulacion.txt       # Reporte de simulación
```

## 🎯 Objetivos

✓ Simular **45 años de historia económica** de Colombia (1980-2024)
✓ Crear dataset realista con **33 departamentos**
✓ Incorporar **eventos históricos** relevantes (crisis, pandemia)
✓ Usar **distribuciones probabilísticas diferentes** para cada variable
✓ Calibrar valores con **datos reales observados** (2023-2024)

## 📋 Variables Simuladas

### Macroeconómicas (National)

| Variable | Tipo | Distribución | Rango |
|----------|------|--------------|-------|
| `pib_total_miles_millones_cop` | Continua | Lognormal + tendencia logística | 100 - 1M |
| `pib_variacion_pct_anual_vs_anio_previo` | Continua | Normal derivada | -15% a +15% |
| `ipc_nacional_total_var_mensual_media` | Continua | Normal por régimen | 0.1% a 3.5% |
| `ipc_nacional_total_var_mensual_mediana` | Continua | Normal | 0.1% a 3.5% |
| `ipc_nacional_total_var_mensual_std` | Continua | Normal | 0.05% a 2% |

### Por Categoría de IPC

- `ipc_nacional_educacion_*` - Variación del IPC educación
- `ipc_nacional_alimentos_*` - Variación del IPC alimentos  
- `ipc_nacional_transporte_*` - Variación del IPC transporte
- `ipc_capital_*` - IPC para la capital (Bogotá)

### Empleo (National)

| Variable | Tipo | Distribución | Rango |
|----------|------|--------------|-------|
| `geih_td_nacional_media_anual` | Continua | Normal + ciclos 8 años | 4% - 22% |
| `geih_to_nacional_media_anual` | Continua | Normal inversa | 45% - 70% |
| `geih_tgp_nacional_media_anual` | Continua | Normal + tendencia | 52% - 68% |

### Educación (por departamento)

| Variable | Tipo | Distribución | Interpretación |
|----------|------|--------------|-----------------|
| `total_admitidos` | Discreta | Poisson | Nuevos admitidos al año |
| `total_matriculados` | Continua | Normal | Estudiantes activos |
| `ratio_matriculados_sobre_admitidos` | Continua | Derivada | Tasa de aceptación |
| `var_pct_matriculados_vs_anio_previo` | Continua | Normal | Crecimiento interanual |

### Deserción Escolar (SPADIES)

| Variable | Tipo | Distribución | Nivel |
|----------|------|--------------|-------|
| `spadies_td_anual_universitario` | Continua | Beta | Educación universitaria |
| `spadies_td_anual_tyt` | Continua | Beta | Técnico y tecnológico |
| `spadies_td_anual_tecnico_profesional` | Continua | Beta | Técnico profesional |
| `spadies_td_anual_tecnologico` | Continua | Beta | Educación tecnológica |
| **`outcome_tasa_desercion_snies`** | Continua | Beta | ⭐ TARGET VARIABLE |

## 🔄 Procesos de Simulación

### 1. Tendencia PIB (Lognormal)

```
PIB(t) = Base × (Logística(t) × 2 + Tendencia(t) + Ciclo(t) + Crisis(t)) × Ruido_Lognormal
```

**Componentes:**
- **Logística**: Crecimiento S (1980-2024)
- **Tendencia**: Crecimiento lineal (inflexión en 2005)
- **Ciclo**: Oscilación económica cada 8 años
- **Crisis**: Shocks en 1999, 2008, 2020
- **Ruido**: Variabilidad 10%

### 2. IPC (Normal por Régimen)

```
1980-1990: Media=2.5%, Std=0.8%  (Inflación Alta)
1991-2000: Media=1.2%, Std=0.4%  (Inflación Moderada)
2001-2024: Media=0.6%, Std=0.2%  (Inflación Baja)
```

**Eventos especiales:**
- 1999: +2.1% (crisis bancaria)
- 2008: +1.5% (crisis global)
- 2020: +1.2% (COVID)
- 2022: +1.8% (inflación post-COVID)

### 3. Empleo (Normal + Ciclos)

**Desempleo (TD):**
```
TD(t) = 13% + Tendencia(-0.1% /año) + Ciclo_Econ(8 años) + Ruido
```

**Ocupación (TO):**
```
TO(t) = Inversa a TD + Componente constante
```

**Participación (TGP):**
```
TGP(t) = 55% + Tendencia(+0.15% /año) + Estacional + Ruido
```
→ Refleja feminización del mercado laboral

### 4. Matriculados (Poisson → Normal)

```
Matriculados(t,dept) = Base_dept × (1 + t/45 × 2.5) + Ruido_Normal
```

**Factores departamentales:**
- Bogotá: 5.0x (mayor ciudad)
- Antioquia: 1.2x (segundo centro urbano)
- Atlántico: 0.5x
- Otros: 0.6-1.0x

### 5. Deserción (Beta con variación)

```
Deserción(t,dept) = 12% - (t/45 × 4%) + Ajuste_Departamental + Shocks
```

**Ajustes departamentales:**
- Bogotá, Antioquia: -2% (mejor desempeño)
- La Guajira, Chocó: +2-3% (mayor vulnerabilidad)

**Shocks:**
- 1999: +2pp (crisis)
- 2008: +1.5pp (crisis)
- 2020: +3pp (COVID)

## 🚀 Uso

### Paso 1: Generar Dataset

```bash
cd supervised/data
python data_generator.py
```

**Salida:**
```
data_simulado_1980_2024.csv  (1,485 registros × 32 variables)
resumen_simulacion.txt        (documentación completa)
```

### Paso 2: Validar Calidad

```bash
python data_validator.py data_simulado_1980_2024.csv
```

**Valida:**
✓ Estructura y tipos de datos
✓ Integridad (nulos, duplicados)
✓ Distribuciones realistas
✓ Tendencias históricas
✓ Correlaciones económicas

### Paso 3: Cargar en Pipeline

```python
# En supervised/main.py
df = pd.read_csv('data/data_simulado_1980_2024.csv')

# Filtrar años específicos si es necesario
df_reciente = df[df['anio'] >= 2015]
```

## 📊 Estadísticas Esperadas

| Variable | Media | Std | Min | Max |
|----------|-------|-----|-----|-----|
| PIB (miles M COP) | 30K | 50K | 100 | 400K |
| Matriculados | 20K | 80K | 500 | 2.6M |
| Deserción (%) | 10 | 5 | 3 | 25 |
| IPC mensual (%) | 1.0 | 0.8 | -5 | 10 |
| Desempleo (%) | 11 | 3 | 4 | 22 |

## 🎯 Calibración con Datos Reales

Las simulaciones están calibradas para **coincidir** con datos reales observados:

```
2023:
- PIB Total: ~1,050,000 miles M COP ✓
- Desempleo: 10.17% ✓
- Ocupación: 57.62% ✓
- Participación: 64.14% ✓

2024:
- PIB Total: ~1,150,000 miles M COP ✓
- Desempleo: 10.16% ✓
- Ocupación: 57.41% ✓
- Participación: 63.91% ✓
```

## 📝 Eventos Históricos Incorporados

| Año | Evento | Impacto | Variables |
|-----|--------|--------|-----------|
| 1999 | Crisis bancaria | Desempleo ↑, Matriculados ↓ | TD, Matriculados |
| 2008 | Crisis financiera global | Contracción económica | PIB, TD |
| 2020 | Pandemia COVID-19 | Volatilidad educativa | Deserción |
| 2022 | Inflación post-pandemia | IPC ↑↑ | IPC |

## 🔍 Reproducibilidad

Para obtener **exactamente los mismos números**:

```python
np.random.seed(42)
df = generar_dataset_completo()
```

Para **nuevos números** (aleatoriedad):

```python
# No fijar semilla - cada ejecución es diferente
df = generar_dataset_completo()
```

## ⚠️ Limitaciones Conocidas

1. **Simplificación**: Algunas dinámicas complejas se simulan con ciclos simples
2. **Independencia**: Variables relacionadas pueden no capturar todas las dependencias
3. **Proyección**: Años futuros (> 2024) siguen tendencias observadas
4. **Espacial**: No incluye variación dentro de departamentos (solo promedios)

## 🚀 Próximos Pasos

- [ ] Generar datos diarios/mensuales (más granularidad)
- [ ] Incorporar modelos VAR (vectores autoregresivos)
- [ ] Agregar variables ambientales/climáticas
- [ ] Modelar migraciones internas
- [ ] Incluir sectores económicos específicos

## 📧 Contacto

Módulo desarrollado para proyecto de análisis de deserción estudiantil.
**Responsable**: Equipo de Ciencia de Datos

---

**Última actualización**: 2024
**Versión**: 1.0
