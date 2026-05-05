# Deserción Estudiantil e Indicadores Económicos — Colombia

> Pipeline de Aprendizaje Automático para relacionar indicadores de deserción estudiantil universitaria con indicadores macro y microeconómicos en Colombia.

---

## Objetivo del proyecto

Construir un pipeline completo de ML que permita:

1. **Relacionar** indicadores de deserción estudiantil (MEN/SPADIES) con indicadores económicos (DANE, BanRep).
2. **Construir un índice compuesto** como combinación lineal de variables económicas y educativas.
3. **Predecir** tasas de deserción mediante modelos supervisados.

---

## 🗂️ Estructura del repositorio

```
desercion-economica-col/
│
├── README.md                        ← Este archivo
│
├── data/
│   ├── raw/                         ← Datos originales sin modificar
│   │   ├── .gitkeep
│   │   └── fuentes.md               ← Documentación de fuentes
│   ├── processed/                   ← Datos limpios y transformados
│   │   └── .gitkeep
│   └── interim/                     ← Datos intermedios
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_carga_exploracion/        ← SANTI
│   ├── 02_unsupervised_indice/      ← ISABELA
│   └── 03_modelos_supervisados/     ← JUANES
│
├── src/
│   ├── data_loader.py               ← SANTI
│   ├── preprocessing.py             ← SANTI
│   ├── index_builder.py             ← ISABELA
│   ├── models.py                    ← JUANES
│   └── evaluation.py                ← JUANES
│
├── reports/
│   └── figures/                     ← Gráficas exportadas
│
├── requirements.txt
└── .gitignore
```

---

## Equipo y responsabilidades

| Integrante        | Rama                           | Responsabilidad                    |
| ----------------- | ------------------------------ | ---------------------------------- |
| **Santi**   | `feature/data-pipeline`      | Carga, limpieza y EDA              |
| **Isabela** | `feature/unsupervised-index` | Análisis no supervisado + índice |
| **Juanes**  | `feature/supervised-models`  | Modelos supervisados + evaluación |

---

## Pipeline general

```
[Fuentes de datos]
      ↓
[01. Carga & Preprocesamiento]  ← Santi
      ↓
[02. EDA + Índice compuesto]    ← Isabela
      ↓
[03. Modelos supervisados]      ← Juanes
      ↓
[Reporte de resultados]
```

---

## Variables del dataset

### Variable objetivo

* `desercion` — Tasa de deserción anual por departamento (fuente: MEN/SPADIES)

### Variables macroeconómicas

* `pib_departamental` — PIB por departamento (DANE)
* `crecimiento_pib` — Variación % anual del PIB (DANE)
* `inflacion` — IPC general (DANE / BanRep)
* `ipc_educacion` — IPC específico del sector educativo (DANE)
* `tasa_interes` — Tasa de interés de referencia (BanRep)

### Variables socioeconómicas

* `desempleo` — Tasa de desempleo departamental (DANE)
* `desempleo_juvenil` — Tasa de desempleo en jóvenes 18-28 años (DANE, si disponible)
* `informalidad` — Tasa de informalidad laboral (DANE)
* `pobreza` — Incidencia de pobreza monetaria (DANE)
* `ingreso_promedio` — Ingreso promedio del hogar (DANE ENPH)

### Variables educativas

* `matricula_total` — Matrícula universitaria total por departamento (MEN)
* `cobertura_educacion` — Tasa de cobertura en educación superior (MEN)

### Variables derivadas (a construir)

* `ingreso_real` = ingreso_promedio / inflacion
* `ratio_desempleo_pobreza` = desempleo × pobreza
* `cambio_desercion` = desercion_t - desercion_t-1

---

## Fuentes de datos

| Fuente        | URL                             | Datos                         |
| ------------- | ------------------------------- | ----------------------------- |
| DANE          | https://www.dane.gov.co         | PIB, empleo, pobreza, IPC     |
| BanRep        | https://www.banrep.gov.co       | Tasas de interés, inflación |
| MEN / SPADIES | https://www.mineducacion.gov.co | Deserción, matrícula        |
| Banco Mundial | https://data.worldbank.org      | Complementario                |

---

## ⚙️ Instalación

```bash
git clone https://github.com/tu-usuario/desercion-economica-col.git
cd desercion-economica-col
pip install -r requirements.txt
```

---

## Convenciones del proyecto

* Datos crudos **nunca se modifican** directamente — siempre en `data/raw/`
* Cada notebook comienza con una celda de configuración estándar
* Todas las figuras se exportan a `reports/figures/`
* Los supuestos de limpieza se documentan en el notebook correspondiente

---

## 📅 Estado del proyecto

* [X] Estructura del repositorio
* [ ] Recolección de datos
* [ ] Preprocesamiento (Santi)
* [ ] EDA + índice (Isabela)
* [ ] Modelos supervisados (Juanes)
* [ ] Reporte final
