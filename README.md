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

```text
desercion-economica-col/
│
├── README.md
│
├── data/
│   ├── raw/                         ← Datos originales sin modificar
│   ├── processed/                   ← Datos limpios y transformados
│   └── final/                       ← Datos finales listos para modelado
│
├── load-data/
|
├── eda/
|
├── indexes-scores/
|
├── supervised/
|
├── unsupervised/
|
├── requirements.txt
|
└── .gitignore
```

### Organización del proyecto

Cada etapa del proyecto se encuentra organizada en una carpeta independiente correspondiente a su rama:

- `load-data`
- `eda`
- `indexes-scores`
- `unsupervised`
- `supervised`

Las etapas comparten la misma estructura interna:

- `src/` → Código fuente y scripts principales.
- `docs/` → Documentación y README específicos de la etapa.
- `reports/` → Gráficas, resultados y salidas generadas.

La etapa `unsupervised` explora patrones y clusters en los indicadores socioeconómicos para apoyar el modelado supervisado de la tasa de deserción.

La carpeta global `data/` centraliza los datasets utilizados en todo el proyecto:

- `raw/` → Datos originales.
- `processed/` → Datos transformados y limpios.
- `final/` → Datos finales preparados para análisis y modelado junto a un **README.md** con toda la información de la carga y procesamiento de datos.

---

## Equipo y responsabilidades

| Integrante        | Responsabilidad                    |
| ----------------- | ---------------------------------- |
| **Santi**   | Carga, limpieza y EDA              |
| **Isabella** | Análisis no supervisado + índice |
| **Juanes**  | Modelos supervisados + evaluación |

---

## Pipeline general

```
[Fuentes de datos]
      ↓
[01. Carga & Preprocesamiento]  ← Santi
      ↓
[02. EDA + Índice compuesto]    ← Isabella
      ↓
[03. Análisis no supervisado]   ← Santi
      ↓
[04. Modelos supervisados]      ← Juanes
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
* [X] Recolección de datos
* [X] Preprocesamiento (Santi)
* [X] EDA + índice (Isabela)
* [X] Análisis no supervisado (Isabela)
* [X] Modelos supervisados (Juanes)
* [ ] Reporte final