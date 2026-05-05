"""
Construye el panel departamento × año para relacionar deserción estudiantil con
variables socioeconómicas, integrando todo lo usable en DATOS/.

Salidas:
  - DATOS/panel_desercion_socioeconomico_completo.csv
  - DATOS/panel_desercion_socioeconomico_meta.json

Fuentes:
  - PIB total departamental (DANE, TotalDep)
  - Admitidos / matriculados MEN (agregado por departamento domicilio IES)
  - IPC variación mensual (archivo largo 2014–2024): Nacional + capitales IPC
  - GEIH tasa de desempleo TD nacional (serie mensual, media anual)
  - SPADIES (articles-415244_recurso_7.xlsx, hoja «TD - departamento»): TD anual por
    nivel de formación y departamento de oferta (2010–2023).

La etiqueta principal de deserción se toma de SPADIES — TD nivel universitario;
los demás niveles quedan como covariables desagregadas.

Uso:
    python build_panel_desercion_socioeconomico.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATOS = Path(__file__).resolve().parent / "DATOS"
REPORT_JSON = DATOS / "analysis_report.json"
OUT_CSV = DATOS / "panel_desercion_socioeconomico_completo.csv"
OUT_META = DATOS / "panel_desercion_socioeconomico_meta.json"

SHEET_DATA = "1."
HEADER_ROW = 5
PIB_FILE = DATOS / "PIB" / "anex-PIBDep-TotalDep-2024pr.xlsx"
PIB_SHEET = "Cuadro 1"
PIB_HEADER_ROW = 9
PIB_DATA_START_ROW = 10

IPC_FILE_LONG = DATOS / "IPC" / "ipc_from_2014-09_to_2024-08.csv"
GEIH_FILE = DATOS / "GEIH" / "anex-GEIH-feb2026.xlsx"
GEIH_SHEET_TD = "Total nacional"
GEIH_ROW_TD = 16

SPADIES_FILE = DATOS / "SPADIES" / "articles-415244_recurso_7.xlsx"
SPADIES_SHEET_TD_DEPT = "TD - departamento"
SPADIES_MAX_ANIO_TD = 2023  # el Excel solo publica hasta este año para TD anual


def load_merge_years() -> list[int]:
    if REPORT_JSON.is_file():
        with open(REPORT_JSON, encoding="utf-8") as f:
            rep = json.load(f)
        panel = rep.get("temporal_alignment", {}).get("panel_education_ipc_pib", {})
        ys = panel.get("suggested_merge_years_calendario")
        if ys:
            return [int(y) for y in ys]
    return [2023, 2024]


def parse_pib_year_label(val) -> int | None:
    if pd.isna(val):
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, float) and not np.isnan(val):
        return int(val)
    s = str(val).strip()
    m = re.match(r"^(20\d{2})", s)
    return int(m.group(1)) if m else None


def norm_dept_code(x) -> int | None:
    if pd.isna(x):
        return None
    s = str(x).strip().replace("'", "")
    try:
        return int(float(s))
    except ValueError:
        return None


def load_pib_year_columns(raw: pd.DataFrame, years: list[int]) -> dict[int, int]:
    hdr = raw.iloc[PIB_HEADER_ROW]
    ycol: dict[int, int] = {}
    for j in range(len(hdr)):
        py = parse_pib_year_label(hdr.iloc[j])
        if py is not None and py in years and py not in ycol:
            ycol[py] = j
    return ycol


def load_pib_wide(years: list[int]) -> pd.DataFrame:
    """PIB miles millones COP por depto; incluye años previos para variación %."""
    need_years = sorted(set(years) | set(y - 1 for y in years))
    raw = pd.read_excel(PIB_FILE, sheet_name=PIB_SHEET, header=None)
    ycol = load_pib_year_columns(raw, need_years)
    missing = set(need_years) - set(ycol.keys())
    if missing:
        raise RuntimeError(f"PIB: faltan columnas para años {missing}")

    rows = []
    for i in range(PIB_DATA_START_ROW, len(raw)):
        code = raw.iloc[i, 0]
        name = raw.iloc[i, 1]
        if pd.isna(name):
            continue
        name_s = str(name).strip()
        if name_s.upper() == "COLOMBIA":
            continue
        cd = norm_dept_code(code)
        if cd is None:
            continue
        rec = {"codigo_departamento": cd, "departamento": name_s}
        for y, j in ycol.items():
            rec[f"pib_{y}"] = pd.to_numeric(raw.iloc[i, j], errors="coerce")
        rows.append(rec)

    return pd.DataFrame(rows)


def find_column(df: pd.DataFrame, must_include: tuple[str, ...]) -> str | None:
    for c in df.columns:
        cs = str(c).upper()
        if all(k in cs for k in must_include):
            return c
    return None


def aggregate_men(path: Path, year: int, value_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_DATA, header=HEADER_ROW)
    code_col = find_column(df, ("CÓDIGO", "DEPARTAMENTO", "IES"))
    val_col = find_column(df, (value_name,))
    if not code_col or not val_col:
        raise RuntimeError(f"{path}: code={code_col}, val={val_col}")
    g = (
        df.groupby(code_col, dropna=False)[val_col]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={code_col: "codigo_departamento", val_col: f"{value_name.lower()}_{year}"})
    )
    g["codigo_departamento"] = g["codigo_departamento"].map(norm_dept_code)
    return g.dropna(subset=["codigo_departamento"])


def fold_str(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def resolve_ipc_city_label(preferred: str, ipc_cities: list[str]) -> str | None:
    """Prefiere etiquetas largas (evita 'Cartagena' vacío vs 'Cartagena De Indias')."""
    fp = fold_str(preferred)
    matches: list[str] = []
    for c in ipc_cities:
        fc = fold_str(c)
        if fc == fp or fp in fc or fc in fp:
            matches.append(c)
    if not matches:
        return None
    return max(matches, key=len)


CAPITAL_PREFERIDA_POR_DEPT: dict[int, str] = {
    5: "Medellín",
    8: "Barranquilla",
    11: "Bogotá",
    13: "Cartagena De Indias",
    15: "Tunja",
    17: "Manizales",
    18: "Florencia",
    19: "Popayán",
    20: "Valledupar",
    23: "Montería",
    25: "Bogotá",
    27: "Quibdó",
    41: "Neiva",
    44: "Riohacha",
    47: "Santa Marta",
    50: "Villavicencio",
    52: "Pasto",
    54: "Cúcuta",
    63: "Armenia",
    66: "Pereira",
    68: "Bucaramanga",
    70: "Sincelejo",
    73: "Ibagué",
    76: "Cali",
    81: "Arauca",
    85: "Yopal",
    86: "Mocoa",
    88: "San Andrés",
    91: "Leticia",
    94: "Inírida",
    95: "San José del Guaviare",
    97: "Mitú",
    99: "Puerto Carreño",
}


def load_ipc_full(encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(IPC_FILE_LONG, encoding=encoding, low_memory=False)


def ipc_nacional_por_anio(ipc: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    sub = ipc[
        (ipc["city"].astype(str).str.strip() == "Nacional") & (ipc["year"].isin(years))
    ].copy()

    def agg_cat(cat_pat: str, prefix: str) -> pd.DataFrame:
        sc = sub[sub["category"].astype(str).str.contains(cat_pat, case=False, na=False)]
        return (
            sc.groupby("year", as_index=False)["ipc"]
            .agg(["mean", "median", "std"])
            .rename(
                columns={
                    "mean": f"ipc_nacional_{prefix}_var_mensual_media",
                    "median": f"ipc_nacional_{prefix}_var_mensual_mediana",
                    "std": f"ipc_nacional_{prefix}_var_mensual_std",
                }
            )
            .assign(year=lambda d: d["year"].astype(int))
            .rename(columns={"year": "anio"})
        )

    total = agg_cat("Total", "total")
    edu = agg_cat("Educación", "educacion")
    ali = agg_cat("Alimentos", "alimentos")
    tra = agg_cat("Transporte", "transporte")

    out = total.merge(edu, on="anio", how="outer").merge(ali, on="anio", how="outer").merge(
        tra, on="anio", how="outer"
    )
    return out


def ipc_capital_por_anio(
    ipc: pd.DataFrame, dept_codes: list[int], years: list[int], ipc_city_labels: dict[int, str | None]
) -> pd.DataFrame:
    records = []
    for anio in years:
        for cd in dept_codes:
            city = ipc_city_labels.get(cd)
            row: dict = {"anio": anio, "codigo_departamento": cd}
            if city is None:
                row["ipc_capital_total_var_mensual_media"] = np.nan
                row["ipc_capital_educacion_var_mensual_media"] = np.nan
                records.append(row)
                continue
            chunk = ipc[(ipc["city"] == city) & (ipc["year"] == anio)]
            for cat_pat, suffix in [("Total", "total"), ("Educación", "educacion")]:
                m = chunk[chunk["category"].astype(str).str.contains(cat_pat, case=False, na=False)][
                    "ipc"
                ]
                row[f"ipc_capital_{suffix}_var_mensual_media"] = float(m.mean()) if len(m) else np.nan
            records.append(row)
    return pd.DataFrame(records)


def parse_geih_monthly_series(sheet_name: str, metric_row: int) -> dict[int, list[float]]:
    raw = pd.read_excel(GEIH_FILE, sheet_name=sheet_name, header=None)
    r11 = raw.iloc[11]
    r12 = raw.iloc[12]
    vals = raw.iloc[metric_row]
    current_year: int | None = None
    by_year: dict[int, list[float]] = {}
    for j in range(1, raw.shape[1]):
        ycell = r11.iloc[j]
        mcell = r12.iloc[j]
        if pd.notna(ycell):
            try:
                if isinstance(ycell, (int, float)) and not isinstance(ycell, bool):
                    current_year = int(ycell)
                elif isinstance(ycell, str) and ycell.strip().isdigit():
                    current_year = int(ycell.strip())
            except (ValueError, TypeError):
                pass
        mes = None
        if isinstance(mcell, str) and len(mcell) >= 3:
            meses = {
                "ene": 1,
                "feb": 2,
                "mar": 3,
                "abr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "ago": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dic": 12,
            }
            mes = meses.get(mcell.strip().lower()[:3])
        if current_year and mes:
            v = vals.iloc[j]
            if pd.notna(v):
                by_year.setdefault(current_year, []).append(float(v))
    return by_year


def geih_nacional_medias_anuales(years: list[int]) -> pd.DataFrame:
    td = parse_geih_monthly_series(GEIH_SHEET_TD, GEIH_ROW_TD)
    to = parse_geih_monthly_series(GEIH_SHEET_TD, 15)
    tgp = parse_geih_monthly_series(GEIH_SHEET_TD, 14)
    rows = []
    for y in years:
        rows.append(
            {
                "anio": y,
                "geih_td_nacional_media_anual": float(np.nanmean(td.get(y, [np.nan]))),
                "geih_to_nacional_media_anual": float(np.nanmean(to.get(y, [np.nan]))),
                "geih_tgp_nacional_media_anual": float(np.nanmean(tgp.get(y, [np.nan]))),
            }
        )
    return pd.DataFrame(rows)


def try_load_enph_snippet() -> dict:
    path = DATOS / "ENPH" / "total-nacional-enph-2017.xls"
    out: dict = {"archivo": path.name, "ok": False}
    if not path.is_file():
        out["error"] = "archivo no encontrado"
        return out
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        out["error"] = str(e)
        return out
    out["hojas"] = xl.sheet_names[:5]
    out["ok"] = True
    return out


def year_col_in_header(header_row: pd.Series, year: int) -> int | None:
    for j in range(1, len(header_row)):
        v = header_row.iloc[j]
        if pd.isna(v):
            continue
        try:
            if int(float(v)) == year:
                return j
        except (TypeError, ValueError):
            continue
    return None


def build_fold_to_codigo(ref: pd.DataFrame) -> dict[str, int]:
    m: dict[str, int] = {}
    for _, r in ref[["codigo_departamento", "departamento"]].drop_duplicates().iterrows():
        m[fold_str(str(r["departamento"]))] = int(r["codigo_departamento"])
    return m


def spadies_name_to_codigo(name: str, fold_to_codigo: dict[str, int]) -> int | None:
    if pd.isna(name):
        return None
    s = str(name).strip()
    if s.upper().startswith("NA:"):
        return None
    k = fold_str(s)
    if k in fold_to_codigo:
        return fold_to_codigo[k]
    alias_panel_fold = {
        fold_str("Archipiélago De San Andrés, Providencia Y Santa Catalina"): fold_str(
            "San Andrés, Providencia y Santa Catalina (Archipiélago)"
        ),
        fold_str("Archipiélago de San Andrés, Providencia y Santa Catalina"): fold_str(
            "San Andrés, Providencia y Santa Catalina (Archipiélago)"
        ),
    }
    if k in alias_panel_fold:
        return fold_to_codigo.get(alias_panel_fold[k])
    if "valle" in k and "cauca" in k:
        return fold_to_codigo.get(fold_str("Valle del Cauca"))
    if "norte" in k and "santander" in k:
        return fold_to_codigo.get(fold_str("Norte de Santander"))
    return None


def load_spadies_td_departamento(
    path: Path,
    fold_to_codigo: dict[str, int],
    year_td: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Lee TD anual por departamento de oferta (varios niveles de formación).
    Devuelve DataFrame con codigo_departamento + columnas spadies_td_* para year_td,
    y estadísticas de cruce (sin_match, etc.).
    """
    if not path.is_file():
        return pd.DataFrame(), {"error": "archivo no encontrado", "archivo": str(path.name)}

    raw = pd.read_excel(path, sheet_name=SPADIES_SHEET_TD_DEPT, header=None)
    if year_col_in_header(raw.iloc[12], year_td) is None:
        return pd.DataFrame(), {"error": f"sin columna numérica para el año {year_td} en SPADIES"}

    sections: list[tuple[str, int, range]] = [
        ("spadies_td_anual_universitario", 12, range(13, 46)),
        ("spadies_td_anual_tyt", 49, range(50, 83)),
        ("spadies_td_anual_tecnico_profesional", 86, range(87, 118)),
        ("spadies_td_anual_tecnologico", 122, range(123, 156)),
    ]

    sin_match = 0
    wide: dict[int, dict[str, float]] = {}

    for col_name, hdr_row, row_range in sections:
        j_year = year_col_in_header(raw.iloc[hdr_row], year_td)
        if j_year is None:
            continue
        for i in row_range:
            if i >= len(raw):
                break
            nm = raw.iloc[i, 0]
            cod = spadies_name_to_codigo(nm, fold_to_codigo)
            if cod is None:
                if pd.notna(nm) and not str(nm).strip().upper().startswith("NA:"):
                    sin_match += 1
                continue
            val = raw.iloc[i, j_year]
            if pd.isna(val):
                continue
            wide.setdefault(cod, {})[col_name] = float(val)

    rows = []
    for cod, vals in wide.items():
        rec = {"codigo_departamento": cod}
        rec.update(vals)
        rows.append(rec)

    df = pd.DataFrame(rows)
    for col_name, _, _ in sections:
        if col_name not in df.columns:
            df[col_name] = np.nan

    stats = {
        "year_td_usado": year_td,
        "departamentos_con_filas": len(df),
        "nombres_spadies_sin_match_divipola": sin_match,
        "archivo": path.name,
    }
    return df, stats


def main() -> None:
    years = load_merge_years()
    years_prev = sorted(set(y - 1 for y in years))

    pib_df = load_pib_wide(sorted(set(years) | set(years_prev)))

    ipc = load_ipc_full()
    ipc_cities = sorted(ipc["city"].astype(str).unique())
    ipc_city_labels: dict[int, str | None] = {}
    for cd, pref in CAPITAL_PREFERIDA_POR_DEPT.items():
        ipc_city_labels[cd] = resolve_ipc_city_label(pref, ipc_cities)

    ipc_nat = ipc_nacional_por_anio(ipc, years)
    dept_codes = pib_df["codigo_departamento"].astype(int).tolist()
    ipc_cap = ipc_capital_por_anio(ipc, dept_codes, years, ipc_city_labels)
    geih_nat = geih_nacional_medias_anuales(years)

    admit_parts = []
    mat_parts = []
    for y in years:
        pa = DATOS / "Educacion" / f"admitidos-{y}.xlsx"
        pm = DATOS / "Educacion" / f"matriculados-{y}.xlsx"
        if pa.is_file():
            admit_parts.append(aggregate_men(pa, y, "ADMITIDOS"))
        if pm.is_file():
            mat_parts.append(aggregate_men(pm, y, "MATRICULADOS"))

    def merge_wide(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on="codigo_departamento", how="outer")
        return out

    admit_w = merge_wide(admit_parts)
    mat_w = merge_wide(mat_parts)

    records = []
    for _, prow in pib_df.iterrows():
        cd = int(prow["codigo_departamento"])
        dept = prow["departamento"]
        for y in years:
            pib_y = prow.get(f"pib_{y}")
            pib_prev = prow.get(f"pib_{y - 1}")
            var_pib = np.nan
            if pd.notna(pib_y) and pd.notna(pib_prev) and pib_prev != 0:
                var_pib = (float(pib_y) - float(pib_prev)) / float(pib_prev) * 100.0

            acol = f"admitidos_{y}"
            mcol = f"matriculados_{y}"
            adm = np.nan
            if not admit_w.empty and acol in admit_w.columns:
                sub = admit_w.loc[admit_w["codigo_departamento"] == cd, acol]
                if len(sub):
                    adm = float(sub.iloc[0])
            mat = np.nan
            if not mat_w.empty and mcol in mat_w.columns:
                subm = mat_w.loc[mat_w["codigo_departamento"] == cd, mcol]
                if len(subm):
                    mat = float(subm.iloc[0])

            ratio_ma = np.nan
            if pd.notna(adm) and pd.notna(mat) and adm != 0:
                ratio_ma = mat / adm

            mat_prev_col = f"matriculados_{y - 1}"
            var_mat_pct = np.nan
            if not mat_w.empty and mat_prev_col in mat_w.columns:
                mp = mat_w.loc[mat_w["codigo_departamento"] == cd, mat_prev_col]
                if len(mp) and pd.notna(mp.iloc[0]) and mp.iloc[0] != 0 and pd.notna(mat):
                    var_mat_pct = (mat - float(mp.iloc[0])) / float(mp.iloc[0]) * 100.0

            row: dict = {
                "codigo_departamento": cd,
                "departamento": dept,
                "anio": y,
                "pib_total_miles_millones_cop": pib_y,
                "pib_variacion_pct_anual_vs_anio_previo": var_pib,
                "total_admitidos": adm,
                "total_matriculados": mat,
                "ratio_matriculados_sobre_admitidos": ratio_ma,
                "var_pct_matriculados_vs_anio_previo": var_mat_pct,
            }

            inat = ipc_nat.loc[ipc_nat["anio"] == y]
            if len(inat):
                for c in inat.columns:
                    if c != "anio":
                        row[c] = inat[c].iloc[0]

            ic = ipc_cap.loc[(ipc_cap["codigo_departamento"] == cd) & (ipc_cap["anio"] == y)]
            if len(ic):
                row["ipc_capital_total_var_mensual_media"] = ic["ipc_capital_total_var_mensual_media"].iloc[
                    0
                ]
                row["ipc_capital_educacion_var_mensual_media"] = ic[
                    "ipc_capital_educacion_var_mensual_media"
                ].iloc[0]
            else:
                row["ipc_capital_total_var_mensual_media"] = np.nan
                row["ipc_capital_educacion_var_mensual_media"] = np.nan

            gn = geih_nat.loc[geih_nat["anio"] == y]
            if len(gn):
                row["geih_td_nacional_media_anual"] = gn["geih_td_nacional_media_anual"].iloc[0]
                row["geih_to_nacional_media_anual"] = gn["geih_to_nacional_media_anual"].iloc[0]
                row["geih_tgp_nacional_media_anual"] = gn["geih_tgp_nacional_media_anual"].iloc[0]

            if pd.notna(mat) and mat and pd.notna(pib_y):
                row["proxy_pib_miles_mm_cop_por_matriculado"] = float(pib_y) / mat
            else:
                row["proxy_pib_miles_mm_cop_por_matriculado"] = np.nan

            records.append(row)

    out = pd.DataFrame(records)
    out = out.sort_values(["anio", "codigo_departamento"]).reset_index(drop=True)

    fold_to_codigo = build_fold_to_codigo(pib_df)
    spadies_blk, spadies_stats = load_spadies_td_departamento(
        SPADIES_FILE,
        fold_to_codigo,
        SPADIES_MAX_ANIO_TD,
    )

    if len(spadies_blk):
        spadies_blk = spadies_blk.copy()
        spadies_blk["anio"] = SPADIES_MAX_ANIO_TD
        out = out.merge(spadies_blk, on=["codigo_departamento", "anio"], how="left")

    if "spadies_td_anual_universitario" in out.columns:
        out["outcome_tasa_desercion_snies"] = out["spadies_td_anual_universitario"]
    else:
        out["outcome_tasa_desercion_snies"] = np.nan
    out["outcome_merge_pendiente"] = (
        np.where(out["outcome_tasa_desercion_snies"].notna(), 0, 1).astype(int)
    )
    out = out.sort_values(["anio", "codigo_departamento"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    enph_meta = try_load_enph_snippet()

    meta = {
        "panel_csv": OUT_CSV.name,
        "filas": len(out),
        "columnas": list(out.columns),
        "anios": years,
        "nota_desercion": (
            "outcome_tasa_desercion_snies coincide con SPADIES TD anual nivel universitario "
            f"(archivo SPADIES, año columna {SPADIES_MAX_ANIO_TD}). "
            "Las columnas spadies_td_anual_* desglosan otros niveles de formación. "
            f"Años > {SPADIES_MAX_ANIO_TD} no tienen TD publicado en este Excel — outcome_merge_pendiente=1."
        ),
        "spadies_merge": spadies_stats,
        "fuentes_integradas": [
            "PIB DANE anex-PIBDep-TotalDep-2024pr.xlsx Cuadro 1 (primer bloque valores absolutos)",
            "Educacion/admitidos-YYYY.xlsx, matriculados-YYYY.xlsx hoja 1.",
            "IPC ipc_from_2014-09_to_2024-08.csv — Nacional + capitales vía mapeo DIVIPOLA",
            "GEIH anex-GEIH-feb2026.xlsx Total nacional filas TD/TO/TGP",
            "SPADIES articles-415244_recurso_7.xlsx hoja «TD - departamento» (TD anual por nivel y departamento de oferta)",
        ],
        "variables_contexto_educativo": [
            "ratio_matriculados_sobre_admitidos y var_pct_matriculados_vs_anio_previo son presión/cambio de matrícula, no deserción.",
            "proxy_pib_miles_mm_cop_por_matriculado = PIB departamental / matriculados.",
        ],
        "no_integrado": [
            "Pobreza Monetaria .xml: solo metadatos DDI.",
            "ENPH .xls: ver enph_snippet.",
            "ipc_from_2019 y ipc_from_2022: redundantes con el archivo largo.",
        ],
        "enph_snippet": enph_meta,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Filas: {len(out)}  Columnas: {len(out.columns)}")
    print(f"CSV: {OUT_CSV}")
    print(f"Meta: {OUT_META}")


if __name__ == "__main__":
    main()
