"""
data_loader.py
==============
Funciones de carga de datos para el proyecto de deserción estudiantil.
Responsable: Santi | Rama: feature/data-pipeline

Fuentes soportadas:
- DANE (desempleo, pobreza, PIB, IPC)
- BanRep (tasas de interés)
- MEN / SPADIES (deserción, matrícula)
"""

import pandas as pd
import os
from pathlib import Path

# ── Rutas base ────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
INTERIM_DIR = Path("data/interim")


# ── Funciones de carga individuales ──────────────────────────────────────────

def load_dane_desempleo(filepath: str) -> pd.DataFrame:
    """
    Carga datos de desempleo departamental del DANE.
    Espera columnas: Departamento, Año, Tasa_Desempleo
    """
    df = pd.read_excel(filepath)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"tasa_desempleo": "desempleo"})
    df["departamento"] = df["departamento"].str.title().str.strip()
    return df[["año", "departamento", "desempleo"]]


def load_dane_pobreza(filepath: str) -> pd.DataFrame:
    """
    Carga datos de pobreza monetaria del DANE.
    Espera columnas: Departamento, Año, Incidencia_Pobreza
    """
    df = pd.read_excel(filepath)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"incidencia_pobreza": "pobreza"})
    df["departamento"] = df["departamento"].str.title().str.strip()
    return df[["año", "departamento", "pobreza"]]


def load_dane_ipc(filepath: str) -> pd.DataFrame:
    """
    Carga datos de inflación (IPC) del DANE o BanRep.
    Espera columnas: Año, IPC_General, IPC_Educacion
    """
    df = pd.read_excel(filepath)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={
        "ipc_general": "inflacion",
        "ipc_educacion": "ipc_educacion"
    })
    return df[["año", "inflacion", "ipc_educacion"]]


def load_banrep_tasas(filepath: str) -> pd.DataFrame:
    """
    Carga tasas de interés del Banco de la República.
    Espera columnas: Año, Tasa_Referencia
    """
    df = pd.read_excel(filepath)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"tasa_referencia": "tasa_interes"})
    return df[["año", "tasa_interes"]]


def load_men_desercion(filepath: str) -> pd.DataFrame:
    """
    Carga datos de deserción estudiantil del MEN / SPADIES.
    Espera columnas: Departamento, Año, Tasa_Desercion, Matricula_Total
    """
    df = pd.read_excel(filepath)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={
        "tasa_desercion": "desercion",
        "matricula_total": "matricula_total"
    })
    df["departamento"] = df["departamento"].str.title().str.strip()
    return df[["año", "departamento", "desercion", "matricula_total"]]


def load_dane_pib(filepath: str) -> pd.DataFrame:
    """
    Carga datos de PIB departamental del DANE.
    Espera columnas: Departamento, Año, PIB, Crecimiento_PIB
    """
    df = pd.read_excel(filepath)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={
        "pib": "pib_departamental",
        "crecimiento_pib": "crecimiento_pib"
    })
    df["departamento"] = df["departamento"].str.title().str.strip()
    return df[["año", "departamento", "pib_departamental", "crecimiento_pib"]]


# ── Función de merge principal ────────────────────────────────────────────────

def build_master_dataset(raw_paths: dict) -> pd.DataFrame:
    """
    Carga y une todas las fuentes en un único dataset maestro.

    Parámetros
    ----------
    raw_paths : dict con claves:
        - 'desempleo'
        - 'pobreza'
        - 'ipc'
        - 'tasas'
        - 'desercion'
        - 'pib'

    Retorna
    -------
    pd.DataFrame con clave departamento + año
    """
    print("📥 Cargando fuentes de datos...")

    df_desercion = load_men_desercion(raw_paths["desercion"])
    df_desempleo = load_dane_desempleo(raw_paths["desempleo"])
    df_pobreza = load_dane_pobreza(raw_paths["pobreza"])
    df_pib = load_dane_pib(raw_paths["pib"])
    df_ipc = load_dane_ipc(raw_paths["ipc"])
    df_tasas = load_banrep_tasas(raw_paths["tasas"])

    # Merge por departamento + año
    df = df_desercion.copy()
    df = df.merge(df_desempleo, on=["año", "departamento"], how="left")
    df = df.merge(df_pobreza, on=["año", "departamento"], how="left")
    df = df.merge(df_pib, on=["año", "departamento"], how="left")

    # IPC y tasas son nacionales (solo por año)
    df = df.merge(df_ipc, on="año", how="left")
    df = df.merge(df_tasas, on="año", how="left")

    print(f"✅ Dataset maestro construido: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


# ── Guardado ──────────────────────────────────────────────────────────────────

def save_interim(df: pd.DataFrame, filename: str = "dataset_interim.csv"):
    """Guarda el dataset en la carpeta interim."""
    os.makedirs(INTERIM_DIR, exist_ok=True)
    path = INTERIM_DIR / filename
    df.to_csv(path, index=False)
    print(f"💾 Guardado en {path}")


def save_processed(df: pd.DataFrame, filename: str = "dataset_final.csv"):
    """Guarda el dataset final limpio en la carpeta processed."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    print(f"✅ Dataset final guardado en {path}")


# ── Uso de ejemplo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TODO: reemplazar con rutas reales cuando los datos estén disponibles
    rutas = {
        "desercion": RAW_DIR / "men_desercion.xlsx",
        "desempleo": RAW_DIR / "dane_desempleo.xlsx",
        "pobreza":   RAW_DIR / "dane_pobreza.xlsx",
        "pib":       RAW_DIR / "dane_pib.xlsx",
        "ipc":       RAW_DIR / "dane_ipc.xlsx",
        "tasas":     RAW_DIR / "banrep_tasas.xlsx",
    }
    df = build_master_dataset(rutas)
    save_interim(df)
    print(df.head())
