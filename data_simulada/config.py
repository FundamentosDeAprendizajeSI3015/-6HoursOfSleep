# filepath: config.py
"""
config.py
=========
Configuración centralizada de rutas para el módulo de datos simulados.

Este archivo gestiona todas las rutas de forma centralizada para facilitar
cambios futuros de ubicaciones sin necesidad de actualizar múltiples archivos.

Ubicación base: data_simulada/
"""

from pathlib import Path

# ============================================================================
# RUTAS PRINCIPALES
# ============================================================================

# Carpeta raíz del módulo data_simulada
DATA_SIMULADA_DIR = Path(__file__).parent.absolute()

# Subcarpeta processed (donde se guardan datos generados)
PROCESSED_DIR = DATA_SIMULADA_DIR / "processed"

# Subcarpeta docs (documentación técnica)
DOCS_DIR = DATA_SIMULADA_DIR / "docs"

# Subcarpeta graficas (donde se guardan las visualizaciones)
GRAPHICS_DIR = DATA_SIMULADA_DIR / "graficas"

# ============================================================================
# PERÍODO DE SIMULACIÓN
# ============================================================================

# Rango de años para la simulación
YEAR_START = 1980
YEAR_END = 2026

# ============================================================================
# ARCHIVOS DE DATOS
# ============================================================================

# Dataset simulado completo (1980-2026)
DATASET_PATH = PROCESSED_DIR / "data_simulado_1980_2026.csv"

# Resumen de simulación
SUMMARY_PATH = PROCESSED_DIR / "resumen_simulacion.txt"

# ============================================================================
# ARCHIVOS DE GRÁFICOS
# ============================================================================

GRAPHICS = {
    "pib_nacional": GRAPHICS_DIR / "grafico_pib_nacional.png",
    "desercion_temporal": GRAPHICS_DIR / "grafico_desercion_temporal.png",
    "empleo": GRAPHICS_DIR / "grafico_empleo.png",
    "educacion": GRAPHICS_DIR / "grafico_educacion.png",
    "correlaciones": GRAPHICS_DIR / "grafico_correlaciones.png",
    "comparativa_1980_2026": GRAPHICS_DIR / "grafico_comparativa_1980_2026.png",
    "distribucion_2026": GRAPHICS_DIR / "grafico_distribucion_2026.png",
}

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def crear_directorios():
    """Crea los directorios necesarios si no existen."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directorios creados en: {DATA_SIMULADA_DIR}")


def obtener_ruta_dataset():
    """Retorna la ruta del dataset simulado."""
    return DATASET_PATH


def obtener_ruta_figura(nombre):
    """
    Retorna la ruta de una figura específica.
    
    Args:
        nombre: clave del diccionario GRAPHICS
    
    Returns:
        Path del archivo de figura
    """
    return GRAPHICS.get(nombre, GRAPHICS_DIR / f"{nombre}.png")


def validar_configuracion():
    """Valida que la configuración sea correcta."""
    print("\n" + "="*80)
    print("VALIDACIÓN DE CONFIGURACIÓN")
    print("="*80)
    
    rutas = {
        "DATA_SIMULADA_DIR": DATA_SIMULADA_DIR,
        "PROCESSED_DIR": PROCESSED_DIR,
        "GRAPHICS_DIR": GRAPHICS_DIR,
        "DOCS_DIR": DOCS_DIR,
        "DATASET_PATH": DATASET_PATH,
        "YEAR_RANGE": f"{YEAR_START}-{YEAR_END}",
    }
    
    print("\n📁 Rutas configuradas:")
    for nombre, ruta in rutas.items():
        if nombre == "YEAR_RANGE":
            print(f"   📅 {nombre}: {ruta}")
        else:
            existe = "✓" if (isinstance(ruta, Path) and ruta.exists()) else "✗"
            print(f"   {existe} {nombre}")
            print(f"      {ruta}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    validar_configuracion()
