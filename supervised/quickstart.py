#!/usr/bin/env python
"""
quickstart.py
=============
Script rápido para ejecutar el pipeline en modo batch.

Uso:
    python quickstart.py
    
Genera todo en < 2 minutos:
    ✓ Carga datos
    ✓ Entrena modelos
    ✓ Evalúa
    ✓ Exporta resultados
"""

import sys
from pathlib import Path

# Setup path para imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd

# Importar módulos propios
try:
    from data_loader import load_and_prepare
    from Models import preparar_datos, entrenar_todos_modelos
    from Evaluation import evaluar_todos, guardar_tabla_resultados
    from utils import (
        imprimir_encabezado,
        imprimir_subseccion,
        resumen_datos,
        guardar_informe_texto
    )
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print(f"   Path: {src_path}")
    sys.exit(1)


def quickstart():
    """Ejecuta pipeline rápido."""
    
    imprimir_encabezado("⚡ QUICKSTART — PIPELINE SUPERVISADO")
    
    # 1. Cargar
    print("\n[1/5] Cargando datos...")
    X, y, loader = load_and_prepare()
    resumen_datos(X, y)
    
    # 2. Preparar
    print("\n[2/5] Preparando split train/test...")
    X_train, X_test, y_train, y_test = preparar_datos(X, y)
    
    # 3. Entrenar
    print("\n[3/5] Entrenando modelos con CV...")
    modelos, cv_results = entrenar_todos_modelos(X_train, y_train)
    
    # 4. Evaluar
    print("\n[4/5] Evaluando en test set...")
    df_metricas = evaluar_todos(modelos, X_test.values, y_test.values)
    
    # 5. Exportar
    print("\n[5/5] Guardando resultados...")
    guardar_tabla_resultados(df_metricas, cv_results)
    
    # Informe final
    informe = guardar_informe_texto(modelos, df_metricas)
    print(informe)
    
    imprimir_encabezado("✅ COMPLETADO")
    return {
        "modelos": modelos,
        "metricas": df_metricas,
        "X_test": X_test,
        "y_test": y_test
    }


if __name__ == "__main__":
    resultados = quickstart()
