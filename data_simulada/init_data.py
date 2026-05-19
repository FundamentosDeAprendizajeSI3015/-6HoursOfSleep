"""
init_data.py
============
Script de inicialización rápida del módulo de datos.

Funciones:
- Generar dataset simulado
- Validar calidad
- Mostrar resumen
- Exportar estadísticas

Ejecución:
    python init_data.py
"""

import sys
from pathlib import Path

# Agregar padre al path
sys.path.insert(0, str(Path(__file__).parent))

from data_generator import generar_dataset_completo, generar_reporte_simulacion, YEAR_START, YEAR_END
from data_validator import DatasetValidator
from config import DATASET_PATH, SUMMARY_PATH, crear_directorios


def main():
    """Función principal de inicialización."""
    
    print("\n" + "="*80)
    print("INICIALIZADOR DE MÓDULO DE DATOS")
    print("="*80)
    
    # Crear directorios necesarios
    crear_directorios()
    
    # Paths (desde configuración centralizada)
    dataset_path = DATASET_PATH
    summary_path = SUMMARY_PATH
    
    # ========================================================================
    # PASO 1: GENERAR DATASET
    # ========================================================================
    print("\n[1/3] GENERACIÓN DE DATASET")
    print("-" * 80)
    
    import numpy as np
    np.random.seed(42)  # Reproducibilidad
    
    df = generar_dataset_completo()
    
    # Guardar
    print(f"\n💾 Guardando dataset...")
    df.to_csv(dataset_path, index=False)
    print(f"   ✓ {dataset_path.name}")
    
    # Generar reporte
    print(f"\n📄 Generando reporte...")
    reporte = generar_reporte_simulacion(df)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(reporte)
    print(f"   ✓ {summary_path.name}")
    
    # ========================================================================
    # PASO 2: VALIDAR DATASET
    # ========================================================================
    print("\n\n[2/3] VALIDACIÓN DE CALIDAD")
    print("-" * 80)
    
    validator = DatasetValidator(str(dataset_path))
    validator.generar_reporte_completo()
    
    # ========================================================================
    # PASO 3: RESUMEN FINAL
    # ========================================================================
    print("\n[3/3] RESUMEN FINAL")
    print("-" * 80)
    
    print(f"\n📊 Dataset generado exitosamente:")
    print(f"\n   Archivo: {dataset_path.name}")
    print(f"   Tamaño: {len(df):,} registros × {df.shape[1]} columnas")
    print(f"   Período: {YEAR_START} - {YEAR_END}")
    
    print(f"\n   Top 5 departamentos (PIB 2024):")
    pib_2024 = df[df['anio'] == 2024].nlargest(5, 'pib_total_miles_millones_cop')[
        ['departamento', 'pib_total_miles_millones_cop']
    ]
    for idx, (_, row) in enumerate(pib_2024.iterrows(), 1):
        print(f"   {idx}. {row['departamento']:40s} ${row['pib_total_miles_millones_cop']:12,.0f} M COP")
    
    print(f"\n   Arquivos generados:")
    print(f"   ✓ {dataset_path.name:50s} ({len(df):,} registros)")
    print(f"   ✓ {summary_path.name}")
    print(f"   ✓ {Path(__file__).parent / 'README.md'}")
    
    print(f"\n💡 Próximos pasos:")
    print(f"   1. Usar dataset en supervised/main.py:")
    print(f"      df = pd.read_csv('data/data_simulado_1980_2024.csv')")
    print(f"   2. Ejecutar pipeline supervisado:")
    print(f"      python main.py")
    print(f"   3. Revisar visualizaciones en reports/")
    
    print("\n" + "="*80)
    print("✅ INICIALIZACIÓN COMPLETADA")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    df = main()
