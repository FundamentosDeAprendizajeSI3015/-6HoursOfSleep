"""
main.py
=======
Script principal que orquesta todo el pipeline supervisado.
Responsable: Juanes | Rama: feature/supervised-models

Flujo completo:
1. Cargar datos desde dataset_con_indice.csv
2. Análisis exploratorio básico
3. Preparar datos (split train/test)
4. Entrenar todos los modelos con validación cruzada
5. Evaluar en test set
6. Generar visualizaciones
7. Exportar resultados

Ejecución:
    python main.py
    
Requisitos:
    - pandas, numpy
    - scikit-learn
    - matplotlib
"""

import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_loader import load_and_prepare
from Models import preparar_datos, entrenar_todos_modelos, predecir_test
from Evaluation import (
    evaluar_todos,
    plot_predicho_vs_real,
    plot_residuales,
    plot_comparacion_metricas,
    plot_importancia_variables,
    guardar_tabla_resultados,
    generar_reporte
)


def main():
    """
    Ejecuta el pipeline completo de modelado supervisado.
    """
    
    print("\n" + "="*70)
    print("🎯 PIPELINE DE MODELADO SUPERVISADO - DESERCIÓN ESTUDIANTIL")
    print("="*70)
    
    # ──────────────────────────────────────────────────────────────────────────
    # PASO 1: CARGAR DATOS
    # ──────────────────────────────────────────────────────────────────────────
    
    print("\n\n[1/7] CARGA DE DATOS")
    print("-"*70)
    
    X, y, loader = load_and_prepare()
    
    # Mostrar resumen
    print(loader.get_summary())
    
    # ──────────────────────────────────────────────────────────────────────────
    # PASO 2: ANÁLISIS EXPLORATORIO BÁSICO
    # ──────────────────────────────────────────────────────────────────────────
    
    print("\n[2/7] ANÁLISIS EXPLORATORIO")
    print("-"*70)
    
    stats = loader.check_data_quality()
    
    print(f"\n📈 Estadísticas de features:")
    print(f"   Dimensión: {X.shape[0]} muestras × {X.shape[1]} features")
    print(f"   Target variable: 'outcome_tasa_desercion_snies'")
    
    # Verificar correlaciones básicas
    import pandas as pd
    df_temp = pd.concat([X, y.rename("target")], axis=1)
    
    # Top 5 features más correlacionadas con target
    correlaciones = df_temp.corr()["target"].drop("target").abs().sort_values(ascending=False)
    print(f"\n   Top 5 features (correlación con target):")
    for i, (feat, corr) in enumerate(correlaciones.head(5).items(), 1):
        print(f"      {i}. {feat:40s} → {corr:.4f}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # PASO 3: PREPARACIÓN DE DATOS (SPLIT TRAIN/TEST)
    # ──────────────────────────────────────────────────────────────────────────
    
    print("\n\n[3/7] PREPARACIÓN DE DATOS")
    print("-"*70)
    
    X_train, X_test, y_train, y_test = preparar_datos(X, y)
    
    # ──────────────────────────────────────────────────────────────────────────
    # PASO 4: ENTRENAMIENTO CON VALIDACIÓN CRUZADA
    # ──────────────────────────────────────────────────────────────────────────
    
    print("\n\n[4/7] ENTRENAMIENTO DE MODELOS")
    print("-"*70)
    
    modelos_entrenados, df_cv_results = entrenar_todos_modelos(X_train, y_train)
    
    # ──────────────────────────────────────────────────────────────────────────
    # PASO 5: EVALUACIÓN EN TEST SET
    # ──────────────────────────────────────────────────────────────────────────
    
    print("\n\n[5/7] EVALUACIÓN EN TEST SET")
    print("-"*70)
    
    df_metricas = evaluar_todos(modelos_entrenados, X_test.values, y_test.values)
    
    # ──────────────────────────────────────────────────────────────────────────
    # PASO 6: VISUALIZACIONES
    # ──────────────────────────────────────────────────────────────────────────
    
    print("\n\n[6/7] GENERACIÓN DE VISUALIZACIONES")
    print("-"*70)
    
    print("\n📊 Creando gráficos...")
    
    # Plots principales
    plot_predicho_vs_real(modelos_entrenados, X_test.values, y_test.values, guardar=True)
    plot_residuales(modelos_entrenados, X_test.values, y_test.values, guardar=True)
    plot_comparacion_metricas(df_metricas, guardar=True)
    
    # Feature importance (si aplica)
    if "RandomForest" in modelos_entrenados:
        print("\n📈 Generando feature importance para Random Forest...")
        plot_importancia_variables(
            modelos_entrenados["RandomForest"],
            X.columns.tolist(),
            "RandomForest",
            top_n=15,
            guardar=True
        )
    
    if "GradientBoosting" in modelos_entrenados:
        print("\n📈 Generando feature importance para Gradient Boosting...")
        plot_importancia_variables(
            modelos_entrenados["GradientBoosting"],
            X.columns.tolist(),
            "GradientBoosting",
            top_n=15,
            guardar=True
        )
    
    # ──────────────────────────────────────────────────────────────────────────
    # PASO 7: EXPORTACIÓN DE RESULTADOS
    # ──────────────────────────────────────────────────────────────────────────
    
    print("\n\n[7/7] EXPORTACIÓN DE RESULTADOS")
    print("-"*70)
    
    df_completo = guardar_tabla_resultados(df_metricas, df_cv_results)
    
    # ──────────────────────────────────────────────────────────────────────────
    # RESUMEN FINAL
    # ──────────────────────────────────────────────────────────────────────────
    
    mejor_modelo = df_metricas.index[0]
    generar_reporte(df_metricas, mejor_modelo)
    
    print("\n\n" + "="*70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70)
    print("\n📁 Archivos generados:")
    print("   • Gráficos: supervised/reports/eda_figures/supervisado_*.png")
    print("   • Tabla:    supervised/reports/tabla_comparativa_modelos_supervisados.csv")
    print("\n💡 Próximos pasos:")
    print("   1. Revisar visualizaciones en reports/eda_figures/")
    print("   2. Analizar feature importance")
    print("   3. Considerar tuning de hiperparámetros para modelo ganador")
    print("   4. Documentar conclusiones en el README")
    print("\n" + "="*70 + "\n")
    
    return {
        "modelos": modelos_entrenados,
        "metricas": df_metricas,
        "cv_results": df_cv_results,
        "X_test": X_test,
        "y_test": y_test
    }


if __name__ == "__main__":
    resultados = main()
