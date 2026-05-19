"""
supervised.src
==============
Módulo de modelado supervisado para predicción de deserción estudiantil.

Componentes:
  • data_loader: Carga y limpieza de datos
  • models: Definición y entrenamiento de modelos
  • evaluation: Métricas y visualizaciones

Uso típico:
    from src.data_loader import load_and_prepare
    from src.models import preparar_datos, entrenar_todos_modelos
    from src.evaluation import evaluar_todos
    
    X, y, loader = load_and_prepare()
    X_train, X_test, y_train, y_test = preparar_datos(X, y)
    modelos, cv_results = entrenar_todos_modelos(X_train, y_train)
    metricas = evaluar_todos(modelos, X_test.values, y_test.values)
"""

__version__ = "1.0.0"
__author__ = "Juanes"
__email__ = "juanes@example.com"

# Importaciones principales para conveniencia
try:
    from .data_loader import DataLoader, load_and_prepare
    from .models import (
        preparar_datos,
        entrenar_todos_modelos,
        get_todos_los_modelos,
        predecir_test
    )
    from .evaluation import (
        evaluar_todos,
        plot_predicho_vs_real,
        plot_residuales,
        plot_comparacion_metricas,
        plot_importancia_variables
    )
except ImportError as e:
    print(f"Advertencia: No se pudieron importar todos los módulos: {e}")

__all__ = [
    "DataLoader",
    "load_and_prepare",
    "preparar_datos",
    "entrenar_todos_modelos",
    "get_todos_los_modelos",
    "predecir_test",
    "evaluar_todos",
    "plot_predicho_vs_real",
    "plot_residuales",
    "plot_comparacion_metricas",
    "plot_importancia_variables"
]
