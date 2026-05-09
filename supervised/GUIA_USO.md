GUIA DE USO - PIPELINE SUPERVISADO
===================================

ESTRUCTURA DEL PROYECTO
-----------------------

supervised/
    01_pipeline_completo.ipynb     Notebook interactivo con todo el pipeline
    main.py                         Script ejecutable del pipeline
    INDEX.md                        Indice de documentacion
    
    Readmes/                        Documentacion completa
        README.md                   Descripcion general
        QUICK_REFERENCE.md          Guia rapida
        CONCLUSIONES.md             Analisis detallado
        RESUMEN_FINAL.md            Estado del proyecto
    
    src/                            Codigo fuente
        data_loader.py              Carga y limpieza de datos
        Models.py                   8 modelos de regresion
        Evaluation.py               Evaluacion y visualizaciones
        utils.py                    Utilidades varias
    
    reports/                        Resultados generados
        informe_supervisado.txt     Reporte final
        tabla_comparativa_modelos_supervisados.csv
        eda_figures/                Visualizaciones


COMO EJECUTAR
-------------

OPCION 1: JUPYTER NOTEBOOK (Interactivo - Recomendado)
    1. Abre: 01_pipeline_completo.ipynb
    2. Ejecuta celdas secuencialmente
    3. Ver graficos y resultados en tiempo real

OPCION 2: Script Python (Batch)
    cd supervised
    python main.py

OPCION 3: Imports en otro script
    import sys
    sys.path.insert(0, 'supervised/src')
    from data_loader import load_and_prepare
    from Models import entrenar_todos_modelos
    from Evaluation import evaluar_todos


RESULTADOS GENERADOS
--------------------

Tabla comparativa de modelos:
    reports/tabla_comparativa_modelos_supervisados.csv

Reporte de texto:
    reports/informe_supervisado.txt

Visualizaciones (5 graficos):
    1. Predicho vs Real
    2. Analisis de residuales
    3. Comparacion de metricas (bar chart)
    4. Feature Importance - RandomForest
    5. Feature Importance - GradientBoosting


MODELOS INCLUIDOS (8 TOTALES)
-----------------------------

BASELINE (4 lineales):
    1. OLS                 Regresion ordinaria (sin regularizacion)
    2. Ridge (L2)          Regularizacion L2 (mantiene todas variables)
    3. Lasso (L1)          Regularizacion L1 (selecciona variables)
    4. ElasticNet          Combinacion L1+L2 (balance entre ambas)

AVANZADOS (4 no-lineales):
    5. RandomForest        Ensemble de 100 arboles de decision
    6. GradientBoosting    Ensemble secuencial (boosting)
    7. SVR                 Support Vector Machine (kernel RBF)
    8. KNN                 K-Nearest Neighbors (k=5)


METRICAS DE EVALUACION
----------------------

- RMSE: Root Mean Squared Error (error cuadratico medio)
- MAE: Mean Absolute Error (error absoluto medio)
- R²: Coeficiente de determinacion (proporcion varianza explicada)
- MAPE: Mean Absolute Percentage Error (% error promedio)

Validacion Cruzada: 5-fold CV en training set
Test Set: 20% de datos separados


PARA ENTENDER EL CODIGO
-----------------------

1. Lee: Readmes/README.md
   - Explicacion general
   - Justificacion de cada modelo

2. Lee: src/Models.py
   - Comentarios detallados en cada funcion
   - Explicacion de por que cada modelo

3. Lee: Readmes/CONCLUSIONES.md
   - Analisis de resultados
   - Limitaciones y proximos pasos


NOTAS IMPORTANTES
-----------------

- Dataset: 33 muestras (despues de limpieza)
- Features: 32 variables
- Target: Tasa de desercion estudiantil
- Random State: 42 (reproducibilidad garantizada)
- Los emojis fueron removidos del codigo
- Rutas ajustadas a estructura nueva (reports dentro de supervised)


TROUBLESHOOTING
---------------

Problema: ModuleNotFoundError en imports
Solucion: Asegura que estés en la carpeta supervised/
    cd supervised
    python main.py

Problema: FileNotFoundError para dataset
Solucion: El dataset debe estar en ../data/final/dataset_con_indice.csv
    desde la carpeta supervised/

Problema: Errores de joblib en Windows
Solucion: Ya esta configurado en Models.py con n_jobs=1


PARAMETROS CONFIGURABLES
------------------------

En src/Models.py:
    RANDOM_STATE = 42      Cambiar para diferentes splits
    TEST_SIZE = 0.20       Cambiar proporcion train/test
    CV_FOLDS = 5           Cambiar folds de validacion cruzada

En src/Evaluation.py:
    top_n = 15             Variables mostradas en feature importance
    figsize = (12, 8)      Tamaño de figuras


CONTACTO Y SOPORTE
------------------

Responsable: Juanes
Rama: feature/supervised-models
Fecha creacion: Mayo 5, 2026
