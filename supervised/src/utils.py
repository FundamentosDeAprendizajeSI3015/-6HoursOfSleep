"""
utils.py
========
Funciones auxiliares y utilidades para el pipeline supervisado.
Responsable: Juanes

Incluye:
- Funciones de formato y reportes
- Ayudas para validación de datos
- Utilidades de exportación
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def crear_directorio_salida(base_path: str = "./reports") -> Path:
    """
    Crea estructura de directorios para salidas si no existe.
    
    Args:
        base_path: Ruta base para reports
    
    Returns:
        Path al directorio de figuras
    """
    reports_dir = Path(base_path)
    figures_dir = reports_dir / "eda_figures"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    return figures_dir


def imprimir_encabezado(titulo: str, ancho: int = 70):
    """
    Imprime un encabezado formateado.
    
    Args:
        titulo: Texto del encabezado
        ancho: Ancho total de línea
    """
    print("\n" + "="*ancho)
    print(f"  {titulo}")
    print("="*ancho)


def imprimir_subseccion(titulo: str, ancho: int = 70):
    """
    Imprime una subsección formateada.
    
    Args:
        titulo: Texto de la subsección
        ancho: Ancho total de línea
    """
    print("\n" + "-"*ancho)
    print(f"  {titulo}")
    print("-"*ancho)


def resumen_datos(X: pd.DataFrame, y: pd.Series):
    """
    Imprime resumen estadístico de datos.
    
    Args:
        X: DataFrame de features
        y: Series de target
    """
    print("\n RESUMEN DE DATOS")
    print(f"   Muestras: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target: {y.name}")
    print(f"\n   Estadísticas del Target:")
    print(f"      Media:    {y.mean():.4f}")
    print(f"      Mediana:  {y.median():.4f}")
    print(f"      Std:      {y.std():.4f}")
    print(f"      Mín:      {y.min():.4f}")
    print(f"      Máx:      {y.max():.4f}")
    print(f"\n   Features con nulos:")
    
    nulos = X.isnull().sum()
    if nulos.sum() > 0:
        for col in nulos[nulos > 0].index:
            pct = 100 * nulos[col] / len(X)
            print(f"      {col}: {pct:.1f}%")
    else:
        print(f"      Ninguno ✓")


def tabla_modelos_formateada(df_metricas: pd.DataFrame) -> str:
    """
    Genera tabla formateada de resultados de modelos.
    
    Args:
        df_metricas: DataFrame con métricas
    
    Returns:
        String formateado para imprimir
    """
    tabla = "\n"
    tabla += f"{'Modelo':<20} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8} | {'MAPE':>8}\n"
    tabla += "-" * 60 + "\n"
    
    for modelo, row in df_metricas.iterrows():
        tabla += f"{modelo:<20} | {row['RMSE']:>8.4f} | {row['MAE']:>8.4f} | {row['R2']:>8.4f} | {row['MAPE']:>8.2f}\n"
    
    tabla += "-" * 60
    return tabla


def comparar_modelos(df_test: pd.DataFrame, df_cv: pd.DataFrame = None) -> pd.DataFrame:
    """
    Crea tabla comparativa de modelos combinando test y CV results.
    
    Args:
        df_test: Métricas de test set
        df_cv: Métricas de validación cruzada (opcional)
    
    Returns:
        DataFrame combinado y formateado
    """
    if df_cv is not None:
        # Combinar
        df_combo = df_test.copy()
        df_combo["CV_RMSE"] = df_cv["rmse_mean"]
        df_combo["CV_Std"] = df_cv["rmse_std"]
        return df_combo
    else:
        return df_test


def guardar_informe_texto(modelos: dict,
                         df_metricas: pd.DataFrame,
                         archivo_salida: str = "./reports/informe_supervisado.txt"):
    """
    Guarda un informe de texto con resultados principales.
    
    Args:
        modelos: Dict de modelos entrenados
        df_metricas: DataFrame con métricas
        archivo_salida: Ruta de salida
    """
    Path(archivo_salida).parent.mkdir(parents=True, exist_ok=True)
    
    mejor_modelo = df_metricas.index[0]
    
    informe = f"""
╔════════════════════════════════════════════════════════════════════════╗
║        INFORME FINAL — MODELADO SUPERVISADO DE DESERCIÓN             ║
╚════════════════════════════════════════════════════════════════════════╝

Fecha Generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MODELO GANADOR: {mejor_modelo}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Métricas en Test Set:
  ├─ RMSE: {df_metricas.loc[mejor_modelo, 'RMSE']:.4f}
  ├─ MAE:  {df_metricas.loc[mejor_modelo, 'MAE']:.4f}
  ├─ R²:   {df_metricas.loc[mejor_modelo, 'R2']:.4f}
  └─ MAPE: {df_metricas.loc[mejor_modelo, 'MAPE']:.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TOP 5 MODELOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tabla_modelos_formateada(df_metricas.head(5))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MEJORA VS BASELINE (OLS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Baseline RMSE (OLS):      {df_metricas.loc['OLS', 'RMSE']:.4f}
Mejor RMSE ({mejor_modelo:20s}): {df_metricas.loc[mejor_modelo, 'RMSE']:.4f}

Mejora:
  ├─ Diferencia: {df_metricas.loc['OLS', 'RMSE'] - df_metricas.loc[mejor_modelo, 'RMSE']:.4f}
  └─ Porcentaje: {100*(df_metricas.loc['OLS', 'RMSE'] - df_metricas.loc[mejor_modelo, 'RMSE'])/df_metricas.loc['OLS', 'RMSE']:+.2f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[OK] COMPLETADO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ 8 modelos entrenados (4 baseline + 4 avanzados)
✓ Validación cruzada 5-fold
✓ Evaluación en test set
✓ Visualizaciones generadas
✓ Tabla comparativa exportada
✓ Informe generado

Archivos:  ├─ ./reports/eda_figures/supervisado_*.png
  ├─ ./reports/tabla_comparativa_modelos_supervisados.csv
  └─ ./reports/informe_supervisado.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write(informe)
    
    print(f"[OK] Informe guardado: {archivo_salida}")
    return informe


if __name__ == "__main__":
    # Ejemplo de uso
    print("[OK] Módulo utils.py cargado correctamente")
