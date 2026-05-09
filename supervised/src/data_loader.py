"""
data_loader.py
==============
Carga, limpieza y preparación de datos para modelado supervisado.
Responsable: Juanes | Módulo: data_loader

Funcionalidades:
- Cargar dataset desde CSV
- Validar integridad de datos
- Eliminar data leakage conocido
- Separar features y target
- Documentar cambios realizados
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class DataLoader:
    """
    Gestor de carga y limpieza de datos para el pipeline supervisado.
    
    Atributos:
        df_raw: DataFrame original cargado
        df_clean: DataFrame después de limpieza
        features: Lista de variables predictoras
        target: Nombre de la variable objetivo
    """
    
    def __init__(self, data_path: str = None):
        """
        Inicializa el DataLoader.
        
        Args:
            data_path: Ruta al archivo CSV con datos. Si es None, busca automáticamente.
        """
        self.df_raw = None
        self.df_clean = None
        self.features = None
        self.target = "outcome_tasa_desercion_snies"
        self.excluded_cols = []
        
        if data_path is None:
            data_path = self._find_data_file()
        
        self._load_data(data_path)
    
    def _find_data_file(self) -> str:
        """Busca automáticamente el archivo dataset_con_indice.csv"""
        possible_paths = [
            Path("data/final/dataset_con_indice.csv"),
            Path("../data/final/dataset_con_indice.csv"),
            Path("../../data/final/dataset_con_indice.csv"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            "No se encontró dataset_con_indice.csv. "
            "Asegúrate de estar en el directorio correcto."
        )
    
    def _load_data(self, data_path: str) -> None:
        """Carga el archivo CSV."""
        print(f"\n[INFO] Cargando datos desde: {data_path}")
        self.df_raw = pd.read_csv(data_path)
        print(f"   [OK] Dimensiones: {self.df_raw.shape[0]} filas × {self.df_raw.shape[1]} columnas")
        
        # Detectar columna de año
        year_col = 'year' if 'year' in self.df_raw.columns else 'anio'
        if year_col in self.df_raw.columns:
            print(f"   [OK] Años: {sorted(self.df_raw[year_col].unique())}")
        
        print(f"   [OK] Departamentos únicos: {self.df_raw['departamento'].nunique()}")
    
    def clean_data(self) -> pd.DataFrame:
        """
        Realiza limpieza del dataset según recomendaciones del EDA.
        """
        print("\n Iniciando limpieza de datos...")
        self.df_clean = self.df_raw.copy()
        
        # Eliminar data leakage conocido
        leakage_cols = ["spadies_td_anual_universitario"]
        for col in leakage_cols:
            if col in self.df_clean.columns:
                self.df_clean.drop(col, axis=1, inplace=True)
                self.excluded_cols.append(f"{col} (data leakage)")
                print(f"   [OK] Eliminado: {col} (data leakage)")
        
        # Eliminar columnas técnicas
        tech_cols = ["outcome_merge_pendiente"]
        for col in tech_cols:
            if col in self.df_clean.columns:
                self.df_clean.drop(col, axis=1, inplace=True)
                self.excluded_cols.append(f"{col} (columna técnica)")
                print(f"   [OK] Eliminado: {col} (columna técnica)")
        
        # Verificar target
        if self.target not in self.df_clean.columns:
            raise ValueError(f"Variable objetivo '{self.target}' no encontrada")
        
        # Filtrar filas con target disponible
        target_nulls = self.df_clean[self.target].isna().sum()
        if target_nulls > 0:
            print(f"   [WARNING]  {target_nulls} filas con target nulo - serán excluidas")
            self.df_clean = self.df_clean[self.df_clean[self.target].notna()].copy()
          # Eliminar duplicados exactos
        duplicates = self.df_clean.duplicated().sum()
        if duplicates > 0:
            self.df_clean.drop_duplicates(inplace=True)
            print(f"   [OK] Eliminados {duplicates} duplicados exactos")
        
        print(f"\n    Dataset limpio: {self.df_clean.shape[0]} filas × {self.df_clean.shape[1]} columnas")
        
        return self.df_clean
    
    def prepare_features(self, exclude_cols: list = None) -> tuple:
        """
        Selecciona features para modelado.
        """
        # Columnas que siempre se excluyen
        standard_exclude = ["departamento", "codigo_departamento", "anio", "year", "departamento_id"]
        if exclude_cols:
            standard_exclude.extend(exclude_cols)
        
        # Eliminar columnas con >50% nulos (no tienen información suficiente)
        null_pct = (self.df_clean.isnull().sum() / len(self.df_clean)) * 100
        cols_high_null = null_pct[null_pct > 50].index.tolist()
        
        if cols_high_null:
            print(f"\n[WARNING]  Eliminando {len(cols_high_null)} columnas con >50% nulos:")
            for col in cols_high_null:
                print(f"      {col} ({null_pct[col]:.1f}% nulos)")
            standard_exclude.extend(cols_high_null)
        
        # Seleccionar features
        self.features = [col for col in self.df_clean.columns 
                        if col not in standard_exclude and col != self.target]
        
        X = self.df_clean[self.features].copy()
        y = self.df_clean[self.target].copy()
        
        print(f"\n[SUCCESS] Features seleccionadas: {len(self.features)}")
        print(f"   Dimensión X: {X.shape}")
        print(f"   Dimensión y: {y.shape}")
        
        return X, y
    
    def check_data_quality(self) -> dict:
        """Análisis rápido de la calidad de datos."""
        print("\n Análisis de calidad de datos:")
        
        stats = {
            "total_rows": len(self.df_clean),
            "total_cols": len(self.df_clean.columns),
            "nulls_por_col": self.df_clean.isnull().sum().to_dict(),
            "dtype_counts": self.df_clean.dtypes.value_counts().to_dict(),
            "target_stats": {
                "media": self.df_clean[self.target].mean(),
                "std": self.df_clean[self.target].std(),
                "min": self.df_clean[self.target].min(),
                "max": self.df_clean[self.target].max(),
                "nulls": self.df_clean[self.target].isna().sum()
            }
        }
        
        # Mostrar columnas con muchos nulos
        null_cols = self.df_clean.columns[self.df_clean.isnull().sum() > 0]
        if len(null_cols) > 0:
            print(f"   [WARNING]  Columnas con nulos:")
            for col in null_cols:
                pct = 100 * self.df_clean[col].isna().sum() / len(self.df_clean)
                print(f"      {col}: {pct:.1f}%")
        else:
            print(f"   [OK] Sin valores nulos detectados")
        
        print(f"\n   Target '{self.target}':")
        print(f"      Media: {stats['target_stats']['media']:.4f}")
        print(f"      Std:  {stats['target_stats']['std']:.4f}")
        print(f"      Rango: [{stats['target_stats']['min']:.4f}, {stats['target_stats']['max']:.4f}]")
        
        return stats
    
    def get_clean_data(self) -> pd.DataFrame:
        """Retorna el DataFrame limpio"""
        if self.df_clean is None:
            raise ValueError("Debes ejecutar clean_data() primero")
        return self.df_clean.copy()
    
    def get_summary(self) -> str:
        """Retorna un resumen de la limpieza realizada"""
        summary = f"""
===============================================================
 RESUMEN DE CARGA Y LIMPIEZA DE DATOS
===============================================================
Dataset Original:  {self.df_raw.shape[0]} filas × {self.df_raw.shape[1]} columnas
Dataset Limpio:    {self.df_clean.shape[0]} filas × {self.df_clean.shape[1]} columnas
Columnas Eliminadas: {len(self.excluded_cols)}

Variable Objetivo: {self.target}
Número de Features: {len(self.features)}
===============================================================
        """
        return summary


def load_and_prepare(data_path: str = None) -> tuple:
    """
    Función auxiliar que carga, limpia y prepara datos en una línea.
    """
    loader = DataLoader(data_path)
    loader.clean_data()
    loader.check_data_quality()
    X, y = loader.prepare_features()
    print(loader.get_summary())
    
    return X, y, loader


if __name__ == "__main__":
    X, y, loader = load_and_prepare()
    print("\n[OK] Datos listos para modelado")
