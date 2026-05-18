import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class DataLoader:
    def __init__(self, data_path=None):
        self.df_raw = None
        self.df_clean = None
        self.features = None
        self.target = "outcome_tasa_desercion_snies"
        self.excluded_cols = []
        
        if data_path is None:
            data_path = self._find_data_file()
        
        self._load_data(data_path)
    
    def _find_data_file(self):
        possible_paths = [
            Path(r"c:\Users\juane\OneDrive\Documentos\EAFIT\2026-01\Aprendizaje Automatico\PROYECTO FINAL\-6HoursOfSleep\data_simulada\processed\data_simulado_1980_2026.csv"),
            Path("../../data_simulada/processed/data_simulado_1980_2026.csv"),
            Path("../data_simulada/processed/data_simulado_1980_2026.csv"),
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    df_check = pd.read_csv(path, nrows=1)
                    if 'mes' in df_check.columns:
                        n_rows = pd.read_csv(path).shape[0]
                        print(f"OK Dataset MENSUAL: {path}")
                        print(f"Registros: {n_rows:,}")
                        return str(path.absolute())
                except Exception:
                    continue
        
        raise FileNotFoundError("No se encontro el dataset mensual")
    
    def _load_data(self, data_path):
        print(f"\nCargando datos desde: {data_path}")
        self.df_raw = pd.read_csv(data_path)
        print(f"Dimensiones: {self.df_raw.shape[0]:,} filas x {self.df_raw.shape[1]} columnas")
        
        year_col = 'year' if 'year' in self.df_raw.columns else 'anio'
        if year_col in self.df_raw.columns:
            unique_years = sorted(self.df_raw[year_col].unique())
            print(f"Anos: {unique_years[0]}-{unique_years[-1]} ({len(unique_years)} anos)")
        
        if 'mes' in self.df_raw.columns:
            print(f"Meses: 1-12 (datos mensuales)")
        
        print(f"Departamentos: {self.df_raw['departamento'].nunique()}")
    
    def clean_data(self):
        print("\nIniciando limpieza...")
        self.df_clean = self.df_raw.copy()
        
        leakage_cols = ["spadies_td_anual_universitario"]
        for col in leakage_cols:
            if col in self.df_clean.columns:
                self.df_clean.drop(col, axis=1, inplace=True)
                self.excluded_cols.append(f"{col} (data leakage)")
                print(f"Eliminado: {col}")
        
        tech_cols = ["outcome_merge_pendiente"]
        for col in tech_cols:
            if col in self.df_clean.columns:
                self.df_clean.drop(col, axis=1, inplace=True)
                self.excluded_cols.append(f"{col} (tecnica)")
                print(f"Eliminado: {col}")
        
        if self.target not in self.df_clean.columns:
            raise ValueError(f"Target '{self.target}' no encontrado")
        
        target_nulls = self.df_clean[self.target].isna().sum()
        if target_nulls > 0:
            self.df_clean = self.df_clean[self.df_clean[self.target].notna()].copy()
        
        duplicates = self.df_clean.duplicated().sum()
        if duplicates > 0:
            self.df_clean.drop_duplicates(inplace=True)
        
        print(f"Dataset limpio: {self.df_clean.shape[0]:,} filas x {self.df_clean.shape[1]} columnas")
        return self.df_clean
    
    def prepare_features(self, exclude_cols=None):
        standard_exclude = [
            "departamento", "codigo_departamento", "anio", "year", 
            "departamento_id", "mes", "outcome_merge_pendiente"
        ]
        if exclude_cols:
            standard_exclude.extend(exclude_cols)
        
        null_pct = (self.df_clean.isnull().sum() / len(self.df_clean)) * 100
        cols_high_null = null_pct[null_pct > 50].index.tolist()
        
        if cols_high_null:
            print(f"Eliminando {len(cols_high_null)} columnas con >50% nulos")
            standard_exclude.extend(cols_high_null)
        
        self.features = [col for col in self.df_clean.columns 
                        if col not in standard_exclude and col != self.target]
        
        X = self.df_clean[self.features].copy()
        y = self.df_clean[self.target].copy()
        
        print(f"Features seleccionadas: {len(self.features)}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        return X, y
    
    def check_data_quality(self):
        print("\nAnalisis de calidad de datos:")
        
        stats = {
            "total_rows": len(self.df_clean),
            "total_cols": len(self.df_clean.columns),
            "target_stats": {
                "media": self.df_clean[self.target].mean(),
                "std": self.df_clean[self.target].std(),
                "min": self.df_clean[self.target].min(),
                "max": self.df_clean[self.target].max(),
            }
        }
        
        print(f"Target media: {stats['target_stats']['media']:.4f}")
        print(f"Target std: {stats['target_stats']['std']:.4f}")
        
        return stats
    
    def get_summary(self):
        summary = f"""
=========================================
RESUMEN DE CARGA Y LIMPIEZA
=========================================
Dataset Original:  {self.df_raw.shape[0]:,} filas
Dataset Limpio:    {self.df_clean.shape[0]:,} filas
Features: {len(self.features)}
Target: {self.target}
=========================================
        """
        return summary


def load_and_prepare(data_path=None):
    loader = DataLoader(data_path)
    loader.clean_data()
    loader.check_data_quality()
    X, y = loader.prepare_features()
    print(loader.get_summary())
    
    return X, y, loader


if __name__ == "__main__":
    X, y, loader = load_and_prepare()
    print("OK - Datos listos!")
