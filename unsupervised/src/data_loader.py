"""
data_loader.py
==============
Carga y prepara datos para el pipeline no supervisado.
Responsable: Equipo no supervisado

Funcionalidades:
- Cargar dataset desde data/final/dataset_con_indice.csv
- Eliminar columnas de data leakage y variables técnicas
- Seleccionar features numéricas útiles
- Imputar nulos con la media
- Retornar datos listos para clustering
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class DataLoader:
    """Gestor de carga y limpieza para el pipeline no supervisado."""

    def __init__(self, data_path: str = None):
        self.df_raw = None
        self.df_clean = None
        self.features = None
        self.target = "outcome_tasa_desercion_snies"
        self.excluded_cols = []

        if data_path is None:
            data_path = self._find_data_file()

        self._load_data(data_path)

    def _find_data_file(self) -> str:
        possible_paths = [
            Path("data/final/dataset_con_indice.csv"),
            Path("../data/final/dataset_con_indice.csv"),
            Path("../../data/final/dataset_con_indice.csv"),
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        raise FileNotFoundError(
            "No se encontró dataset_con_indice.csv. Asegúrate de ejecutar desde la ruta raíz del proyecto."
        )

    def _load_data(self, data_path: str) -> None:
        print(f"\n[INFO] Cargando datos desde: {data_path}")
        self.df_raw = pd.read_csv(data_path)
        print(f"   [OK] Dimensiones: {self.df_raw.shape[0]} filas × {self.df_raw.shape[1]} columnas")

    def clean_data(self) -> pd.DataFrame:
        """Limpia el dataset eliminando leakage y columnas técnicas."""
        print("\n[INFO] Limpiando datos para análisis no supervisado...")
        self.df_clean = self.df_raw.copy()

        leakage_cols = ["spadies_td_anual_universitario"]
        for col in leakage_cols:
            if col in self.df_clean.columns:
                self.df_clean.drop(columns=[col], inplace=True)
                self.excluded_cols.append(f"{col} (data leakage)")
                print(f"   [OK] Eliminado: {col} (data leakage)")

        tech_cols = ["outcome_merge_pendiente"]
        for col in tech_cols:
            if col in self.df_clean.columns:
                self.df_clean.drop(columns=[col], inplace=True)
                self.excluded_cols.append(f"{col} (columna técnica)")
                print(f"   [OK] Eliminado: {col} (columna técnica)")

        duplicates = self.df_clean.duplicated().sum()
        if duplicates > 0:
            self.df_clean.drop_duplicates(inplace=True)
            print(f"   [OK] Eliminados {duplicates} duplicados exactos")

        return self.df_clean

    def prepare_features(self, exclude_cols: list = None) -> tuple:
        """Selecciona features numéricas para clustering y devuelve datos listos."""
        standard_exclude = [
            "departamento",
            "codigo_departamento",
            "anio",
            "year",
            "departamento_id",
            self.target,
        ]
        if exclude_cols:
            standard_exclude.extend(exclude_cols)

        null_pct = (self.df_clean.isnull().sum() / len(self.df_clean)) * 100
        cols_high_null = null_pct[null_pct > 50].index.tolist()
        if cols_high_null:
            print(f"\n[WARNING] Eliminando {len(cols_high_null)} columnas con >50% nulos:")
            for col in cols_high_null:
                print(f"      {col}: {null_pct[col]:.1f}%")
            standard_exclude.extend(cols_high_null)

        self.features = [
            col for col in self.df_clean.columns
            if col not in standard_exclude and pd.api.types.is_numeric_dtype(self.df_clean[col])
        ]

        X = self.df_clean[self.features].copy()
        X = X.fillna(X.mean(numeric_only=True))

        y = None
        if self.target in self.df_clean.columns:
            y = self.df_clean[self.target].copy()

        print(f"\n[SUCCESS] Features seleccionadas: {len(self.features)}")
        print(f"   Dimensión X: {X.shape}")
        if y is not None:
            print(f"   Dimensión y (etiqueta opcional): {y.shape}")

        return X, y, self.df_clean

    def check_data_quality(self) -> dict:
        print("\n[INFO] Analizando calidad de datos...")
        null_cols = self.df_clean.columns[self.df_clean.isnull().sum() > 0]
        if len(null_cols) > 0:
            print("   [WARNING] Columnas con nulos:")
            for col in null_cols:
                pct = 100 * self.df_clean[col].isnull().sum() / len(self.df_clean)
                print(f"      {col}: {pct:.1f}%")
        else:
            print("   [OK] Sin valores nulos detectados")

        stats = {
            "total_rows": len(self.df_clean),
            "total_cols": len(self.df_clean.columns),
            "nulls_por_col": self.df_clean.isnull().sum().to_dict(),
            "dtype_counts": self.df_clean.dtypes.value_counts().to_dict(),
        }
        return stats

    def get_summary(self) -> str:
        summary = f"""
===============================================================
 RESUMEN DE CARGA Y LIMPIEZA NO SUPERVISADA
===============================================================
Dataset Original:  {self.df_raw.shape[0]} filas × {self.df_raw.shape[1]} columnas
Dataset Limpio:    {self.df_clean.shape[0]} filas × {self.df_clean.shape[1]} columnas
Columnas Eliminadas: {len(self.excluded_cols)}
Variable Objetivo (no usada): {self.target}
Número de Features: {len(self.features)}
===============================================================
"""
        return summary


def load_and_prepare(data_path: str = None) -> tuple:
    loader = DataLoader(data_path)
    loader.clean_data()
    loader.check_data_quality()
    X, y, df_clean = loader.prepare_features()
    print(loader.get_summary())
    return X, y, loader


if __name__ == "__main__":
    X, y, loader = load_and_prepare()
    print("\n[OK] Datos listos para clustering")
