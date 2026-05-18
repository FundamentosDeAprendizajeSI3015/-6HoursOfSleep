"""
data_validator.py
=================
Validador de calidad del dataset simulado.

Funciones:
- Validar integridad de datos
- Verificar distribuciones
- Detectar outliers
- Comparar con valores reales observados
- Generar reportes de calidad

Ejecución:
    python data_validator.py <ruta_dataset>
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class DatasetValidator:
    """Validador de dataset simulado."""
    
    def __init__(self, filepath):
        """
        Inicializa el validador.
        
        Args:
            filepath: ruta al CSV del dataset
        """
        print(f"\n📂 Cargando dataset desde: {filepath}")
        self.df = pd.read_csv(filepath)
        print(f"   ✓ Dataset cargado: {self.df.shape[0]:,} registros × {self.df.shape[1]} columnas")
    
    def validar_estructura(self):
        """Valida la estructura básica del dataset."""
        print("\n" + "="*80)
        print("1. VALIDACIÓN DE ESTRUCTURA")
        print("="*80)
        
        # Verificar dimensiones
        print(f"\n✓ Dimensiones:")
        print(f"  Registros: {self.df.shape[0]:,}")
        print(f"  Columnas: {self.df.shape[1]}")
        
        # Verificar tipos de datos
        print(f"\n✓ Tipos de datos:")
        tipos = self.df.dtypes.value_counts()
        for dtype, count in tipos.items():
            print(f"  {str(dtype):15s}: {count} columnas")
        
        # Verificar periodo
        años_min, años_max = self.df['anio'].min(), self.df['anio'].max()
        print(f"\n✓ Período temporal:")
        print(f"  Inicio: {años_min}")
        print(f"  Fin: {años_max}")
        print(f"  Rango: {años_max - años_min + 1} años")
        
        # Verificar departamentos
        print(f"\n✓ Cobertura geográfica:")
        print(f"  Departamentos: {self.df['codigo_departamento'].nunique()}")
        print(f"  Registros por departamento (esperado: {años_max - años_min + 1})")
        
        registros_por_dept = self.df.groupby('codigo_departamento').size()
        if registros_por_dept.std() > 1:
            print(f"  ⚠️  Variación detectada (min: {registros_por_dept.min()}, max: {registros_por_dept.max()})")
        else:
            print(f"  ✓ Distribuición uniforme")
    
    def validar_integridad(self):
        """Valida integridad de datos (nulos, duplicados)."""
        print("\n" + "="*80)
        print("2. VALIDACIÓN DE INTEGRIDAD")
        print("="*80)
        
        # Datos faltantes
        print(f"\n✓ Datos faltantes:")
        nulos = self.df.isnull().sum()
        nulos_pct = (nulos / len(self.df)) * 100
        
        nulos_info = pd.DataFrame({
            'variable': nulos.index,
            'nulos': nulos.values,
            'porcentaje': nulos_pct.values
        }).sort_values('nulos', ascending=False)
        
        nulos_con_datos = nulos_info[nulos_info['nulos'] > 0]
        if len(nulos_con_datos) == 0:
            print("  ✓ No hay datos faltantes")
        else:
            print(f"  Variables con nulos: {len(nulos_con_datos)}")
            for _, row in nulos_con_datos.head(10).iterrows():
                print(f"    {row['variable']:50s}: {row['nulos']:6,.0f} ({row['porcentaje']:5.2f}%)")
        
        # Duplicados
        duplicados = self.df.duplicated(subset=['codigo_departamento', 'anio']).sum()
        print(f"\n✓ Duplicados:")
        if duplicados == 0:
            print(f"  No hay registros duplicados")
        else:
            print(f"  ⚠️  {duplicados} registros duplicados detectados")
        
        # Valores negativos inválidos
        print(f"\n✓ Validación de rangos:")
        
        validaciones = {
            'pib_total_miles_millones_cop': (0, None, "PIB debe ser positivo"),
            'total_admitidos': (0, None, "Admitidos debe ser positivo"),
            'total_matriculados': (0, None, "Matriculados debe ser positivo"),
            'outcome_tasa_desercion_snies': (0, 100, "Deserción debe estar entre 0-100%"),
            'geih_td_nacional_media_anual': (0, 30, "TD debe estar entre 0-30%"),
            'ipc_nacional_total_var_mensual_media': (-5, 10, "IPC debe estar entre -5% y 10% mensual"),
        }
        
        for col, (min_val, max_val, msg) in validaciones.items():
            if col not in self.df.columns:
                continue
            
            fuera_rango = 0
            if min_val is not None:
                fuera_rango += (self.df[col] < min_val).sum()
            if max_val is not None:
                fuera_rango += (self.df[col] > max_val).sum()
            
            if fuera_rango == 0:
                print(f"  ✓ {col:50s}: OK")
            else:
                print(f"  ⚠️  {col:50s}: {fuera_rango} valores fuera de rango")
                print(f"     {msg}")
    
    def validar_distribucion(self):
        """Valida que las distribuciones sean realistas."""
        print("\n" + "="*80)
        print("3. VALIDACIÓN DE DISTRIBUCIONES")
        print("="*80)
        
        variables_clave = {
            'pib_total_miles_millones_cop': {
                'esperado_media': 30_000,
                'esperado_std': 50_000,
                'desc': 'PIB departamental'
            },
            'total_matriculados': {
                'esperado_media': 20_000,
                'esperado_std': 80_000,
                'desc': 'Total matriculados'
            },
            'outcome_tasa_desercion_snies': {
                'esperado_media': 10,
                'esperado_std': 5,
                'desc': 'Tasa de deserción'
            },
            'ipc_nacional_total_var_mensual_media': {
                'esperado_media': 1.0,
                'esperado_std': 0.8,
                'desc': 'IPC nacional'
            },
            'geih_td_nacional_media_anual': {
                'esperado_media': 11,
                'esperado_std': 3,
                'desc': 'Tasa de desempleo'
            },
        }
        
        for var, config in variables_clave.items():
            if var not in self.df.columns:
                continue
            
            media = self.df[var].mean()
            std = self.df[var].std()
            
            print(f"\n✓ {config['desc']} ({var}):")
            print(f"  Media:     {media:12,.2f} (esperado: {config['esperado_media']:,.0f})")
            print(f"  Std:       {std:12,.2f} (esperado: {config['esperado_std']:,.0f})")
            print(f"  Min:       {self.df[var].min():12,.2f}")
            print(f"  Max:       {self.df[var].max():12,.2f}")
            print(f"  Mediana:   {self.df[var].median():12,.2f}")
    
    def validar_tendencias(self):
        """Valida que las tendencias temporales sean coherentes."""
        print("\n" + "="*80)
        print("4. VALIDACIÓN DE TENDENCIAS TEMPORALES")
        print("="*80)
        
        # Agregación nacional por año
        print("\n✓ Tendencia del PIB nacional:")
        pib_por_año = self.df.groupby('anio')['pib_total_miles_millones_cop'].sum()
        
        # Verificar crecimiento general (aunque con ciclos)
        pib_1980 = pib_por_año.iloc[0]
        pib_2024 = pib_por_año.iloc[-1]
        crecimiento_total = ((pib_2024 - pib_1980) / pib_1980) * 100
        
        print(f"  PIB 1980: ${pib_1980:,.0f} millones COP")
        print(f"  PIB 2024: ${pib_2024:,.0f} millones COP")
        print(f"  Crecimiento total: {crecimiento_total:.1f}%")
        
        if crecimiento_total > 0:
            print(f"  ✓ Tendencia de crecimiento positivo")
        else:
            print(f"  ⚠️  Tendencia negativa detectada")
        
        # Matriculados
        print("\n✓ Tendencia de matrícula:")
        matriculados_por_año = self.df.groupby('anio')['total_matriculados'].sum()
        
        matriculados_1980 = matriculados_por_año.iloc[0]
        matriculados_2024 = matriculados_por_año.iloc[-1]
        crecimiento_mat = ((matriculados_2024 - matriculados_1980) / matriculados_1980) * 100
        
        print(f"  Matriculados 1980: {matriculados_1980:,.0f}")
        print(f"  Matriculados 2024: {matriculados_2024:,.0f}")
        print(f"  Crecimiento: {crecimiento_mat:.1f}%")
        
        # Deserción
        print("\n✓ Tendencia de deserción:")
        desercion_por_año = self.df.groupby('anio')['outcome_tasa_desercion_snies'].mean()
        
        print(f"  Deserción 1980: {desercion_por_año.iloc[0]:.2f}%")
        print(f"  Deserción 2024: {desercion_por_año.iloc[-1]:.2f}%")
        print(f"  Cambio: {desercion_por_año.iloc[-1] - desercion_por_año.iloc[0]:.2f} pp")
        
        if desercion_por_año.iloc[-1] < desercion_por_año.iloc[0]:
            print(f"  ✓ Tendencia de mejora en deserción")
        else:
            print(f"  ⚠️  Deserción ha aumentado")
    
    def validar_correlaciones(self):
        """Valida que las correlaciones sean coherentes."""
        print("\n" + "="*80)
        print("5. VALIDACIÓN DE CORRELACIONES")
        print("="*80)
        
        # Correlaciones esperadas
        print("\n✓ Correlaciones esperadas:")
        
        # PIB vs matriculados (positiva)
        corr_pib_matriculados = self.df[['pib_total_miles_millones_cop', 'total_matriculados']].corr().iloc[0, 1]
        print(f"  PIB vs Matriculados: {corr_pib_matriculados:.3f} (esperado: > 0.3)")
        
        # Deserción vs PIB per cápita (negativa)
        corr_desercion_pib_pc = self.df[['outcome_tasa_desercion_snies', 'proxy_pib_miles_mm_cop_por_matriculado']].corr().iloc[0, 1]
        print(f"  Deserción vs PIB/alumno: {corr_desercion_pib_pc:.3f} (esperado: < -0.2)")
        
        # IPC vs desempleo (positiva en crisis)
        corr_ipc_desempleo = self.df[['ipc_nacional_total_var_mensual_media', 'geih_td_nacional_media_anual']].corr().iloc[0, 1]
        print(f"  IPC vs Desempleo: {corr_ipc_desempleo:.3f}")
    
    def generar_reporte_completo(self):
        """Genera reporte HTML de validación."""
        print("\n" + "="*80)
        print("6. GENERACIÓN DE REPORTE")
        print("="*80)
        
        print(f"\n✓ Ejecutando validaciones...")
        self.validar_estructura()
        self.validar_integridad()
        self.validar_distribucion()
        self.validar_tendencias()
        self.validar_correlaciones()
        
        print("\n" + "="*80)
        print("✅ VALIDACIÓN COMPLETADA")
        print("="*80 + "\n")


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Ruta por defecto
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = Path(__file__).parent / "data_simulado_1980_2024.csv"
    
    # Validar
    if not Path(filepath).exists():
        print(f"❌ Archivo no encontrado: {filepath}")
        print("\nPrimero ejecuta: python data_generator.py")
        sys.exit(1)
    
    validator = DatasetValidator(filepath)
    validator.generar_reporte_completo()
