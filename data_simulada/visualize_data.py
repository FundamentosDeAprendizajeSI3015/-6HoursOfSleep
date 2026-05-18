"""
visualize_data.py
=================
Script de visualización exploratoria del dataset simulado.

Genera gráficos:
- Evolución temporal de variables clave
- Distribuciones por departamento
- Correlaciones
- Comparativas 1980 vs 2026

Ejecución:
    python visualize_data.py

Ubicación de salida:
    data_simulada/graficas/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import DATASET_PATH, obtener_ruta_figura, crear_directorios

# Crear directorios necesarios al importar
crear_directorios()

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def cargar_datos():
    """
    Carga el dataset simulado desde la ubicación configurada.
    
    Returns:
        DataFrame con los datos cargados, o None si no existe el archivo
    """
    if not DATASET_PATH.exists():
        print(f"❌ Archivo no encontrado: {DATASET_PATH}")
        print("   Ejecuta primero: python init_data.py")
        print(f"   Ubicación esperada: {DATASET_PATH}")
        return None
    
    print(f"📂 Cargando {DATASET_PATH.name}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"   ✓ {len(df):,} registros cargados\n")
    
    return df


def plot_pib_nacional(df):
    """Grafica evolución del PIB nacional agregado desde 1980 hasta 2026."""
    print("📊 Generando gráfico: PIB Nacional (1980-2026)")
    
    pib_por_año = df.groupby('anio')['pib_total_miles_millones_cop'].sum()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # PIB absoluto
    ax1.plot(pib_por_año.index, pib_por_año.values, linewidth=2.5, color='#1f77b4', marker='o', markersize=4)
    ax1.fill_between(pib_por_año.index, pib_por_año.values, alpha=0.3, color='#1f77b4')
    ax1.set_xlabel('Año')
    ax1.set_ylabel('PIB (miles de millones COP)')
    ax1.set_title('PIB Nacional Simulado (1980-2026)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Annotate key points
    for año, crisis_year in [(1999, 'Crisis Bancaria'), (2008, 'Crisis Global'), (2020, 'COVID-19')]:
        idx = pib_por_año.index.get_loc(año)
        ax1.annotate(crisis_year, xy=(año, pib_por_año.iloc[idx]), 
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Crecimiento porcentual anual
    crecimiento = pib_por_año.pct_change() * 100
    ax2.bar(crecimiento.index, crecimiento.values, color=['red' if x < 0 else 'green' for x in crecimiento.values], alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Año')
    ax2.set_ylabel('Crecimiento (%)')
    ax2.set_title('Crecimiento PIB Anual', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = obtener_ruta_figura("pib_nacional")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path.name}\n")
    plt.close()


def plot_desercion_temporal(df):
    """Grafica evolución de la tasa de deserción."""
    print("📊 Generando gráfico: Evolución de Deserción")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Promedio nacional
    desercion_nacional = df.groupby('anio')['outcome_tasa_desercion_snies'].mean()
    axes[0, 0].plot(desercion_nacional.index, desercion_nacional.values, 
                   linewidth=2.5, color='#d62728', marker='o', markersize=4)
    axes[0, 0].fill_between(desercion_nacional.index, desercion_nacional.values, alpha=0.3, color='#d62728')
    axes[0, 0].set_title('Promedio Nacional de Deserción', fontweight='bold')
    axes[0, 0].set_ylabel('Tasa (%)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Std desviación
    desercion_std = df.groupby('anio')['outcome_tasa_desercion_snies'].std()
    axes[0, 1].fill_between(desercion_std.index, desercion_std.values, alpha=0.5, color='#ff7f0e')
    axes[0, 1].set_title('Variabilidad por Departamento (Std Dev)', fontweight='bold')
    axes[0, 1].set_ylabel('Std Dev (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Top 5 departamentos (deserción más alta 2026)
    df_2026 = df[df['anio'] == 2026].nlargest(5, 'outcome_tasa_desercion_snies')
    axes[1, 0].barh(df_2026['departamento'], df_2026['outcome_tasa_desercion_snies'], 
                   color=plt.cm.Reds(np.linspace(0.4, 0.8, 5)))
    axes[1, 0].set_title('Top 5 Departamentos - Mayor Deserción (2026)', fontweight='bold')
    axes[1, 0].set_xlabel('Tasa (%)')
    
    # 4. Top 5 departamentos (deserción más baja 2026)
    df_2026_low = df[df['anio'] == 2026].nsmallest(5, 'outcome_tasa_desercion_snies')
    axes[1, 1].barh(df_2026_low['departamento'], df_2026_low['outcome_tasa_desercion_snies'], 
                   color=plt.cm.Greens(np.linspace(0.4, 0.8, 5)))
    axes[1, 1].set_title('Top 5 Departamentos - Menor Deserción (2026)', fontweight='bold')
    axes[1, 1].set_xlabel('Tasa (%)')
    
    plt.tight_layout()
    output_path = obtener_ruta_figura("desercion_temporal")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path.name}\n")
    plt.close()


def plot_empleo_variables(df):
    """Grafica variables de empleo."""
    print("📊 Generando gráfico: Indicadores de Empleo")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Desempleo
    td = df.groupby('anio')['geih_td_nacional_media_anual'].mean()
    axes[0].plot(td.index, td.values, linewidth=2.5, color='#d62728', marker='o')
    axes[0].fill_between(td.index, td.values, alpha=0.3, color='#d62728')
    axes[0].set_title('Tasa de Desempleo (TD)', fontweight='bold')
    axes[0].set_ylabel('Porcentaje (%)')
    axes[0].grid(True, alpha=0.3)
    
    # Ocupación
    to = df.groupby('anio')['geih_to_nacional_media_anual'].mean()
    axes[1].plot(to.index, to.values, linewidth=2.5, color='#2ca02c', marker='s')
    axes[1].fill_between(to.index, to.values, alpha=0.3, color='#2ca02c')
    axes[1].set_title('Tasa de Ocupación (TO)', fontweight='bold')
    axes[1].set_ylabel('Porcentaje (%)')
    axes[1].grid(True, alpha=0.3)
    
    # Participación
    tgp = df.groupby('anio')['geih_tgp_nacional_media_anual'].mean()
    axes[2].plot(tgp.index, tgp.values, linewidth=2.5, color='#1f77b4', marker='^')
    axes[2].fill_between(tgp.index, tgp.values, alpha=0.3, color='#1f77b4')
    axes[2].set_title('Tasa Global de Participación (TGP)', fontweight='bold')
    axes[2].set_ylabel('Porcentaje (%)')
    axes[2].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xlabel('Año')
    
    plt.tight_layout()
    output_path = obtener_ruta_figura("empleo")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path.name}\n")
    plt.close()


def plot_educacion_variables(df):
    """Grafica variables educativas."""
    print("📊 Generando gráfico: Indicadores Educativos")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total matriculados
    matriculados = df.groupby('anio')['total_matriculados'].sum()
    axes[0, 0].plot(matriculados.index, matriculados.values / 1000, linewidth=2.5, color='#1f77b4')
    axes[0, 0].fill_between(matriculados.index, matriculados.values / 1000, alpha=0.3, color='#1f77b4')
    axes[0, 0].set_title('Total de Matriculados Nacionales', fontweight='bold')
    axes[0, 0].set_ylabel('Matriculados (miles)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Total admitidos
    admitidos = df.groupby('anio')['total_admitidos'].sum()
    axes[0, 1].plot(admitidos.index, admitidos.values / 1000, linewidth=2.5, color='#ff7f0e')
    axes[0, 1].fill_between(admitidos.index, admitidos.values / 1000, alpha=0.3, color='#ff7f0e')
    axes[0, 1].set_title('Total de Admitidos Nacionales', fontweight='bold')
    axes[0, 1].set_ylabel('Admitidos (miles)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ratio
    ratio = (matriculados / admitidos).fillna(0)
    axes[1, 0].plot(ratio.index, ratio.values, linewidth=2.5, color='#2ca02c', marker='s')
    axes[1, 0].fill_between(ratio.index, ratio.values, alpha=0.3, color='#2ca02c')
    axes[1, 0].set_title('Ratio: Matriculados / Admitidos', fontweight='bold')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Variación PIB por matriculado
    proxy = df.groupby('anio')['proxy_pib_miles_mm_cop_por_matriculado'].mean()
    axes[1, 1].plot(proxy.index, proxy.values, linewidth=2.5, color='#d62728', marker='^')
    axes[1, 1].fill_between(proxy.index, proxy.values, alpha=0.3, color='#d62728')
    axes[1, 1].set_title('PIB por Matriculado', fontweight='bold')
    axes[1, 1].set_ylabel('M COP por estudiante')
    axes[1, 1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xlabel('Año')
    
    plt.tight_layout()
    output_path = obtener_ruta_figura("educacion")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path.name}\n")
    plt.close()


def plot_correlaciones(df):
    """Grafica matriz de correlaciones."""
    print("📊 Generando gráfico: Matriz de Correlaciones")
    
    # Seleccionar variables clave
    cols = [
        'pib_total_miles_millones_cop',
        'total_matriculados',
        'outcome_tasa_desercion_snies',
        'ipc_nacional_total_var_mensual_media',
        'geih_td_nacional_media_anual',
        'geih_tgp_nacional_media_anual',
        'proxy_pib_miles_mm_cop_por_matriculado',
    ]
    
    # Simplificar nombres
    rename_map = {
        'pib_total_miles_millones_cop': 'PIB',
        'total_matriculados': 'Matriculados',
        'outcome_tasa_desercion_snies': 'Deserción',
        'ipc_nacional_total_var_mensual_media': 'IPC',
        'geih_td_nacional_media_anual': 'Desempleo',
        'geih_tgp_nacional_media_anual': 'Participación',
        'proxy_pib_miles_mm_cop_por_matriculado': 'PIB/Estudiante',
    }
    
    corr_matrix = df[cols].corr()
    corr_matrix = corr_matrix.rename(columns=rename_map, index=rename_map)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    
    ax.set_title('Matriz de Correlaciones - Variables Clave', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = obtener_ruta_figura("correlaciones")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path.name}\n")
    plt.close()


def plot_comparativa_1980_2026(df):
    """
    Compara el ranking de PIB departamental en 1980 vs 2026.
    
    Permite ver transformaciones económicas regionales en 46 años.
    """
    print("📊 Generando gráfico: Comparativa 1980 vs 2026")
    
    df_1980 = df[df['anio'] == 1980].sort_values('pib_total_miles_millones_cop', ascending=False).head(10)
    df_2026 = df[df['anio'] == 2026].sort_values('pib_total_miles_millones_cop', ascending=False).head(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # 1980
    axes[0].barh(df_1980['departamento'], df_1980['pib_total_miles_millones_cop'], color='#ff7f0e', alpha=0.7)
    axes[0].set_title('Top 10 PIB Departamental - 1980', fontweight='bold')
    axes[0].set_xlabel('PIB (miles de millones COP)')
    
    # 2026
    axes[1].barh(df_2026['departamento'], df_2026['pib_total_miles_millones_cop'], color='#1f77b4', alpha=0.7)
    axes[1].set_title('Top 10 PIB Departamental - 2026', fontweight='bold')
    axes[1].set_xlabel('PIB (miles de millones COP)')
    
    plt.tight_layout()
    output_path = obtener_ruta_figura("comparativa_1980_2026")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path.name}\n")
    plt.close()


def plot_distribucion_departamentos_2026(df):
    """Grafica distribuciones por departamento en 2026."""
    print("📊 Generando gráfico: Distribución Departamental 2026")
    
    df_2026 = df[df['anio'] == 2026]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PIB
    top_pib = df_2026.nlargest(15, 'pib_total_miles_millones_cop')
    axes[0, 0].barh(top_pib['departamento'], top_pib['pib_total_miles_millones_cop'], color='#1f77b4')
    axes[0, 0].set_title('Top 15 PIB Departamental', fontweight='bold')
    axes[0, 0].set_xlabel('PIB (M COP)')
    
    # Matriculados
    top_mat = df_2026.nlargest(15, 'total_matriculados')
    axes[0, 1].barh(top_mat['departamento'], top_mat['total_matriculados'], color='#2ca02c')
    axes[0, 1].set_title('Top 15 Matriculados', fontweight='bold')
    axes[0, 1].set_xlabel('Estudiantes')
    
    # Deserción baja
    low_deser = df_2026.nsmallest(10, 'outcome_tasa_desercion_snies')
    axes[1, 0].barh(low_deser['departamento'], low_deser['outcome_tasa_desercion_snies'], 
                   color=plt.cm.Greens(np.linspace(0.4, 0.8, 10)))
    axes[1, 0].set_title('10 Departamentos - Menor Deserción', fontweight='bold')
    axes[1, 0].set_xlabel('Tasa (%)')
    
    # Deserción alta
    high_deser = df_2026.nlargest(10, 'outcome_tasa_desercion_snies')
    axes[1, 1].barh(high_deser['departamento'], high_deser['outcome_tasa_desercion_snies'], 
                   color=plt.cm.Reds(np.linspace(0.4, 0.8, 10)))
    axes[1, 1].set_title('10 Departamentos - Mayor Deserción', fontweight='bold')
    axes[1, 1].set_xlabel('Tasa (%)')
    
    plt.tight_layout()
    output_path = obtener_ruta_figura("distribucion_2026")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardado: {output_path.name}\n")
    plt.close()


def main():
    """Función principal."""
    print("\n" + "="*80)
    print("VISUALIZACIÓN DEL DATASET SIMULADO")
    print("="*80 + "\n")
    
    # Cargar datos
    df = cargar_datos()
    if df is None:
        return
    
    # Generar gráficos
    plot_pib_nacional(df)
    plot_desercion_temporal(df)
    plot_empleo_variables(df)
    plot_educacion_variables(df)
    plot_correlaciones(df)
    plot_comparativa_1980_2026(df)
    plot_distribucion_departamentos_2026(df)
    
    # Resumen
    print("="*80)
    print("✅ VISUALIZACIONES GENERADAS")
    print("="*80)
    print("\n📁 Archivos creados en:")
    print(f"   {Path(__file__).parent / 'graficas'}")
    print("\n   • grafico_pib_nacional.png")
    print("   • grafico_desercion_temporal.png")
    print("   • grafico_empleo.png")
    print("   • grafico_educacion.png")
    print("   • grafico_correlaciones.png")
    print("   • grafico_comparativa_1980_2026.png")
    print("   • grafico_distribucion_2026.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
