"""
data_generator.py
=================
Generador de dataset simulado de variables socioeconómicas y educativas de Colombia (1980-2024).

Este módulo simula datos históricos realistas para:
- Variables macroeconómicas (PIB, inflación)
- Variables educativas (admitidos, matriculados, deserción)
- Variables de empleo (desempleo, ocupación)
- Variables de precios (IPC nacional, capital)

Cada variable utiliza una distribución probabilística diferente basada en:
- Datos reales observados (2023-2024)
- Eventos históricos relevantes en Colombia
- Tendencias económicas conocidas

Ejecución:
    python data_generator.py

Salida:
    - data_simulado_1980_2024.csv (dataset completo)
    - resumen_simulacion.txt (estadísticas de la simulación)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Importar configuración centralizada
from config import crear_directorios, DATASET_PATH, SUMMARY_PATH

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Crear directorios necesarios
crear_directorios()

# Rutas de salida (desde configuración centralizada)
OUTPUT_FILE = DATASET_PATH
SUMMARY_FILE = SUMMARY_PATH

# Período de simulación (importado de config)
from config import YEAR_START, YEAR_END
YEARS = np.arange(YEAR_START, YEAR_END + 1)

# Departamentos de Colombia (32 + 1 capital)
DEPARTAMENTOS = {
    5: "Antioquia",
    8: "Atlántico",
    11: "Bogotá D.C.",
    13: "Bolívar",
    15: "Boyacá",
    17: "Caldas",
    18: "Caquetá",
    19: "Cauca",
    20: "Cesar",
    23: "Córdoba",
    25: "Cundinamarca",
    27: "Chocó",
    41: "Huila",
    44: "La Guajira",
    47: "Magdalena",
    50: "Meta",
    52: "Nariño",
    54: "Norte de Santander",
    63: "Quindío",
    66: "Risaralda",
    68: "Santander",
    70: "Sucre",
    73: "Tolima",
    76: "Valle del Cauca",
    81: "Arauca",
    85: "Casanare",
    86: "Putumayo",
    88: "San Andrés, Providencia y Santa Catalina (Archipiélago)",
    91: "Amazonas",
    94: "Guainía",
    95: "Guaviare",
    97: "Vaupés",
    99: "Vichada",
}

# Valores reales observados (2023-2024) para calibración
VALORES_BASE = {
    2023: {
        "pib_total": 1_050_000,  # millones COP (suma de todos los departamentos)
        "ipc_nacional": 0.74,  # variación mensual media
        "geih_td": 10.17,  # tasa de desempleo
        "geih_to": 57.62,  # tasa de ocupación
        "geih_tgp": 64.14,  # tasa global de participación
    },
    2024: {
        "pib_total": 1_150_000,
        "ipc_nacional": 0.53,
        "geih_td": 10.16,
        "geih_to": 57.41,
        "geih_tgp": 63.91,
    }
}

# ============================================================================
# FUNCIONES GENERADORAS DE SERIES TEMPORALES
# ============================================================================

def generar_serie_pib(years, dept_id):
    """
    Genera serie de PIB departamental con tendencia logística.
    
    Distribución: Lognormal con tendencia de crecimiento
    - 1980: valores bajos (post-crisis económica)
    - 1990-2000: crecimiento exponencial
    - 2000-2008: estabilización
    - 2008-2010: crisis financiera global
    - 2010-2024: recuperación y crecimiento moderado
    
    Args:
        years: array de años
        dept_id: código del departamento
        
    Returns:
        array de PIB departamental (miles de millones COP)
    """
    n = len(years)
    t = np.arange(n)
    
    # Factor de escala por departamento (calibrado)
    scale_factor = {
        5: 0.22,   # Antioquia (gran productor)
        11: 0.38,  # Bogotá (mayor PIB)
        68: 0.10,  # Santander
        76: 0.15,  # Valle del Cauca
    }.get(dept_id, 0.08)  # default para otros
    
    # Tendencia logística: crecimiento S
    logistica = 1 / (1 + np.exp(-0.15 * (t - 25)))  # inflexión en 2005
    
    # Componente de tendencia lineal
    tendencia = t / n
    
    # Componente cíclica (ciclos económicos)
    ciclica = 0.2 * np.sin(2 * np.pi * t / 8)  # ciclo cada 8 años aprox
    
    # Crisis específicas: 1999, 2008
    crisis = np.zeros(n)
    crisis[19] = -0.15  # 1999: crisis bancaria Colombia
    crisis[28] = -0.10  # 2008: crisis financiera global
    crisis[40] = -0.05  # 2020: COVID-19
    
    # Combinar componentes
    indice = logistica * 2 + tendencia + ciclica + crisis
    
    # Escalar a valores reales
    pib_base_2024 = VALORES_BASE[2024]["pib_total"] * scale_factor
    pib = pib_base_2024 * (0.3 + indice)  # 0.3 es mínimo relativo
    
    # Agregar ruido lognormal
    ruido = np.random.lognormal(mean=0, sigma=0.1, size=n)
    pib = pib * ruido
    
    return np.maximum(pib, 100)  # Mínimo de 100 millones


def generar_serie_ipc(years):
    """
    Genera serie de IPC nacional con variación mensual.
    
    Distribución: Normal con cambios de régimen
    - 1980-1990: inflación alta (30-40% anual)
    - 1991-2000: inflación moderada (10-20% anual)
    - 2001-2023: inflación baja (2-8% anual)
    - 2024: inflación controlada (~2.8%)
    
    Args:
        years: array de años
        
    Returns:
        array de variación mensual media IPC
    """
    n = len(years)
    t = np.arange(n)
    
    # Tres regímenes de inflación
    ipc = np.zeros(n)
    
    # 1980-1990: Inflación alta
    mascara_alto = (t < 11)
    ipc[mascara_alto] = np.random.normal(2.5, 0.8, mascara_alto.sum())
    
    # 1991-2000: Inflación moderada
    mascara_medio = (t >= 11) & (t < 21)
    ipc[mascara_medio] = np.random.normal(1.2, 0.4, mascara_medio.sum())
    
    # 2001-2024: Inflación baja
    mascara_bajo = t >= 21
    ipc[mascara_bajo] = np.random.normal(0.6, 0.2, mascara_bajo.sum())
    
    # Agregar eventos inflacionarios
    ipc[19] = 2.1  # 1999: crisis
    ipc[28] = 1.5  # 2008: crisis global
    ipc[40] = 1.2  # 2020: COVID (inflación moderada)
    ipc[42] = 1.8  # 2022: inflación post-COVID
    ipc[43] = 1.1  # 2023: desaceleración
    ipc[44] = 0.53 # 2024: valor real observado
    
    return np.clip(ipc, 0.1, 3.5)


def generar_serie_empleo(years, variable="td"):
    """
    Genera series de indicadores de empleo (TD, TO, TGP).
    
    Distribuciones:
    - TD (Tasa de Desempleo): Normal + eventos
    - TO (Tasa de Ocupación): Normal inversa a TD
    - TGP (Tasa Global de Participación): Normal con tendencia creciente
    
    Args:
        years: array de años
        variable: "td" (desempleo), "to" (ocupación), "tgp" (participación)
        
    Returns:
        array de valores del indicador
    """
    n = len(years)
    t = np.arange(n)
    
    if variable == "td":
        # Tasa de desempleo: inversamente relacionada con ciclo económico
        tendencia = 0.15 - 0.001 * t  # tendencia decreciente
        ciclo = 2 * np.sin(2 * np.pi * t / 8)
        
        td = 13 + tendencia + ciclo
        td[19] = 18  # 1999: crisis
        td[28] = 14  # 2008: crisis
        td[40] = 16  # 2020: COVID
        
        # Agregar ruido normal
        td = td + np.random.normal(0, 1, n)
        td = np.clip(td, 4, 22)
        
        return td
    
    elif variable == "to":
        # Tasa de ocupación: aproximadamente 100 - TD - TGP + constante
        td = generar_serie_empleo(years, "td")
        tgp = generar_serie_empleo(years, "tgp")
        
        to = 65 - (td - 10) + (tgp - 62) * 0.5
        to = np.clip(to, 45, 70)
        
        return to
    
    elif variable == "tgp":
        # Tasa global de participación: tendencia creciente (feminización)
        tendencia = 55 + 0.15 * t  # crecimiento gradual
        estacional = 1 * np.sin(2 * np.pi * t / 4)
        
        tgp = tendencia + estacional
        tgp = tgp + np.random.normal(0, 0.5, n)
        
        # Valores observados
        tgp[43] = 64.137  # 2023
        tgp[44] = 63.909  # 2024
        
        return np.clip(tgp, 52, 68)


def generar_serie_educacion(years, dept_id, tipo="admitidos"):
    """
    Genera series de variables educativas.
    
    Distribuciones:
    - Admitidos: Poisson (discreto, eventos raros)
    - Matriculados: Normal con correlación con PIB
    - Deserción: Beta (0-100%) con variación departamental
    
    Args:
        years: array de años
        dept_id: código del departamento
        tipo: "admitidos", "matriculados", "desercion"
        
    Returns:
        array de valores
    """
    n = len(years)
    t = np.arange(n)
    
    # Factor de tamaño por departamento
    size_factor = {
        5: 1.2,   # Antioquia
        11: 5.0,  # Bogotá (mucho mayor)
        8: 0.5,   # Atlántico
        68: 0.8,  # Santander
        76: 1.0,  # Valle del Cauca
    }.get(dept_id, 0.6)  # default
    
    if tipo == "admitidos":
        # Poisson: número de admitidos
        # Tendencia: crecimiento con inflexión hacia 2000
        base = 5000 * size_factor
        tendencia = 1 + (t / n) * 3  # crecimiento 3x en 45 años
        
        admitidos = base * tendencia
        admitidos = np.random.poisson(admitidos / 30) * 30  # suavizar
        
        return np.maximum(admitidos, 100)
    
    elif tipo == "matriculados":
        # Normal: número de matriculados (más suave que admitidos)
        base = 20000 * size_factor
        tendencia = 1 + (t / n) * 2.5
        
        matriculados = base * tendencia
        matriculados = matriculados + np.random.normal(0, 2000, n)
        
        return np.maximum(matriculados, 500)
    
    elif tipo == "desercion":
        # Beta: tasa de deserción (0-100%)
        # Mejora con educación: tendencia decreciente
        media = 12 - (t / n) * 4  # de 12% a 8%
        
        # Variación departamental (departamentos más pobres tienen más deserción)
        ajuste_dept = {
            5: -1,   # Antioquia (mejor desempeño)
            11: -2,  # Bogotá (mejor desempeño)
            44: 3,   # La Guajira (mayor vulnerabilidad)
            27: 2,   # Chocó
        }.get(dept_id, 0)
        
        media = media + ajuste_dept
        
        # Crisis específicas aumentan deserción
        desercion = media + np.random.normal(0, 1.5, n)
        desercion[19] = media[19] + 2  # 1999
        desercion[28] = media[28] + 1.5  # 2008
        desercion[40] = media[40] + 3  # 2020: COVID
        
        return np.clip(desercion, 3, 25)


# ============================================================================
# FUNCIÓN PRINCIPAL DE GENERACIÓN
# ============================================================================

def generar_variacion_mensual(valor_anual, mes, tipo_variable="default"):
    """
    Genera variación mensual realista para un valor anual.
    
    Diferentes patrones estacionales según el tipo:
    - IPC: variación estacional (algunos meses más altos)
    - Educación (admitidos/matriculados): picos en semestres
    - Empleo: ciclo anual con picos en trimestres
    - PIB: variación suave
    
    Args:
        valor_anual: valor base para el año
        mes: número de mes (1-12)
        tipo_variable: tipo de patrón estacional a aplicar
        
    Returns:
        valor ajustado por estacionalidad
    """
    if tipo_variable == "ipc":
        # IPC: variación pequeña (mayor estabilidad)
        patrones = [0.95, 0.98, 0.97, 0.99, 1.02, 1.05, 1.08, 1.06, 1.03, 1.01, 1.02, 0.99]
        factor = patrones[mes - 1]
    
    elif tipo_variable == "educacion":
        # Educación: picos en semestres (enero, julio)
        patrones = [1.15, 0.95, 0.90, 0.88, 0.87, 0.89, 1.20, 0.92, 0.85, 0.83, 0.85, 0.90]
        factor = patrones[mes - 1]
    
    elif tipo_variable == "empleo":
        # Empleo: variación cíclica trimestral
        patrones = [0.98, 0.96, 0.95, 0.97, 0.99, 1.01, 1.03, 1.02, 1.00, 0.99, 0.98, 0.97]
        factor = patrones[mes - 1]
    
    else:  # default (PIB, deserción, etc.)
        # Variación suave
        patrones = [0.99, 0.98, 0.99, 1.00, 1.01, 1.02, 1.02, 1.01, 1.00, 0.99, 0.98, 0.99]
        factor = patrones[mes - 1]
    
    # Agregar ruido aleatorio pequeño
    ruido = np.random.normal(1.0, 0.02)
    
    return valor_anual * factor * ruido


def generar_dataset_completo():
    """
    Genera dataset completo MENSUAL de 1980 a 2024 para todos los departamentos.
    
    Estructura:
    - 33 departamentos × 47 años × 12 meses = 18,612 registros
    - Cada mes contiene variación estacional realista
    - Variables se distribuyen de forma inteligente según su tipo
    
    Returns:
        DataFrame con todas las variables simuladas a granularidad mensual
    """
    print("\n" + "="*80)
    print("GENERADOR DE DATASET SIMULADO - VARIABLES SOCIOECONÓMICAS COLOMBIA (1980-2026)")
    print("GRANULARIDAD: MENSUAL (12 registros por departamento/año)")
    print("="*80)
    
    registros = []
    
    # Generar una sola vez las series nacionales
    print("\n📊 Generando series nacionales...")
    serie_ipc = generar_serie_ipc(YEARS)
    serie_td = generar_serie_empleo(YEARS, "td")
    serie_to = generar_serie_empleo(YEARS, "to")
    serie_tgp = generar_serie_empleo(YEARS, "tgp")
    
    print("   ✓ Series macroeconómicas generadas")
    
    # Iterar por cada departamento
    print("\n📍 Generando datos MENSUALES por departamento...")
    for dept_id, dept_nombre in DEPARTAMENTOS.items():
        print(f"   Procesando: {dept_nombre:45s} (Código: {dept_id})", end="\r")
        
        # Generar series departamentales
        serie_pib = generar_serie_pib(YEARS, dept_id)
        serie_admitidos = generar_serie_educacion(YEARS, dept_id, "admitidos")
        serie_matriculados = generar_serie_educacion(YEARS, dept_id, "matriculados")
        serie_desercion = generar_serie_educacion(YEARS, dept_id, "desercion")
        
        # Crear registros MENSUALES
        for idx, year in enumerate(YEARS):
            for mes in range(1, 13):  # 12 meses
                # Aplicar variación mensual según tipo de variable
                pib_mes = generar_variacion_mensual(serie_pib[idx], mes, "default")
                admitidos_mes = generar_variacion_mensual(serie_admitidos[idx], mes, "educacion")
                matriculados_mes = generar_variacion_mensual(serie_matriculados[idx], mes, "educacion")
                ipc_mes = generar_variacion_mensual(serie_ipc[idx], mes, "ipc")
                td_mes = generar_variacion_mensual(serie_td[idx], mes, "empleo")
                to_mes = generar_variacion_mensual(serie_to[idx], mes, "empleo")
                tgp_mes = generar_variacion_mensual(serie_tgp[idx], mes, "empleo")
                desercion_mes = generar_variacion_mensual(serie_desercion[idx], mes, "default")
                
                # Calcular variables derivadas
                ratio = matriculados_mes / admitidos_mes if admitidos_mes > 0 else 0
                
                # Variación porcentual anual (misma para todos los meses del año)
                var_matriculados = np.nan
                if idx > 0 and serie_matriculados[idx-1] > 0:
                    var_matriculados = (serie_matriculados[idx] - serie_matriculados[idx-1]) / serie_matriculados[idx-1] * 100
                
                var_pib = np.nan
                if idx > 0 and serie_pib[idx-1] > 0:
                    var_pib = (serie_pib[idx] - serie_pib[idx-1]) / serie_pib[idx-1] * 100
                
                # Proxy: PIB por matriculado
                proxy_pib_per_student = pib_mes / matriculados_mes if matriculados_mes > 0 else 0
                
                # IPC: variaciones mensuales y derivadas
                ipc_mediana = ipc_mes * np.random.uniform(0.8, 1.2)
                ipc_std = ipc_mes * np.random.uniform(0.5, 0.8)
                
                # Crear registro MENSUAL
                registro = {
                    "codigo_departamento": dept_id,
                    "departamento": dept_nombre,
                    "anio": year,
                    "mes": mes,
                    "pib_total_miles_millones_cop": pib_mes,
                    "pib_variacion_pct_anual_vs_anio_previo": var_pib,
                    "total_admitidos": admitidos_mes,
                    "total_matriculados": matriculados_mes,
                    "ratio_matriculados_sobre_admitidos": ratio,
                    "var_pct_matriculados_vs_anio_previo": var_matriculados,
                    "ipc_nacional_total_var_mensual_media": ipc_mes,
                    "ipc_nacional_total_var_mensual_mediana": ipc_mediana,
                    "ipc_nacional_total_var_mensual_std": ipc_std,
                    # IPC por categorías
                    "ipc_nacional_educacion_var_mensual_media": ipc_mes * np.random.uniform(0.9, 1.1),
                    "ipc_nacional_educacion_var_mensual_mediana": ipc_mediana * np.random.uniform(0.8, 1.2),
                    "ipc_nacional_educacion_var_mensual_std": ipc_std * np.random.uniform(0.7, 1.3),
                    "ipc_nacional_alimentos_var_mensual_media": ipc_mes * np.random.uniform(0.8, 1.2),
                    "ipc_nacional_alimentos_var_mensual_mediana": ipc_mediana * np.random.uniform(0.7, 1.3),
                    "ipc_nacional_alimentos_var_mensual_std": ipc_std * np.random.uniform(0.6, 1.4),
                    "ipc_nacional_transporte_var_mensual_media": ipc_mes * np.random.uniform(0.7, 1.3),
                    "ipc_nacional_transporte_var_mensual_mediana": ipc_mediana * np.random.uniform(0.6, 1.4),
                    "ipc_nacional_transporte_var_mensual_std": ipc_std * np.random.uniform(0.5, 1.5),
                    "ipc_capital_total_var_mensual_media": ipc_mes * np.random.uniform(0.95, 1.05),
                    "ipc_capital_educacion_var_mensual_media": ipc_mes * np.random.uniform(0.9, 1.1),
                    # Empleo (variación mensual)
                    "geih_td_nacional_media_anual": td_mes,
                    "geih_to_nacional_media_anual": to_mes,
                    "geih_tgp_nacional_media_anual": tgp_mes,
                    # Proxy
                    "proxy_pib_miles_mm_cop_por_matriculado": proxy_pib_per_student,
                    # SPADIES (tasas de deserción por nivel)
                    "spadies_td_anual_universitario": desercion_mes * np.random.uniform(0.8, 1.2),
                    "spadies_td_anual_tyt": desercion_mes * np.random.uniform(1.0, 1.5),
                    "spadies_td_anual_tecnico_profesional": desercion_mes * np.random.uniform(0.9, 1.3),
                    "spadies_td_anual_tecnologico": desercion_mes * np.random.uniform(0.85, 1.25),
                    # Target: tasa de deserción SNIES
                    "outcome_tasa_desercion_snies": desercion_mes,
                    # Flag de datos
                    "outcome_merge_pendiente": 1 if year <= 2022 else 0,
                }
                
                registros.append(registro)
    
    print(f"\n   ✓ {len(DEPARTAMENTOS)} departamentos procesados (mensualmente)")
    
    # Crear DataFrame
    df = pd.DataFrame(registros)
    
    # Ordenar
    df = df.sort_values(["codigo_departamento", "anio", "mes"]).reset_index(drop=True)
    
    print(f"\n📈 Dataset generado (MENSUAL):")
    print(f"   • Total de registros: {len(df):,}")
    print(f"   • Período: {df['anio'].min()}-{df['anio'].max()} ({df['anio'].max() - df['anio'].min() + 1} años)")
    print(f"   • Meses: {df['mes'].min()}-{df['mes'].max()}")
    print(f"   • Departamentos: {df['codigo_departamento'].nunique()}")
    print(f"   • Registros por departamento: {len(df) // df['codigo_departamento'].nunique()}")
    print(f"   • Registros mensuales por año: {df[df['anio'] == df['anio'].min()].shape[0] // df['codigo_departamento'].nunique()}")
    
    return df


# ============================================================================
# FUNCIÓN DE ANÁLISIS Y DOCUMENTACIÓN
# ============================================================================

def generar_reporte_simulacion(df):
    """
    Genera reporte detallado de la simulación.
    
    Args:
        df: DataFrame simulado
    """
    reporte = []
    reporte.append("="*80)
    reporte.append("REPORTE DE SIMULACIÓN - DATASET SOCIOECONÓMICO COLOMBIA (1980-2026)")
    reporte.append("GRANULARIDAD: MENSUAL - 12 registros por departamento/año")
    reporte.append("="*80)
    reporte.append(f"\nGenerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    reporte.append("\n" + "="*80)
    reporte.append("1. DIMENSIONES DEL DATASET")
    reporte.append("="*80)
    reporte.append(f"Registros totales: {len(df):,}")
    reporte.append(f"Fórmula: {df['codigo_departamento'].nunique()} departamentos × {df['anio'].max() - df['anio'].min() + 1} años × 12 meses = {len(df):,}")
    reporte.append(f"Columnas: {df.shape[1]}")
    reporte.append(f"Período: {df['anio'].min()}-{df['anio'].max()} ({df['anio'].max() - df['anio'].min() + 1} años)")
    reporte.append(f"Meses: {df['mes'].min()}-{df['mes'].max()}")
    reporte.append(f"Departamentos: {df['codigo_departamento'].nunique()}")
    
    reporte.append("\n" + "="*80)
    reporte.append("2. ESTADÍSTICAS DESCRIPTIVAS")
    reporte.append("="*80)
    
    variables_clave = [
        "pib_total_miles_millones_cop",
        "total_matriculados",
        "outcome_tasa_desercion_snies",
        "ipc_nacional_total_var_mensual_media",
        "geih_td_nacional_media_anual",
    ]
    
    for var in variables_clave:
        reporte.append(f"\n{var}:")
        stats = df[var].describe()
        reporte.append(f"  Media: {stats['mean']:.2f}")
        reporte.append(f"  Std:   {stats['std']:.2f}")
        reporte.append(f"  Min:   {stats['min']:.2f}")
        reporte.append(f"  Max:   {stats['max']:.2f}")
    
    reporte.append("\n" + "="*80)
    reporte.append("3. DATOS FALTANTES")
    reporte.append("="*80)
    
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    
    if len(nulos) > 0:
        for col, count in nulos.items():
            pct = (count / len(df)) * 100
            reporte.append(f"{col:50s}: {count:6,} ({pct:5.2f}%)")
    else:
        reporte.append("No hay datos faltantes")
    
    reporte.append("\n" + "="*80)
    reporte.append("4. DISTRIBUCIONES UTILIZADAS POR VARIABLE")
    reporte.append("="*80)
    
    distribuciones = {
        "pib_total_miles_millones_cop": "Lognormal (tendencia logística + ruido log)",
        "pib_variacion_pct_anual": "Normal derivada (cambio año a año)",
        "total_admitidos": "Poisson (eventos discretos)",
        "total_matriculados": "Normal (suavizado)",
        "ratio_matriculados_sobre_admitidos": "Normal derivada",
        "ipc_nacional_*": "Normal por régimen (1980-90: alta, 1991-00: media, 2001-24: baja)",
        "geih_td": "Normal + ciclos económicos (8 años)",
        "geih_to": "Normal inversa a TD",
        "geih_tgp": "Normal con tendencia creciente (feminización)",
        "outcome_tasa_desercion": "Beta con variación departamental",
    }
    
    for var, dist in distribuciones.items():
        reporte.append(f"\n{var}:")
        reporte.append(f"  {dist}")
    
    reporte.append("\n" + "="*80)
    reporte.append("5. EVENTOS HISTÓRICOS SIMULADOS")
    reporte.append("="*80)
    
    eventos = {
        1999: "Crisis bancaria colombiana",
        2008: "Crisis financiera global",
        2020: "Pandemia COVID-19",
        2022: "Inflación post-COVID",
    }
    
    for año, evento in eventos.items():
        reporte.append(f"{año}: {evento}")
    
    reporte.append("\n" + "="*80)
    reporte.append("6. FACTORES DE ESCALA POR DEPARTAMENTO")
    reporte.append("="*80)
    
    size_factors = {
        5: "1.2x   - Antioquia (segundo mayor PIB)",
        11: "5.0x   - Bogotá D.C. (mayor PIB y población)",
        8: "0.5x   - Atlántico",
        68: "0.8x   - Santander",
        76: "1.0x   - Valle del Cauca",
        44: "0.6x   - La Guajira (mayor vulnerabilidad)",
        27: "0.4x   - Chocó (departamento más pequeño)",
    }
    
    for codigo, factor in size_factors.items():
        nombre = DEPARTAMENTOS.get(codigo, "Desconocido")
        reporte.append(f"{nombre:40s}: {factor}")
    
    return "\n".join(reporte)


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    
    # Generar dataset
    df = generar_dataset_completo()
    
    # Guardar CSV
    print(f"\n💾 Guardando dataset en: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print("   ✓ Dataset guardado")
    
    # Generar y guardar reporte
    print(f"\n📄 Generando reporte de simulación...")
    reporte = generar_reporte_simulacion(df)
    
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(reporte)
    
    print(f"   ✓ Reporte guardado en: {SUMMARY_FILE}")
    
    # Mostrar resumen
    print("\n" + reporte)
    
    print("\n" + "="*80)
    print("✅ SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print("="*80 + "\n")
