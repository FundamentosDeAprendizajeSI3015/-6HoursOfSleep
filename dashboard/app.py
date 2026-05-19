import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPERVISED_CSV = PROJECT_ROOT / "supervised" / "reports" / "eda_figures" / "tabla_comparativa_modelos_supervisados.csv"
SUPERVISED_JSON = PROJECT_ROOT / "supervised" / "results" / "pipeline_results.json"
UNSUPERVISED_JSON = PROJECT_ROOT / "unsupervised" / "docs" / "unsupervised_results.json"
INDEX_SIM_JSON = PROJECT_ROOT / "indexes-scores" / "reports" / "dataset_simulado" / "metricas_rendimiento.json"
INDEX_ORIG_JSON = PROJECT_ROOT / "indexes-scores" / "reports" / "dataset_original" / "metricas_rendimiento.json"

with open(SUPERVISED_JSON, encoding="utf-8") as f:
    sup_results = json.load(f)
with open(UNSUPERVISED_JSON, encoding="utf-8") as f:
    unsup_results = json.load(f)
with open(INDEX_SIM_JSON, encoding="utf-8") as f:
    idx_sim = json.load(f)
with open(INDEX_ORIG_JSON, encoding="utf-8") as f:
    idx_orig = json.load(f)

df_test = pd.read_csv(SUPERVISED_CSV, usecols=[0, 1, 2, 3, 4])
df_test.columns = ["Modelo", "RMSE", "MAE", "R2", "MAPE"]
df_test["RMSE"] = pd.to_numeric(df_test["RMSE"], errors="coerce")
df_test["MAE"] = pd.to_numeric(df_test["MAE"], errors="coerce")
df_test["R2"] = pd.to_numeric(df_test["R2"], errors="coerce")
df_test["MAPE"] = pd.to_numeric(df_test["MAPE"], errors="coerce")

df_cv_list = []
for m in sup_results["entrenamiento_modelos"]["resultados_cv"]:
    parts = m["rmse"].split(" ± ")
    rmse_mean = float(parts[0])
    rmse_std = float(parts[1]) if len(parts) > 1 else 0
    parts_mae = m["mae"].split(" ± ")
    mae_mean = float(parts_mae[0])
    mae_std = float(parts_mae[1]) if len(parts_mae) > 1 else 0
    parts_r2 = m["r2"].split(" ± ")
    r2_mean = float(parts_r2[0])
    r2_std = float(parts_r2[1]) if len(parts_r2) > 1 else 0
    short_name = m["modelo"].split(" (")[0]
    df_cv_list.append({
        "Modelo": short_name,
        "RMSE_mean": rmse_mean, "RMSE_std": rmse_std,
        "MAE_mean": mae_mean, "MAE_std": mae_std,
        "R2_mean": r2_mean, "R2_std": r2_std,
    })
df_cv = pd.DataFrame(df_cv_list)

winner = sup_results["modelo_ganador"]
improvement = winner["mejora_vs_baseline"]["mejora_porcentaje"]

COLORS = px.colors.qualitative.Plotly
MODEL_COLORS = {m: COLORS[i % len(COLORS)] for i, m in enumerate(df_test["Modelo"])}

app = Dash(__name__, title="Dashboard — Modelos de Deserción Estudiantil")


def card_style(color):
    return {
        "background": color,
        "borderRadius": 10, "padding": "15px 20px", "flex": "1 1 180px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.15)", "minWidth": 160,
    }


app.layout = html.Div([
    html.Div([
        html.H1("Dashboard de Modelos de Deserción Estudiantil",
                style={"textAlign": "center", "color": "#1a1a2e", "marginBottom": 0}),
        html.P("Análisis comparativo de modelos supervisados, clustering e índices de vulnerabilidad — Colombia",
               style={"textAlign": "center", "color": "#6c757d", "fontSize": 16, "marginTop": 5}),
    ], style={"padding": "20px 10px 10px 10px"}),

    html.Div([
        html.Div([
            html.H4("Mejor Modelo", style={"margin": 0, "color": "#fff"}),
            html.H2("RandomForest", style={"margin": "5px 0", "color": "#fff"}),
            html.P(f"R² = {winner['metricas_test']['r2']} | RMSE = {winner['metricas_test']['rmse']}",
                   style={"margin": 0, "color": "#e0e0e0"}),
        ], className="card", style=card_style("#28a745")),

        html.Div([
            html.H4("Mejora vs Baseline", style={"margin": 0, "color": "#fff"}),
            html.H2(improvement, style={"margin": "5px 0", "color": "#fff"}),
            html.P("vs OLS (Regresión Lineal)", style={"margin": 0, "color": "#e0e0e0"}),
        ], className="card", style=card_style("#007bff")),

        html.Div([
            html.H4("Error Absoluto (MAPE)", style={"margin": 0, "color": "#fff"}),
            html.H2(f"{winner['metricas_test']['mape']}", style={"margin": "5px 0", "color": "#fff"}),
            html.P("Error porcentual medio en test", style={"margin": 0, "color": "#e0e0e0"}),
        ], className="card", style=card_style("#dc3545")),

        html.Div([
            html.H4("Modelos Evaluados", style={"margin": 0, "color": "#fff"}),
            html.H2("8", style={"margin": "5px 0", "color": "#fff"}),
            html.P(f"{sup_results['datos_cargados']['registros_totales']:,} registros mensuales",
                   style={"margin": 0, "color": "#e0e0e0"}),
        ], className="card", style=card_style("#6f42c1")),
    ], style={"display": "flex", "gap": 15, "padding": "15px 10px", "flexWrap": "wrap"}),

    dcc.Tabs([
        dcc.Tab(label=" Modelos Supervisados ", children=[
            html.Div([
                html.H3("Comparación de Modelos — Test Set", style={"marginTop": 0}),
                html.P("Métricas de los 8 modelos supervisados evaluados en el conjunto de prueba (20% de datos). "
                        "RandomForest es el modelo ganador con R²=0.9284."),

                html.Div([
                    html.Div([dcc.Graph(id="rmse-chart")], style={"width": "50%"}),
                    html.Div([dcc.Graph(id="r2-chart")], style={"width": "50%"}),
                ], style={"display": "flex"}),

                html.Div([
                    html.Div([dcc.Graph(id="mae-chart")], style={"width": "50%"}),
                    html.Div([dcc.Graph(id="mape-chart")], style={"width": "50%"}),
                ], style={"display": "flex"}),

                html.Div([
                    html.H4("Tabla Comparativa — Test Set"),
                    dcc.Graph(id="test-table"),
                ]),
            ], style={"padding": 15}),
        ]),

        dcc.Tab(label=" Validación Cruzada ", children=[
            html.Div([
                html.H3("Validación Cruzada (5-Fold)", style={"marginTop": 0}),
                html.P("Resultados de validación cruzada con 5 folds. Las barras de error representan ±1 desviación estándar."),

                html.Div([
                    html.Div([dcc.Graph(id="cv-rmse")], style={"width": "50%"}),
                    html.Div([dcc.Graph(id="cv-r2")], style={"width": "50%"}),
                ], style={"display": "flex"}),

                html.Div([
                    html.H4("Resumen de Validación Cruzada"),
                    dcc.Graph(id="cv-table"),
                ]),
            ], style={"padding": 15}),
        ]),

        dcc.Tab(label=" Importancia de Variables ", children=[
            html.Div([
                html.H3("Importancia de Variables — Modelos de Árbol", style={"marginTop": 0}),
                html.P("Las variables SPADIES (tasas de deserción por tipo de educación) dominan la importancia, "
                        "seguidas por indicadores económicos como PIB e IPC."),

                html.Div([
                    html.H4("Top 5 Variables Más Correlacionadas con la Deserción"),
                    dcc.Graph(id="top-features"),
                ]),

                html.Div([
                    html.H4("Notas sobre Feature Importance"),
                    html.Ul([
                        html.Li("RandomForest y GradientBoosting son los únicos modelos que proveen importancia de variables intrínsecamente."),
                        html.Li("Las variables SPADIES (spadies_td_anual_*) capturan tasas de deserción por nivel educativo "
                                "(técnico profesional, tecnológico, TyT), lo que explica su alta correlación."),
                        html.Li("El PIB total nacional aparece como el predictor económico más relevante."),
                        html.Li("Las variables de IPC (inflación) también muestran correlación moderada con la deserción."),
                    ]),
                ], style={"marginTop": 20}),
            ], style={"padding": 15}),
        ]),

        dcc.Tab(label=" Clustering (No Supervisado) ", children=[
            html.Div([
                html.H3("Resultados de Clustering", style={"marginTop": 0}),
                html.P("Evaluación de algoritmos de clustering no supervisado sobre los datos socioeconómicos."),

                html.Div([
                    html.Div([dcc.Graph(id="kmeans-silhouette")], style={"width": "50%"}),
                    html.Div([dcc.Graph(id="clustering-metrics")], style={"width": "50%"}),
                ], style={"display": "flex"}),

                html.Div([
                    html.Div([dcc.Graph(id="dbscan-info")], style={"width": "50%"}),
                    html.Div([dcc.Graph(id="validation-ari")], style={"width": "50%"}),
                ], style={"display": "flex"}),

                html.Div([
                    html.H4("Interpretación"),
                    html.Ul([
                        html.Li("K-Means con k=3 obtiene el mejor silhouette score (0.431), identificando 3 grupos en los datos."),
                        html.Li("DBSCAN no logra encontrar clusters válidos en ninguna configuración (todo clasificado como ruido)."),
                        html.Li("El clustering jerárquico (Agglomerative) muestra menor cohesión (silhouette=0.255)."),
                        html.Li("El Adjusted Rand Index (ARI) es muy bajo (~0.06-0.08), indicando que los clusters no se alinean "
                                "significativamente con la tasa de deserción real."),
                    ]),
                ], style={"marginTop": 20}),
            ], style={"padding": 15}),
        ]),

        dcc.Tab(label=" Índices de Vulnerabilidad ", children=[
            html.Div([
                html.H3("Índices de Vulnerabilidad — PCA vs Teórico", style={"marginTop": 0}),
                html.P("Comparación del Índice de Vulnerabilidad basado en PCA y el Índice Teórico "
                        "(con pesos basados en literatura) para ambos datasets."),

                html.Div([
                    html.Div([dcc.Graph(id="index-sim")], style={"width": "50%"}),
                    html.Div([dcc.Graph(id="index-orig")], style={"width": "50%"}),
                ], style={"display": "flex"}),

                html.Div([
                    html.H4("Comparación de Datasets"),
                    dcc.Graph(id="index-comparison-table"),
                ]),
            ], style={"padding": 15}),
        ]),

        dcc.Tab(label=" Resumen Ejecutivo ", children=[
            html.Div([
                html.H3("Resumen Ejecutivo", style={"marginTop": 0}),

                html.Div([
                    html.H4("Hallazgos Clave"),
                    html.Ul([
                        html.Li(f"RandomForest es el mejor modelo con R² de {winner['metricas_test']['r2']} en test set."),
                        html.Li(f"Mejora del {improvement} respecto al baseline OLS."),
                        html.Li(f"Error porcentual (MAPE) de solo {winner['metricas_test']['mape']}."),
                        html.Li("Las tasas de deserción SPADIES por nivel educativo son los predictores más fuertes."),
                        html.Li("El índice PCA de vulnerabilidad correlaciona significativamente con la deserción "
                                f"(Spearman ρ={idx_sim['INDEX_PCA_spearman_rho']}, p<0.001)."),
                        html.Li("El clustering no logra separar grupos que se alineen con la deserción (ARI ~ 0.06)."),
                    ]),

                    html.H4("Próximos Pasos Recomendados"),
                    html.Ol([
                        html.Li("Tuning de hiperparámetros para RandomForest."),
                        html.Li("Ensemble de modelos (votación entre RF, KNN, GradientBoosting)."),
                        html.Li("Análisis SHAP para interpretabilidad local."),
                        html.Li("Validación estratificada por departamento y año."),
                        html.Li("Bootstrap para intervalos de confianza en predicciones."),
                    ]),
                ], style={"maxWidth": 800}),
            ], style={"padding": 15}),
        ]),
    ], style={"margin": "5px 10px"}),

    html.Footer([
        html.Hr(),
        html.P("Proyecto -6HoursOfSleep | Fundamentos de Aprendizaje Automático | EAFIT 2026",
               style={"textAlign": "center", "color": "#6c757d"}),
    ], style={"padding": "10px 0"}),
], style={"fontFamily": "Segoe UI, Arial, sans-serif", "maxWidth": 1400, "margin": "auto"})




@app.callback(Output("rmse-chart", "figure"), [Input("rmse-chart", "id")])
def make_rmse(_):
    df = df_test.sort_values("RMSE")
    fig = px.bar(df, x="Modelo", y="RMSE", text="RMSE", color="Modelo",
                 color_discrete_map=MODEL_COLORS, title="RMSE (menor = mejor)")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=40, b=40),
                      yaxis_title="RMSE")
    fig.add_hline(y=df["RMSE"].min(), line_dash="dash", line_color="green",
                  annotation_text=f"Mejor: {df.iloc[0]['RMSE']:.4f}")
    return fig


@app.callback(Output("r2-chart", "figure"), [Input("r2-chart", "id")])
def make_r2(_):
    df = df_test.sort_values("R2", ascending=False)
    fig = px.bar(df, x="Modelo", y="R2", text="R2", color="Modelo",
                 color_discrete_map=MODEL_COLORS, title="R² (mayor = mejor)")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=40, b=40),
                      yaxis_title="R²")
    fig.add_hline(y=df["R2"].max(), line_dash="dash", line_color="green",
                  annotation_text=f"Mejor: {df.iloc[0]['R2']:.4f}")
    return fig


@app.callback(Output("mae-chart", "figure"), [Input("mae-chart", "id")])
def make_mae(_):
    df = df_test.sort_values("MAE")
    fig = px.bar(df, x="Modelo", y="MAE", text="MAE", color="Modelo",
                 color_discrete_map=MODEL_COLORS, title="MAE (menor = mejor)")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=40, b=40),
                      yaxis_title="MAE")
    return fig


@app.callback(Output("mape-chart", "figure"), [Input("mape-chart", "id")])
def make_mape(_):
    df = df_test.sort_values("MAPE")
    fig = px.bar(df, x="Modelo", y="MAPE", text="MAPE", color="Modelo",
                 color_discrete_map=MODEL_COLORS, title="MAPE % (menor = mejor)")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=40, b=40),
                      yaxis_title="MAPE (%)")
    return fig


@app.callback(Output("test-table", "figure"), [Input("test-table", "id")])
def make_test_table(_):
    df = df_test.copy()
    df["Ranking"] = range(1, len(df) + 1)
    df = df[["Ranking", "Modelo", "RMSE", "MAE", "R2", "MAPE"]]
    df["MAPE"] = df["MAPE"].apply(lambda x: f"{x:.2f}%")
    headerColor = "#1a1a2e"
    rowEvenColor = "#f8f9fa"
    rowOddColor = "#fff"
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color=headerColor,
                    font=dict(color="white", size=13), align="center", height=35),
        cells=dict(values=[df[c] for c in df.columns],
                   fill_color=[[rowOddColor if i % 2 == 0 else rowEvenColor
                                for i in range(len(df))]],
                   font=dict(size=12), align="center", height=30))])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350)
    return fig


@app.callback(Output("cv-rmse", "figure"), [Input("cv-rmse", "id")])
def make_cv_rmse(_):
    df = df_cv.sort_values("RMSE_mean")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Modelo"], y=df["RMSE_mean"], error_y=dict(type="data", array=df["RMSE_std"]),
        marker_color=[MODEL_COLORS.get(m, "#636efa") for m in df["Modelo"]],
        text=df["RMSE_mean"].round(4), textposition="outside"))
    fig.update_layout(title="RMSE en Validación Cruzada (con ±1σ)",
                      yaxis_title="RMSE", showlegend=False,
                      margin=dict(l=40, r=20, t=40, b=40))
    return fig


@app.callback(Output("cv-r2", "figure"), [Input("cv-r2", "id")])
def make_cv_r2(_):
    df = df_cv.sort_values("R2_mean", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Modelo"], y=df["R2_mean"], error_y=dict(type="data", array=df["R2_std"]),
        marker_color=[MODEL_COLORS.get(m, "#636efa") for m in df["Modelo"]],
        text=df["R2_mean"].round(4), textposition="outside"))
    fig.update_layout(title="R² en Validación Cruzada (con ±1σ)",
                      yaxis_title="R²", showlegend=False,
                      margin=dict(l=40, r=20, t=40, b=40))
    return fig


@app.callback(Output("cv-table", "figure"), [Input("cv-table", "id")])
def make_cv_table(_):
    df = df_cv.copy()
    df = df.rename(columns={
        "Modelo": "Modelo", "RMSE_mean": "RMSE (media)", "RMSE_std": "RMSE (std)",
        "MAE_mean": "MAE (media)", "MAE_std": "MAE (std)",
        "R2_mean": "R² (media)", "R2_std": "R² (std)",
    })
    for c in df.columns:
        if "std" in c:
            df[c] = df[c].round(4)
    for c in df.columns:
        if "media" in c:
            df[c] = df[c].round(4)
    headerColor = "#1a1a2e"
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color=headerColor,
                    font=dict(color="white", size=12), align="center", height=35),
        cells=dict(values=[df[c] for c in df.columns],
                   fill_color="#f8f9fa",
                   font=dict(size=11), align="center", height=28))])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350)
    return fig


@app.callback(Output("top-features", "figure"), [Input("top-features", "id")])
def make_top_features(_):
    top5 = sup_results["analisis_exploratorio"]["top_5_features_correlacionados"]
    names = [f["nombre"] for f in top5][::-1]
    corrs = [f["correlacion"] for f in top5][::-1]
    fig = px.bar(x=corrs, y=names, orientation="h", color=corrs,
                 color_continuous_scale="Blues",
                 labels={"x": "Correlación (Pearson)", "y": ""},
                 title="Top 5 Variables Correlacionadas con Deserción")
    fig.update_traces(texttemplate="%{x:.4f}", textposition="outside")
    fig.update_layout(margin=dict(l=40, r=30, t=40, b=20), height=350,
                      coloraxis_showscale=False)
    return fig


@app.callback(Output("kmeans-silhouette", "figure"), [Input("kmeans-silhouette", "id")])
def make_kmeans_silhouette(_):
    kmeans = [r for r in unsup_results["kmeans_results"] if r["algorithm"] == "KMeans"]
    ks = [r["k"] for r in kmeans]
    sil = [r["silhouette"] for r in kmeans]
    ch = [r["calinski_harabasz"] / 1000 for r in kmeans]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=sil, mode="lines+markers", name="Silhouette",
                             line=dict(color="#007bff", width=3), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=ks, y=ch, mode="lines+markers", name="CH / 1000",
                             line=dict(color="#dc3545", width=3), marker=dict(size=10),
                             yaxis="y2"))

    best_k = max(kmeans, key=lambda r: r["silhouette"])["k"]
    fig.add_vline(x=best_k, line_dash="dash", line_color="green",
                  annotation_text=f"k óptimo = {best_k}")

    fig.update_layout(
        title="K-Means: Métricas por Número de Clusters",
        xaxis=dict(title="k (número de clusters)", dtick=1),
        yaxis=dict(title="Silhouette Score", range=[0, 0.6]),
        yaxis2=dict(title="Calinski-Harabasz / 1000", overlaying="y", side="right"),
        margin=dict(l=40, r=50, t=40, b=20), height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


@app.callback(Output("clustering-metrics", "figure"), [Input("clustering-metrics", "id")])
def make_clustering_metrics(_):
    best_kmeans = max(unsup_results["kmeans_results"], key=lambda r: r["silhouette"])
    agg = unsup_results["agg_metrics"]
    labels = ["K-Means (k=3)", "Agglomerative (k=3)"]
    sil = [best_kmeans["silhouette"], agg["silhouette"]]
    db = [best_kmeans["davies_bouldin"], agg["davies_bouldin"]]
    ch = [best_kmeans["calinski_harabasz"], agg["calinski_harabasz"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Silhouette (↑)", x=labels, y=sil,
                         marker_color=["#007bff", "#6f42c1"],
                         text=[f"{v:.3f}" for v in sil], textposition="outside"))
    fig.add_trace(go.Bar(name="Davies-Bouldin (↓)", x=labels, y=db,
                         marker_color=["#28a745", "#20c997"],
                         text=[f"{v:.3f}" for v in db], textposition="outside"))
    fig.update_layout(title="Comparación de Métricas de Clustering",
                      yaxis_title="Score", barmode="group",
                      margin=dict(l=40, r=20, t=40, b=20), height=350)
    return fig


@app.callback(Output("dbscan-info", "figure"), [Input("dbscan-info", "id")])
def make_dbscan_info(_):
    dbscan_configs = unsup_results["dbscan_results"]
    n_total = len(dbscan_configs)
    n_zero = sum(1 for r in dbscan_configs if r["n_clusters"] == 0)
    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta", value=n_zero,
        number=dict(suffix=f" / {n_total}"),
        gauge=dict(axis=dict(range=[0, n_total]), bar=dict(color="#dc3545"),
                   steps=[dict(range=[0, n_total], color="#f8d7da")]),
        title=dict(text="Configuraciones DBSCAN sin clusters válidos")))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=30))
    return fig


@app.callback(Output("validation-ari", "figure"), [Input("validation-ari", "id")])
def make_validation_ari(_):
    val = unsup_results["validation"]
    labels = [v["algorithm"] for v in val]
    ari = [v["adjusted_rand_index"] for v in val]
    colors = ["#007bff" if v["algorithm"] == "KMeans" else "#6f42c1" for v in val]
    fig = px.bar(x=labels, y=ari, color=labels, color_discrete_map={
        "KMeans": "#007bff", "Agglomerative": "#6f42c1"},
        text=[f"{v:.4f}" for v in ari],
        labels={"x": "", "y": "Adjusted Rand Index"}, title="Validación vs Target (ARI)")
    fig.update_traces(textposition="outside")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=40, b=20), height=300)
    return fig


@app.callback(Output("index-sim", "figure"), [Input("index-sim", "id")])
def make_index_sim(_):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["PCA", "Teórico"],
        y=[float(idx_sim["INDEX_PCA_spearman_rho"]), float(idx_sim["INDEX_THEORETICAL_spearman_rho"])],
        marker_color=["#007bff", "#6f42c1"],
        text=[f"ρ = {idx_sim['INDEX_PCA_spearman_rho']}",
              f"ρ = {idx_sim['INDEX_THEORETICAL_spearman_rho']}"],
        textposition="outside"))
    fig.update_layout(
        title=f"Dataset Simulado (n={idx_sim['INDEX_PCA_n_observations']})",
        yaxis=dict(title="Spearman ρ", range=[0, 1]),
        margin=dict(l=40, r=20, t=40, b=20), height=350)
    fig.add_hline(y=float(idx_sim["INDEX_PCA_spearman_rho"]), line_dash="dash",
                  line_color="#007bff", opacity=0.5)
    return fig


@app.callback(Output("index-orig", "figure"), [Input("index-orig", "id")])
def make_index_orig(_):
    pca_rho = float(idx_orig["INDEX_PCA_spearman_rho"])
    teor_rho = float(idx_orig["INDEX_THEORETICAL_spearman_rho"])
    pca_p = float(idx_orig["INDEX_PCA_spearman_p_value"])
    teor_p = float(idx_orig["INDEX_THEORETICAL_spearman_p_value"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["PCA", "Teórico"],
        y=[pca_rho, teor_rho],
        marker_color=["#007bff", "#6f42c1"],
        text=[f"ρ = {pca_rho:.4f}" + ("*" if pca_p < 0.05 else ""),
              f"ρ = {teor_rho:.4f}" + ("*" if teor_p < 0.05 else "")],
        textposition="outside"))
    fig.update_layout(
        title=f"Dataset Original (n={idx_orig['INDEX_PCA_n_observations']})",
        yaxis=dict(title="Spearman ρ", range=[0, 1]),
        margin=dict(l=40, r=20, t=40, b=20), height=350)
    fig.add_annotation(x=0.5, y=-0.05, xref="paper", yref="paper",
                       text="* p < 0.05 (significativo)", showarrow=False,
                       font=dict(size=11, color="gray"))
    return fig


@app.callback(Output("index-comparison-table", "figure"), [Input("index-comparison-table", "id")])
def make_index_table(_):
    rows = [
        ["Dataset", "PCA ρ (Spearman)", "PCA p-value", "Teórico ρ (Spearman)", "Teórico p-value",
         "Var. Explicada PC1", "Var. Explicada PC2"],
        ["Simulado", idx_sim["INDEX_PCA_spearman_rho"], idx_sim["INDEX_PCA_spearman_p_value"],
         idx_sim["INDEX_THEORETICAL_spearman_rho"], idx_sim["INDEX_THEORETICAL_spearman_p_value"],
         f"{float(idx_sim['PCA_PC1_variance_explained'])*100:.1f}%",
         f"{float(idx_sim['PCA_PC2_variance_explained'])*100:.1f}%"],
        ["Original", idx_orig["INDEX_PCA_spearman_rho"], idx_orig["INDEX_PCA_spearman_p_value"],
         idx_orig["INDEX_THEORETICAL_spearman_rho"], idx_orig["INDEX_THEORETICAL_spearman_p_value"],
         f"{float(idx_orig['PCA_PC1_variance_explained'])*100:.1f}%",
         f"{float(idx_orig['PCA_PC2_variance_explained'])*100:.1f}%"],
    ]
    headerColor = "#1a1a2e"
    fig = go.Figure(data=[go.Table(
        header=dict(values=rows[0], fill_color=headerColor,
                    font=dict(color="white", size=12), align="center", height=35),
        cells=dict(values=[[r[i] for r in rows[1:]] for i in range(len(rows[0]))],
                   fill_color="#f8f9fa", font=dict(size=11), align="center", height=28))])
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200)
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
