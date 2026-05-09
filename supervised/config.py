# Config for supervised models pipeline
# Variables de configuración globales para el pipeline

# ── Rutas ──────────────────────────────────────────────────────────────────────
DATA_PATH = "../data/final/dataset_con_indice.csv"
REPORTS_DIR = "../reports"
FIGURES_DIR = "../reports/eda_figures"

# ── Configuración del Modelado ──────────────────────────────────────────────────
RANDOM_STATE = 42          # Para reproducibilidad
TEST_SIZE = 0.20           # 80% train, 20% test
CV_FOLDS = 5               # Validación cruzada k-fold
TARGET_COL = "outcome_tasa_desercion_snies"

# ── Hiperparámetros de Modelos ────────────────────────────────────────────────

# Baseline
PARAMS_RIDGE = {"alpha": 1.0}
PARAMS_LASSO = {"alpha": 0.1, "max_iter": 5000}
PARAMS_ELASTICNET = {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 5000}

# Avanzados
PARAMS_RF = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2
}

PARAMS_GBM = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_split": 5
}

PARAMS_SVR = {
    "kernel": "rbf",
    "C": 1.0,
    "epsilon": 0.1,
    "gamma": "scale"
}

PARAMS_KNN = {
    "n_neighbors": 5,
    "weights": "distance",
    "metric": "euclidean"
}

# ── Visualización ──────────────────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
PLOT_STYLE = "default"
FEATURE_IMPORTANCE_TOP_N = 15

# ── Logging ────────────────────────────────────────────────────────────────────
VERBOSE = True
SAVE_PLOTS = True
SAVE_TABLES = True
