from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHART_DIR = PROJECT_DIR / "charts"

# Simulation controls
N_PATHS = 10000
STEPS_PER_YEAR = 4
SEED = 42
PFE_LEVELS = [0.95, 0.99]
PATHS_TO_SAVE = 50

# Wrong-way risk controls
WWR_SCENARIOS = ["independent", "moderate", "severe"]
WWR_PFE_LEVEL = 0.95
WWR_CORRELATION_SWEEP = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
WWR_PATHS_TO_SAVE = 50
WWR_SCATTER_PATHS = 2000
WWR_FOCUS_COUNTERPARTY = "Bank B"
