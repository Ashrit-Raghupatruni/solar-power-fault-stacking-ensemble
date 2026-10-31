"""
Global configuration for Solar & Electrical Power Fault and Anomaly Detection.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Database" / "controlled-fault-benchmark-v1.0"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for p in [PROCESSED_DATA_DIR, MODELS_DIR / "fault_category", MODELS_DIR / "fault_severity", MODELS_DIR / "preprocessors", FIGURES_DIR, METRICS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Station CSV files
STATION_FILES = {
    1: DATA_DIR / "station1_normal_features.csv",
    2: DATA_DIR / "station2_medium_features.csv",
    3: DATA_DIR / "station3_high_features.csv"
}

# Raw physical sensor telemetry features
RAW_SENSOR_FEATURES = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

# Edge precomputed statistical / non-linear features in Zenodo benchmark
EDGE_FEATURES = [
    "PE_m3",
    "PE_m4",
    "PE_m5",
    "WPE",
    "MSPE",
    "CV",
    "delta_PF",
    "SII"
]

# Engineered domain features
ENGINEERED_FEATURES = [
    "apparent_power_VA",
    "reactive_power_var",
    "power_active_apparent_ratio",
    "voltage_deviation_pct",
    "frequency_deviation_pct",
    "current_per_watt"
]

FEATURE_COLS = RAW_SENSOR_FEATURES + EDGE_FEATURES + ENGINEERED_FEATURES

# Target definitions
TARGET_CATEGORY = "fault_category"
TARGET_SEVERITY = "severity"
TARGET_BINARY = "anomaly"

# 8 Multiclass Fault Categories (+ normal)
CATEGORY_LABELS = [
    "normal",
    "voltage",
    "load",
    "power_quality",
    "sensor",
    "outage",
    "frequency",
    "context",
    "compound"
]

# Severity grades (+ none)
# In raw CSVs: 'none', 'incipient', 'low', 'med', 'high', 'critical'
# Canonical standardized internal representation: 'medium' mapped from raw 'med'
SEVERITY_LABELS = [
    "none",
    "incipient",
    "low",
    "medium",
    "high",
    "critical"
]

# Raw CSV to canonical mapping dictionary
RAW_SEVERITY_MAP = {
    "none": "none",
    "incipient": "incipient",
    "low": "low",
    "med": "medium",
    "medium": "medium",
    "high": "high",
    "critical": "critical"
}

# Categorical mapping dictionaries
CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORY_LABELS)}
IDX_TO_CATEGORY = {idx: cat for cat, idx in CATEGORY_TO_IDX.items()}

SEVERITY_TO_IDX = {sev: idx for idx, sev in enumerate(SEVERITY_LABELS)}
IDX_TO_SEVERITY = {idx: sev for sev, idx in SEVERITY_TO_IDX.items()}

RANDOM_STATE = 42
