"""
Global configuration for Solar & Electrical Power Fault and Anomaly Detection.
Standardizes paths, features, target taxonomies, and canonical mappings.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Database" / "controlled-fault-benchmark-v1.0"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for p in [
    PROCESSED_DATA_DIR,
    MODELS_DIR / "stage1_binary",
    MODELS_DIR / "stage2_fault_category",
    MODELS_DIR / "stage2_fault_severity",
    MODELS_DIR / "fault_category",
    MODELS_DIR / "fault_severity",
    MODELS_DIR / "preprocessors",
    FIGURES_DIR,
    METRICS_DIR
]:
    p.mkdir(parents=True, exist_ok=True)

# Station CSV files
STATION_FILES = {
    1: DATA_DIR / "station1_normal_features.csv",
    2: DATA_DIR / "station2_medium_features.csv",
    3: DATA_DIR / "station3_high_features.csv"
}

# 1. Raw physical sensor telemetry features (5)
RAW_SENSOR_FEATURES = [
    "voltage_V",
    "current_A",
    "power_W",
    "frequency_Hz",
    "power_factor"
]

# 2. Domain Electrical, Causal Derivatives & Rolling Volatility Features (11)
ENGINEERED_FEATURES = [
    "apparent_power_VA",
    "reactive_power_var",
    "power_active_apparent_ratio",
    "voltage_deviation_pct",
    "frequency_deviation_pct",
    "d_voltage_dt",
    "d_current_dt",
    "d_power_dt",
    "rolling_std_v5",
    "rolling_std_i5",
    "power_to_baseline_ratio"
]

# Unified 16 Operational Features:
# Strictly identical across offline 1.81M dataset, batch validation, API inference, and 1 Hz live streaming
OPERATIONAL_FEATURE_COLS = RAW_SENSOR_FEATURES + ENGINEERED_FEATURES

# The official feature column list for all models
FEATURE_COLS = OPERATIONAL_FEATURE_COLS

# Target definitions
TARGET_CATEGORY = "fault_category"
TARGET_SEVERITY = "severity"
TARGET_BINARY = "anomaly"

# 8 Controlled Anomalies Exclusively for Stage 2 Multi-Class Category Identification
STAGE2_FAULT_CATEGORIES = [
    "voltage",
    "load",
    "frequency",
    "power_quality",
    "outage",
    "sensor",
    "context",
    "compound"
]

# 5 Controlled Anomalies Exclusively for Stage 2 Multi-Class Severity Identification
# Ground truth benchmark uses 'med'
STAGE2_FAULT_SEVERITIES = [
    "incipient",
    "low",
    "med",
    "high",
    "critical"
]

# 9 Multiclass Fault Categories (Including Normal for flat pipelines)
CATEGORY_LABELS = [
    "normal",
    "voltage",
    "load",
    "frequency",
    "power_quality",
    "outage",
    "sensor",
    "context",
    "compound"
]

# 6 Severity grades (Including None for flat pipelines) - Benchmark ground truth uses 'med'
SEVERITY_LABELS = [
    "none",
    "incipient",
    "low",
    "med",
    "high",
    "critical"
]

# Raw CSV to canonical mapping dictionary (maps both 'med' and 'medium' to 'med')
RAW_SEVERITY_MAP = {
    "none": "none",
    "incipient": "incipient",
    "low": "low",
    "med": "med",
    "medium": "med",
    "high": "high",
    "critical": "critical"
}

CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORY_LABELS)}
IDX_TO_CATEGORY = {idx: cat for cat, idx in CATEGORY_TO_IDX.items()}

SEVERITY_TO_IDX = {sev: idx for idx, sev in enumerate(SEVERITY_LABELS)}
IDX_TO_SEVERITY = {idx: sev for sev, idx in SEVERITY_TO_IDX.items()}

STAGE2_CAT_TO_IDX = {cat: idx for idx, cat in enumerate(STAGE2_FAULT_CATEGORIES)}
STAGE2_IDX_TO_CAT = {idx: cat for cat, idx in STAGE2_CAT_TO_IDX.items()}

STAGE2_SEV_TO_IDX = {sev: idx for idx, sev in enumerate(STAGE2_FAULT_SEVERITIES)}
STAGE2_IDX_TO_SEV = {idx: sev for sev, idx in STAGE2_SEV_TO_IDX.items()}

RANDOM_STATE = 42
