"""
Leakage-free dataset loader and group-aware preprocessing pipeline.
Ensures:
1. All records of any fault event are kept within a single split (never split across train and test).
2. Supports Leave-One-Station-Out (LOSO) validation.
3. Prepares separate feature representations for Tree vs Scaled (SVM/KNN) models.
"""
import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.config import (
    STATION_FILES, PROCESSED_DATA_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_CATEGORY, TARGET_SEVERITY,
    CATEGORY_TO_IDX, SEVERITY_TO_IDX, RANDOM_STATE
)
from src.feature_engineering import add_electrical_features

def build_curated_benchmark_dataset(sample_normal_per_station: int = 15000) -> pd.DataFrame:
    """
    Constructs a balanced, high-utility benchmark dataset from the 3 station CSVs:
    - Retains ALL 30,142 controlled fault samples (100% fault event coverage).
    - Samples a stratified proportion of normal baseline records per station.
    - Resulting dataset is ~75,000 samples, fast to train on SVM/KNN without dropping rare faults.
    """
    station_dfs = []
    
    for st_id, file_path in STATION_FILES.items():
        print(f"Loading {file_path.name}...")
        df_st = pd.read_csv(file_path)
        
        # Split anomalies vs normal
        anomalies = df_st[df_st["anomaly"] == 1].copy()
        normals = df_st[df_st["anomaly"] == 0].copy()
        
        # Sample normal operating records
        n_sample = min(len(normals), sample_normal_per_station)
        normals_sampled = normals.sample(n=n_sample, random_state=RANDOM_STATE)
        
        combined_st = pd.concat([anomalies, normals_sampled], ignore_index=True)
        station_dfs.append(combined_st)
        del df_st, anomalies, normals
        gc.collect()

    full_df = pd.concat(station_dfs, ignore_index=True)
    print(f"Total curated records: {len(full_df)}")
    
    # Missing values imputation on raw physical values
    for col in ["voltage_V", "current_A", "frequency_Hz", "power_factor"]:
        full_df[col] = full_df[col].fillna(full_df.groupby("station_id")[col].transform("median"))
        
    for col in ["PE_m3", "PE_m4", "PE_m5", "WPE", "MSPE", "CV", "delta_PF", "SII"]:
        full_df[col] = full_df[col].fillna(0.0)

    # Feature Engineering
    full_df = add_electrical_features(full_df)
    
    # Encode targets
    full_df["target_category_idx"] = full_df[TARGET_CATEGORY].map(CATEGORY_TO_IDX)
    full_df["target_severity_idx"] = full_df[TARGET_SEVERITY].map(SEVERITY_TO_IDX)
    
    # Construct unique event group key: station_id * 10000 + event_id
    # Normal samples (event_id == 0) are assigned unique dummy groups to distribute them evenly
    normal_mask = full_df["event_id"] == 0
    full_df["group_id"] = full_df["station_id"] * 100000 + full_df["event_id"]
    
    # Assign distinct group IDs to normal segments (e.g. grouped by segment_id + station)
    full_df.loc[normal_mask, "group_id"] = (
        full_df.loc[normal_mask, "station_id"] * 100000 + 
        full_df.loc[normal_mask, "segment_id"] * 1000 + 
        (full_df.loc[normal_mask].index % 50)
    )

    return full_df

def create_leakage_free_splits(df: pd.DataFrame, test_station_id: int = None, n_splits: int = 5):
    """
    Creates train and test sets without data leakage:
    - If test_station_id is provided: Leave-One-Station-Out (LOSO) evaluation.
    - Otherwise: StratifiedGroupKFold on (event_id, station_id), ensuring no event overlaps.
    """
    if test_station_id is not None:
        print(f"Executing Leave-One-Station-Out (LOSO) split. Testing on Station {test_station_id}...")
        train_df = df[df["station_id"] != test_station_id].copy()
        test_df = df[df["station_id"] == test_station_id].copy()
        return train_df, test_df
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df["target_category_idx"], df["group_id"]))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    # Verify zero event leakage
    train_fault_events = set(train_df[train_df["event_id"] != 0]["group_id"].unique())
    test_fault_events = set(test_df[test_df["event_id"] != 0]["group_id"].unique())
    overlap = train_fault_events.intersection(test_fault_events)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping fault event groups: {overlap}"
    print(f"Zero event leakage verified. Train events: {len(train_fault_events)}, Test events: {len(test_fault_events)}")
    
    return train_df, test_df

def prepare_and_save_data():
    """Generates and saves the processed train and test datasets and preprocessor scalers."""
    df = build_curated_benchmark_dataset()
    train_df, test_df = create_leakage_free_splits(df)
    
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved {len(train_df)} train samples to {train_path}")
    print(f"Saved {len(test_df)} test samples to {test_path}")
    
    # Fit and save StandardScaler on train split only (prevent test leakage)
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])
    scaler_path = MODELS_DIR / "preprocessors" / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"StandardScaler fitted on train data and saved to {scaler_path}")

if __name__ == "__main__":
    prepare_and_save_data()
