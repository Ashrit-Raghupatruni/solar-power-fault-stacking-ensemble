"""
Full-scale (1.81 Million data points) leakage-free data processing pipeline.
Loads all 3 station CSVs without downsampling:
- Station 1: 604,800 rows
- Station 2: 604,800 rows
- Station 3: 604,800 rows
Total: 1,814,400 per-second observations.

Employs memory-efficient float32 downcasting and group-aware splitting by (station_id, event_id, segment_id).
"""
import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from src.config import (
    STATION_FILES, PROCESSED_DATA_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_CATEGORY, TARGET_SEVERITY,
    CATEGORY_TO_IDX, SEVERITY_TO_IDX, RAW_SEVERITY_MAP, RANDOM_STATE
)
from src.feature_engineering import add_electrical_features

DTYPES = {
    "sample_id": "int32",
    "station_id": "int8",
    "segment_id": "int8",
    "voltage_V": "float32",
    "current_A": "float32",
    "power_W": "float32",
    "energy_kWh": "float32",
    "frequency_Hz": "float32",
    "power_factor": "float32",
    "wifi_rssi_dBm": "float32",
    "is_interpolated": "int8",
    "gap_length_s": "int16",
    "PE_m3": "float32",
    "PE_m4": "float32",
    "PE_m5": "float32",
    "WPE": "float32",
    "MSPE": "float32",
    "CV": "float32",
    "delta_PF": "float32",
    "SII": "float32",
    "threshold": "float32",
    "detector_flag": "int8",
    "anomaly": "int8",
    "event_id": "int32"
}

def load_full_1_8m_dataset(normal_block_length_s: int = 1200) -> pd.DataFrame:
    """
    Loads the entire 1,814,400-row benchmark from all three stations:
    - Retains 100% of all data points.
    - Groups normal records into contiguous chronological 20-minute (1200s) time blocks
      to enable valid group-aware cross-validation without temporal fracturing.
    """
    station_dfs = []
    
    for st_id, file_path in STATION_FILES.items():
        print(f"Loading complete 604,800 records from {file_path.name}...")
        df_st = pd.read_csv(file_path, dtype=DTYPES)
        
        # Standardize raw severity 'med' -> canonical 'medium'
        df_st[TARGET_SEVERITY] = df_st[TARGET_SEVERITY].map(RAW_SEVERITY_MAP).fillna(df_st[TARGET_SEVERITY])
        
        # 1. Group IDs for Anomalies
        anom_mask = df_st["anomaly"] == 1
        df_st["group_id"] = 0
        df_st.loc[anom_mask, "group_id"] = (st_id * 100000 + df_st.loc[anom_mask, "event_id"]).astype("int64")
        
        # 2. Group IDs for Normal Contiguous Time Blocks
        norm_mask = df_st["anomaly"] == 0
        time_block = df_st.loc[norm_mask, "sample_id"] // normal_block_length_s
        df_st.loc[norm_mask, "group_id"] = (
            st_id * 100000 + 50000 + 
            (df_st.loc[norm_mask, "segment_id"] * 1000) + 
            (time_block % 500)
        ).astype("int64")
        
        # Missing values imputation strictly on physical channels
        for col in ["voltage_V", "current_A", "frequency_Hz", "power_factor"]:
            median_val = df_st[col].median()
            df_st[col] = df_st[col].fillna(median_val)
            
        for col in ["PE_m3", "PE_m4", "PE_m5", "WPE", "MSPE", "CV", "delta_PF", "SII"]:
            df_st[col] = df_st[col].fillna(0.0)
            
        # Add upgraded dynamic electrical features
        df_st = add_electrical_features(df_st)
        
        station_dfs.append(df_st)
        gc.collect()

    full_df = pd.concat(station_dfs, ignore_index=True)
    print(f"\nSuccessfully assembled full dataset! Total records: {len(full_df):,}")
    
    # Encode targets
    full_df["target_category_idx"] = full_df[TARGET_CATEGORY].map(CATEGORY_TO_IDX).astype("int8")
    full_df["target_severity_idx"] = full_df[TARGET_SEVERITY].map(SEVERITY_TO_IDX).astype("int8")
    
    return full_df

def create_full_leakage_free_splits(df: pd.DataFrame, n_splits: int = 5):
    """
    Creates an 80/20 train/test split on the full 1.81M records using StratifiedGroupKFold
    guaranteeing zero event leakage.
    """
    print("Splitting full 1.81M dataset using StratifiedGroupKFold...")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df["target_category_idx"], df["group_id"]))
    
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    train_fault_events = set(train_df[train_df["event_id"] != 0]["group_id"].unique())
    test_fault_events = set(test_df[test_df["event_id"] != 0]["group_id"].unique())
    overlap = train_fault_events.intersection(test_fault_events)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping fault event groups: {overlap}"
    
    print(f"Zero event leakage verified on full dataset! Train events: {len(train_fault_events)}, Test events: {len(test_fault_events)}")
    print(f"Train set: {len(train_df):,} samples | Test set: {len(test_df):,} samples")
    
    return train_df, test_df

def run_full_1_8m_preprocessing(save_parquet: bool = True):
    """Processes all 1.81M rows and saves memory-optimized Parquet or CSV partitions."""
    df = load_full_1_8m_dataset()
    train_df, test_df = create_full_leakage_free_splits(df)
    
    # Save partitions
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    test_csv = PROCESSED_DATA_DIR / "test.csv"
    
    print("Writing train and test datasets...")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Saved: {train_csv} ({os.path.getsize(train_csv) / 1e6:.1f} MB)")
    print(f"Saved: {test_csv} ({os.path.getsize(test_csv) / 1e6:.1f} MB)")
    
    # Fit and save RobustScaler
    scaler = RobustScaler()
    scaler.fit(train_df[FEATURE_COLS])
    scaler_path = MODELS_DIR / "preprocessors" / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Fitted RobustScaler on full train set and saved to {scaler_path}")

if __name__ == "__main__":
    run_full_1_8m_preprocessing()
