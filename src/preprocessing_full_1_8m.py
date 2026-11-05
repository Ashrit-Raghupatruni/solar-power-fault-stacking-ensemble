"""
Leak-Free 1.81M Dataset Preprocessing Module.
Methodological Fixes Implemented:
1. Zero Data Leakage: Splits data BEFORE fitting missing-value imputers.
2. Fits SimpleImputer strictly on the training partition; transforms test independently.
3. Enforces boundary-safe temporal derivative calculations.
4. Fits the benchmark RobustScaler strictly on the training partition.
"""
import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from src.config import (
    STATION_FILES, PROCESSED_DATA_DIR, MODELS_DIR,
    FEATURE_COLS, TARGET_CATEGORY, TARGET_SEVERITY,
    CATEGORY_TO_IDX, SEVERITY_TO_IDX, RAW_SEVERITY_MAP, RANDOM_STATE
)
from src.feature_engineering import add_electrical_features
from src.data_loader import verify_dataset_files

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

def load_and_split_leak_free(normal_block_length_s: int = 1200):
    """
    Loads raw station telemetry, creates group-aware split BEFORE imputation,
    and returns properly isolated train and test partitions.
    """
    verify_dataset_files(raise_error=True)
    station_dfs = []
    
    for st_id, file_path in STATION_FILES.items():
        print(f"Loading {file_path.name}...")
        df_st = pd.read_csv(file_path, dtype=DTYPES)
        
        # Sort chronologically by calendar day and timestamp
        df_st = df_st.sort_values(by=["segment_id", "sample_id"]).reset_index(drop=True)
        
        # Standardize raw severity
        df_st[TARGET_SEVERITY] = df_st[TARGET_SEVERITY].map(RAW_SEVERITY_MAP).fillna(df_st[TARGET_SEVERITY])
        
        # 1. Group IDs for Anomalies (per event and station)
        anom_mask = df_st["anomaly"] == 1
        df_st["group_id"] = np.int64(0)
        df_st.loc[anom_mask, "group_id"] = (
            np.int64(st_id) * np.int64(10_000_000) + df_st.loc[anom_mask, "event_id"].astype("int64")
        )
        
        # 2. Group IDs for Normal Contiguous Time Blocks (20-minute chunks)
        # Collision-free: station_id * 10^9 + segment_id * 10^5 + time_block (no modulo wrapping)
        norm_mask = df_st["anomaly"] == 0
        time_block = (df_st.loc[norm_mask, "sample_id"].astype("int64") // normal_block_length_s)
        seg_id_int64 = df_st.loc[norm_mask, "segment_id"].astype("int64")
        df_st.loc[norm_mask, "group_id"] = (
            np.int64(st_id) * np.int64(1_000_000_000) + 
            (seg_id_int64 * np.int64(100_000)) + 
            time_block
        )
        
        # Boundary-safe dynamic feature generation
        df_st = add_electrical_features(df_st)
        
        station_dfs.append(df_st)
        gc.collect()

    full_df = pd.concat(station_dfs, ignore_index=True)
    full_df["target_category_idx"] = full_df[TARGET_CATEGORY].map(CATEGORY_TO_IDX).astype("int8")
    full_df["target_severity_idx"] = full_df[TARGET_SEVERITY].map(SEVERITY_TO_IDX).astype("int8")
    
    # Stratified Group Split
    print("\nSplitting into train/test using StratifiedGroupKFold...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(full_df, full_df["target_category_idx"], full_df["group_id"]))
    
    train_df = full_df.iloc[train_idx].copy()
    test_df = full_df.iloc[test_idx].copy()
    del full_df
    gc.collect()
    
    # Zero leakage assertion
    train_fault_events = set(train_df[train_df["event_id"] != 0]["group_id"].unique())
    test_fault_events = set(test_df[test_df["event_id"] != 0]["group_id"].unique())
    assert len(train_fault_events.intersection(test_fault_events)) == 0, "Data leakage detected across splits!"
    print(f"Zero event leakage verified! Train events: {len(train_fault_events)}, Test events: {len(test_fault_events)}")
    
    # STRICT LEAK-FREE IMPUTATION: Fit median imputer on train partition ONLY
    print("Fitting missing-value imputer strictly on training partition...")
    impute_cols = ["voltage_V", "current_A", "frequency_Hz", "power_factor"]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_df[impute_cols])
    
    train_df[impute_cols] = imputer.transform(train_df[impute_cols])
    test_df[impute_cols] = imputer.transform(test_df[impute_cols])
    
    for col in ["PE_m3", "PE_m4", "PE_m5", "WPE", "MSPE", "CV", "delta_PF", "SII"]:
        train_df[col] = train_df[col].fillna(0.0)
        test_df[col] = test_df[col].fillna(0.0)
        
    joblib.dump(imputer, MODELS_DIR / "preprocessors" / "imputer.pkl")
    print("Saved train-fitted SimpleImputer to models/preprocessors/imputer.pkl")
    
    return train_df, test_df

def run_full_1_8m_preprocessing():
    train_df, test_df = load_and_split_leak_free()
    
    train_csv = PROCESSED_DATA_DIR / "train.csv"
    test_csv = PROCESSED_DATA_DIR / "test.csv"
    
    print("Writing leakage-free processed partitions...")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Saved: {train_csv} ({len(train_df):,} rows)")
    print(f"Saved: {test_csv} ({len(test_df):,} rows)")
    
    # Fit final RobustScaler strictly on train partition
    scaler = RobustScaler()
    scaler.fit(train_df[FEATURE_COLS])
    joblib.dump(scaler, MODELS_DIR / "preprocessors" / "scaler.pkl")
    print("Saved train-fitted RobustScaler to models/preprocessors/scaler.pkl")

if __name__ == "__main__":
    run_full_1_8m_preprocessing()
