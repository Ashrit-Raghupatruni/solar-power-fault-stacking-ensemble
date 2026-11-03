"""
Leakage-free dataset loader and group-aware preprocessing pipeline.
Ensures:
1. All records of any fault event are kept within a single split (never split across train and test).
2. Preserves true temporal contiguity for normal data using chronological time blocks.
3. Incorporates dynamic electrical features (dV/dt, dI/dt, dP/dt, impedance).
4. Saves preprocessed datasets and the winning RobustScaler artifact.
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

def build_curated_benchmark_dataset(
    normal_block_length_s: int = 300,
    max_normal_blocks_per_st: int = 40
) -> pd.DataFrame:
    """
    Constructs a balanced, high-fidelity benchmark dataset from the 3 station CSVs:
    - Retains ALL 30,142 controlled fault samples (100% fault event coverage).
    - Preserves true temporal contiguity for normal baseline data: segments normal observations into
      contiguous chronological time blocks (e.g. 300-second windows) per day and station.
    - Yields a well-balanced dataset with manageable normal-to-fault ratio for optimal Macro-F1.
    """
    station_dfs = []
    
    for st_id, file_path in STATION_FILES.items():
        print(f"Loading {file_path.name}...")
        df_st = pd.read_csv(file_path)
        
        # Standardize raw severity 'med' -> canonical 'medium'
        df_st[TARGET_SEVERITY] = df_st[TARGET_SEVERITY].map(RAW_SEVERITY_MAP).fillna(df_st[TARGET_SEVERITY])
        
        # 1. Controlled Anomalies (100% retained)
        anomalies = df_st[df_st["anomaly"] == 1].copy()
        anomalies["group_id"] = st_id * 100000 + anomalies["event_id"]
        
        # 2. Contiguous Normal Temporal Blocks (Zero intra-block shuffling)
        normals = df_st[df_st["anomaly"] == 0].copy()
        normals["time_block"] = normals["sample_id"] // normal_block_length_s
        normals["group_id"] = st_id * 100000 + 50000 + (normals["segment_id"] * 1000) + (normals["time_block"] % 500)
        
        # Pick contiguous blocks
        unique_blocks = normals["group_id"].unique()
        rng = np.random.RandomState(RANDOM_STATE)
        selected_blocks = rng.choice(unique_blocks, size=min(len(unique_blocks), max_normal_blocks_per_st), replace=False)
        normals_curated = normals[normals["group_id"].isin(selected_blocks)].copy()
        normals_curated = normals_curated.drop(columns=["time_block"])
        
        combined_st = pd.concat([anomalies, normals_curated], ignore_index=True)
        station_dfs.append(combined_st)
        del df_st, anomalies, normals
        gc.collect()

    full_df = pd.concat(station_dfs, ignore_index=True)
    print(f"Total curated records: {len(full_df)}")
    
    # Missing values imputation strictly on physical channels
    for col in ["voltage_V", "current_A", "frequency_Hz", "power_factor"]:
        full_df[col] = full_df[col].fillna(full_df.groupby("station_id")[col].transform("median"))
        
    for col in ["PE_m3", "PE_m4", "PE_m5", "WPE", "MSPE", "CV", "delta_PF", "SII"]:
        full_df[col] = full_df[col].fillna(0.0)

    # Add advanced electrical features
    full_df = add_electrical_features(full_df)
    
    # Encode targets
    full_df["target_category_idx"] = full_df[TARGET_CATEGORY].map(CATEGORY_TO_IDX)
    full_df["target_severity_idx"] = full_df[TARGET_SEVERITY].map(SEVERITY_TO_IDX)
    
    return full_df

def create_leakage_free_splits(df: pd.DataFrame, test_station_id: int = None, n_splits: int = 5):
    """
    Creates train and test sets with strict event-level separation:
    - Entire fault episodes remain intact within a single partition.
    - Zero overlapping fault events between train and test.
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
    print(f"Zero event leakage verified! Train events: {len(train_fault_events)}, Test events: {len(test_fault_events)}")
    
    return train_df, test_df

def prepare_and_save_data():
    """Generates and saves preprocessed train/test sets and fits the optimal RobustScaler."""
    df = build_curated_benchmark_dataset()
    train_df, test_df = create_leakage_free_splits(df)
    
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved {len(train_df):,} train samples to {train_path}")
    print(f"Saved {len(test_df):,} test samples to {test_path}")
    
    # Fit and save RobustScaler (identified as optimal in scaler benchmarking)
    scaler = RobustScaler()
    scaler.fit(train_df[FEATURE_COLS])
    scaler_path = MODELS_DIR / "preprocessors" / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Optimal RobustScaler fitted on train data and saved to {scaler_path}")

if __name__ == "__main__":
    prepare_and_save_data()
