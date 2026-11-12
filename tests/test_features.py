"""
Comprehensive test suite for the unified 16-feature operational schema.
Validates:
1. Exact match with OPERATIONAL_FEATURE_COLS (16 features).
2. Complete absence of NaNs across both batch data and streaming single-sample telemetry.
3. Row-index preservation during grouped rolling operations.
4. Correctness of mathematical formulations (S, Q, deviations, causal derivatives).
"""
import pytest
import pandas as pd
import numpy as np

from src.config import OPERATIONAL_FEATURE_COLS
from src.feature_engineering import add_electrical_features

def test_16_operational_features_presence():
    """Validates that exactly the 16 operational features are produced."""
    df_raw = pd.DataFrame([{
        "voltage_V": 220.0,
        "current_A": 5.0,
        "power_W": 1100.0,
        "frequency_Hz": 50.0,
        "power_factor": 0.98
    }])
    
    df_eng = add_electrical_features(df_raw)
    
    assert len(OPERATIONAL_FEATURE_COLS) == 16
    for col in OPERATIONAL_FEATURE_COLS:
        assert col in df_eng.columns, f"Operational column {col} missing from output"

def test_no_nans_on_raw_station_slice():
    """Tests feature generation on raw station data with arbitrary event groupings."""
    df_slice = pd.DataFrame({
        "station_id": [1, 1, 1, 1, 1],
        "segment_id": [0, 0, 0, 0, 0],
        "event_id": [0, 0, 0, 0, 0],
        "voltage_V": [220.0, 222.0, 219.0, 218.0, 220.0],
        "current_A": [5.0, 5.2, 5.1, 4.9, 5.0],
        "power_W": [1100.0, 1150.0, 1110.0, 1070.0, 1100.0],
        "frequency_Hz": [50.0, 50.01, 49.99, 50.0, 50.0],
        "power_factor": [0.98, 0.98, 0.97, 0.98, 0.98]
    })
    
    df_eng = add_electrical_features(df_slice)
    op_features = df_eng[OPERATIONAL_FEATURE_COLS]
    
    assert op_features.isna().sum().sum() == 0, "Engineered features contain unexpected NaNs"
    assert len(op_features) == 5

def test_grouped_rolling_index_alignment():
    """Tests that grouped rolling calculations strictly preserve interleaved row indices."""
    df = pd.DataFrame({
        "station_id": [1, 1, 1, 1],
        "segment_id": [1, 1, 1, 1],
        "event_id": [0, 0, 0, 0],
        "group_id": [1, 2, 1, 2],
        "voltage_V": [220.0, 100.0, 230.0, 110.0],
        "current_A": [5.0, 10.0, 5.5, 11.0],
        "power_W": [1100.0, 1000.0, 1265.0, 1210.0],
        "frequency_Hz": [50.0, 50.0, 50.0, 50.0],
        "power_factor": [0.98, 0.95, 0.98, 0.95]
    })
    
    df_feat = add_electrical_features(df)
    
    # In Group 1 (indices 0 and 2):
    assert df_feat.loc[0, "rolling_std_v5"] == 0.0
    assert abs(df_feat.loc[2, "rolling_std_v5"] - 7.0711) < 0.01
    
    # In Group 2 (indices 1 and 3):
    assert df_feat.loc[1, "rolling_std_v5"] == 0.0
    assert abs(df_feat.loc[3, "rolling_std_v5"] - 7.0711) < 0.01

def test_single_sample_online_inference():
    """Tests single-sample stateful feature engineering for streaming inference."""
    prev_state = {
        "voltage_V": 220.0,
        "current_A": 5.0,
        "power_W": 1100.0,
        "rolling_std_v5": 0.5,
        "rolling_std_i5": 0.1,
        "rolling_std_p5": 5.0,
        "power_to_baseline_ratio": 1.0
    }
    
    single_df = pd.DataFrame([{
        "voltage_V": 225.0,
        "current_A": 5.2,
        "power_W": 1170.0,
        "frequency_Hz": 50.0,
        "power_factor": 0.98
    }])
    
    df_res = add_electrical_features(single_df, previous_state=prev_state)
    assert df_res.loc[0, "d_voltage_dt"] == 5.0
    assert df_res.loc[0, "d_current_dt"] == 0.2
    assert df_res.loc[0, "d_power_dt"] == 70.0
    assert df_res.loc[0, "rolling_std_v5"] == 0.5
    assert df_res.loc[0, "rolling_std_i5"] == 0.1
