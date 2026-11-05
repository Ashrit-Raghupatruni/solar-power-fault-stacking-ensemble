"""
Unit tests for boundary-safe and index-preserving feature engineering.
"""
import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import add_electrical_features

def test_grouped_rolling_index_alignment():
    """
    Tests that grouped rolling calculations preserve original DataFrame row alignment,
    especially when group IDs are interleaved.
    """
    # Create interleaved group IDs: rows 0, 2 are Group 1; rows 1, 3 are Group 2
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
    # index 0: voltage = 220.0, std = 0.0 (min_periods=1)
    # index 2: voltages = [220.0, 230.0], std = std([220, 230]) = 7.0711
    # In Group 2 (indices 1 and 3):
    # index 1: voltage = 100.0, std = 0.0
    # index 3: voltages = [100.0, 110.0], std = std([100, 110]) = 7.0711
    
    assert df_feat.loc[0, "rolling_std_v5"] == 0.0
    assert abs(df_feat.loc[2, "rolling_std_v5"] - 7.0711) < 0.01
    
    assert df_feat.loc[1, "rolling_std_v5"] == 0.0
    assert abs(df_feat.loc[3, "rolling_std_v5"] - 7.0711) < 0.01
    
    # Check impedance proxy calculation
    assert df_feat.loc[0, "impedance_proxy_ohm"] == round(220.0 / 5.0, 4)
    assert df_feat.loc[1, "impedance_proxy_ohm"] == round(100.0 / 10.0, 4)

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
