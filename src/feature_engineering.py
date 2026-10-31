"""
Feature engineering module for electrical power and solar inverter telemetry.
Derives domain-specific electrical quantities (apparent power, reactive power, deviation metrics).
"""
import numpy as np
import pandas as pd

def add_electrical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physical AC electrical features from raw voltage, current, power, and frequency.
    Works for both Zenodo power monitoring nodes and Solar PV inverters.
    """
    df = df.copy()
    
    # Clip power factor within valid range [-1.0, 1.0]
    pf = df["power_factor"].clip(-1.0, 1.0).fillna(1.0)
    
    # Apparent Power S = V * I (VA)
    df["apparent_power_VA"] = (df["voltage_V"].abs() * df["current_A"].abs()).round(4)
    
    # Reactive Power Q = sqrt(max(0, S^2 - P^2)) or S * sin(acos(PF))
    sin_phi = np.sqrt(np.maximum(0.0, 1.0 - (pf ** 2)))
    df["reactive_power_var"] = (df["apparent_power_VA"] * sin_phi).round(4)
    
    # Active to Apparent Power Ratio (Efficiency / distortion indicator)
    denom = df["apparent_power_VA"].replace(0.0, np.nan)
    df["power_active_apparent_ratio"] = (df["power_W"].abs() / denom).fillna(1.0).clip(0.0, 2.0).round(4)
    
    # Voltage deviation percentage from nominal (220 V standard AC)
    v_nominal = 220.0
    df["voltage_deviation_pct"] = (((df["voltage_V"] - v_nominal) / v_nominal) * 100.0).round(4)
    
    # Frequency deviation percentage from nominal (50.0 Hz standard grid)
    f_nominal = 50.0
    df["frequency_deviation_pct"] = (((df["frequency_Hz"] - f_nominal) / f_nominal) * 100.0).round(4)
    
    # Current per Watt intensity ratio (high during voltage sags or low PF)
    p_denom = df["power_W"].abs().replace(0.0, np.nan)
    df["current_per_watt"] = (df["current_A"].abs() / p_denom).fillna(0.0).clip(0.0, 1.0).round(6)
    
    return df
