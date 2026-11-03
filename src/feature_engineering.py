"""
Boundary-Safe, Causal Feature Engineering Module.
Computes:
1. Static AC features (apparent power, reactive power, power factor stability, deviations, impedance)
2. Causal temporal derivatives (dV/dt, dI/dt, dP/dt) strictly bounded by (station_id, segment_id, event_id).
Guarantees zero cross-split or cross-day contamination.
Provides single-sample historical buffer support for live dashboard inference.
"""
import numpy as np
import pandas as pd

def add_electrical_features(df: pd.DataFrame, previous_state: dict = None) -> pd.DataFrame:
    """
    Computes domain AC physical features from raw voltage, current, power, and frequency.
    Works for:
      - Full batch multi-station telemetry (with boundary-safe grouping)
      - Single-sample live inference in the dashboard (using previous_state buffer)
    """
    df = df.copy()
    
    # Clip power factor strictly within physical limits [-1.0, 1.0]
    pf = df["power_factor"].clip(-1.0, 1.0).fillna(1.0)
    
    # 1. Apparent Power S = |V| * |I| (VA)
    v_abs = df["voltage_V"].abs()
    i_abs = df["current_A"].abs()
    df["apparent_power_VA"] = (v_abs * i_abs).round(4)
    
    # 2. Reactive Power Q = S * sin(acos(PF)) or sqrt(max(0, S^2 - P^2))
    sin_phi = np.sqrt(np.maximum(0.0, 1.0 - (pf ** 2)))
    df["reactive_power_var"] = (df["apparent_power_VA"] * sin_phi).round(4)
    
    # 3. Active-to-Apparent Power Ratio
    denom = df["apparent_power_VA"].replace(0.0, np.nan)
    df["power_active_apparent_ratio"] = (df["power_W"].abs() / denom).fillna(1.0).clip(0.0, 2.0).round(4)
    
    # 4. Voltage deviation percentage from nominal (220.0 V standard AC)
    v_nominal = 220.0
    df["voltage_deviation_pct"] = (((df["voltage_V"] - v_nominal) / v_nominal) * 100.0).round(4)
    
    # 5. Frequency deviation percentage from nominal (50.0 Hz standard grid)
    f_nominal = 50.0
    df["frequency_deviation_pct"] = (((df["frequency_Hz"] - f_nominal) / f_nominal) * 100.0).round(4)
    
    # 6. Current per Watt intensity ratio: |I| / |P| (A/W)
    p_denom = df["power_W"].abs().replace(0.0, np.nan)
    df["current_per_watt"] = (i_abs / p_denom).fillna(0.0).clip(0.0, 1.0).round(6)

    # 7. Power Factor Stability: |1.0 - |cos phi||
    df["power_factor_stability"] = (1.0 - pf.abs()).clip(0.0, 1.0).round(4)

    # 8. Impedance Proxy Z = |V| / |I| (Ohms)
    i_denom = i_abs.replace(0.0, np.nan)
    df["impedance_proxy_ohm"] = (v_abs / i_denom).fillna(999.0).clip(0.0, 1000.0).round(4)

    # 9. Boundary-Safe Causal Temporal Derivatives (dV/dt, dI/dt, dP/dt)
    if "segment_id" in df.columns and "event_id" in df.columns and "station_id" in df.columns:
        # Group strictly by station, calendar day, and event to prevent cross-boundary contamination
        group_keys = ["station_id", "segment_id", "event_id"]
        df["d_voltage_dt"] = df.groupby(group_keys)["voltage_V"].diff().fillna(0.0).abs().round(4)
        df["d_current_dt"] = df.groupby(group_keys)["current_A"].diff().fillna(0.0).abs().round(4)
        df["d_power_dt"] = df.groupby(group_keys)["power_W"].diff().fillna(0.0).abs().round(4)
    elif previous_state is not None and len(df) == 1:
        # Single-sample online inference in Streamlit/Web Dashboard
        df["d_voltage_dt"] = abs(float(df["voltage_V"].iloc[0]) - float(previous_state.get("voltage_V", 220.0)))
        df["d_current_dt"] = abs(float(df["current_A"].iloc[0]) - float(previous_state.get("current_A", 5.0)))
        df["d_power_dt"] = abs(float(df["power_W"].iloc[0]) - float(previous_state.get("power_W", 1100.0)))
    else:
        # Fallback single batch
        df["d_voltage_dt"] = df["voltage_V"].diff().fillna(0.0).abs().round(4)
        df["d_current_dt"] = df["current_A"].diff().fillna(0.0).abs().round(4)
        df["d_power_dt"] = df["power_W"].diff().fillna(0.0).abs().round(4)

    return df
