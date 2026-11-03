"""
Feature engineering module for electrical power and solar photovoltaic (PV) inverter telemetry.
Derives static AC quantities, power-factor stability, dynamic rates of change, and impedance proxies.
"""
import numpy as np
import pandas as pd

def add_electrical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain AC physical features from raw voltage, current, power, and frequency.
    Works seamlessly for both IoT Edge power monitors and Solar PV inverters.
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
    
    # 3. Active-to-Apparent Power Ratio (Efficiency / distortion indicator)
    denom = df["apparent_power_VA"].replace(0.0, np.nan)
    df["power_active_apparent_ratio"] = (df["power_W"].abs() / denom).fillna(1.0).clip(0.0, 2.0).round(4)
    
    # 4. Voltage deviation percentage from nominal (220.0 V standard AC)
    v_nominal = 220.0
    df["voltage_deviation_pct"] = (((df["voltage_V"] - v_nominal) / v_nominal) * 100.0).round(4)
    
    # 5. Frequency deviation percentage from nominal (50.0 Hz standard grid)
    f_nominal = 50.0
    df["frequency_deviation_pct"] = (((df["frequency_Hz"] - f_nominal) / f_nominal) * 100.0).round(4)
    
    # 6. Current per Watt intensity ratio (high during voltage sags or low PF)
    p_denom = df["power_W"].abs().replace(0.0, np.nan)
    df["current_per_watt"] = (i_abs / p_denom).fillna(0.0).clip(0.0, 1.0).round(6)

    # 7. Dynamic Rate-of-Change / Transient Features (dV/dt, dI/dt, dP/dt)
    # If grouped by event/station, compute temporal differences; otherwise fall back to 0.0
    if "station_id" in df.columns:
        df["d_voltage_dt"] = df.groupby("station_id")["voltage_V"].diff().fillna(0.0).abs().round(4)
        df["d_current_dt"] = df.groupby("station_id")["current_A"].diff().fillna(0.0).abs().round(4)
        df["d_power_dt"] = df.groupby("station_id")["power_W"].diff().fillna(0.0).abs().round(4)
    else:
        df["d_voltage_dt"] = df["voltage_V"].diff().fillna(0.0).abs().round(4)
        df["d_current_dt"] = df["current_A"].diff().fillna(0.0).abs().round(4)
        df["d_power_dt"] = df["power_W"].diff().fillna(0.0).abs().round(4)

    # 8. Power Factor Stability Indicator: abs(1.0 - abs(PF))
    df["power_factor_stability"] = (1.0 - pf.abs()).clip(0.0, 1.0).round(4)

    # 9. Impedance Proxy Z = V / I (Ohms, indicators of short circuits or open lines)
    i_denom = i_abs.replace(0.0, np.nan)
    df["impedance_proxy_ohm"] = (v_abs / i_denom).fillna(999.0).clip(0.0, 1000.0).round(4)
    
    return df
