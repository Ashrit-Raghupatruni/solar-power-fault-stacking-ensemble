"""
Solar Photovoltaic (PV) Telemetry Adapter for AhoyDTU / OpenDTU Microinverters.
Ingests multi-channel DC string telemetry and AC grid-tie inverter readings:
1. Calculates DC Array yield and inverter conversion efficiency:
   eta = (P_ac / sum(P_dc)) * 100
2. Extracts AC grid-tie interface metrics strictly mapping to the unified 16-feature operational schema.
3. Keeps DC string readings isolated for PV health monitoring (never polluting AC classifier).
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.feature_engineering import add_electrical_features
from src.config import OPERATIONAL_FEATURE_COLS

REQUIRED_SOLAR_FIELDS = ["voltage_ac", "current_ac"]

def adapt_pv_telemetry(pv_data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Adapts Solar PV Inverter (AhoyDTU / Hoymiles / OpenDTU) telemetry.
    
    Expected schema:
      - voltage_ac (V): AC output line voltage
      - current_ac (A): AC output current
      - power_ac (W, optional): Active power output
      - frequency_ac (Hz, optional): AC frequency (default 50.0)
      - power_factor (optional): AC power factor (default 0.98)
      - ch1_v, ch1_i .. ch4_v, ch4_i: DC string measurements
    
    Returns:
      (dataframe with 16 operational features, metadata dictionary with DC analytics)
    """
    metadata = {
        "valid": True,
        "missing_features": [],
        "warnings": [],
        "dc_analytics": {
            "total_dc_power_w": 0.0,
            "channels": {},
            "efficiency_pct": 0.0,
            "operating_mode": "Normal Generation"
        }
    }
    
    # 1. Validate mandatory AC parameters
    for field in REQUIRED_SOLAR_FIELDS:
        if field not in pv_data or pv_data[field] is None:
            metadata["valid"] = False
            metadata["missing_features"].append(field)
            metadata["warnings"].append(f"Essential AC parameter '{field}' is missing.")
            return pd.DataFrame(), metadata

    v_ac = float(pv_data["voltage_ac"])
    i_ac = float(pv_data["current_ac"])
    
    # Power handling
    if "power_ac" in pv_data and pv_data["power_ac"] is not None:
        p_ac = float(pv_data["power_ac"])
    else:
        metadata["missing_features"].append("power_ac")
        metadata["warnings"].append("power_ac not supplied; estimated using V * I * PF.")
        pf_est = float(pv_data.get("power_factor", 0.98) or 0.98)
        p_ac = round(v_ac * i_ac * pf_est, 2)
        
    # Frequency handling
    if "frequency_ac" in pv_data and pv_data["frequency_ac"] is not None:
        f_ac = float(pv_data["frequency_ac"])
    else:
        metadata["missing_features"].append("frequency_ac")
        f_ac = 50.0
        
    # Power Factor handling
    if "power_factor" in pv_data and pv_data["power_factor"] is not None:
        pf = float(pv_data["power_factor"])
    else:
        metadata["missing_features"].append("power_factor")
        pf = 0.98

    # 2. Process DC String Telemetry (Channels 1 to 4)
    total_dc_w = 0.0
    for ch in range(1, 5):
        v_key = f"ch{ch}_v"
        i_key = f"ch{ch}_i"
        if v_key in pv_data and i_key in pv_data and pv_data[v_key] is not None and pv_data[i_key] is not None:
            v_dc = float(pv_data[v_key])
            i_dc = float(pv_data[i_key])
            p_dc = round(v_dc * i_dc, 2)
            total_dc_w += p_dc
            metadata["dc_analytics"]["channels"][f"string_{ch}"] = {
                "voltage_v": v_dc,
                "current_a": i_dc,
                "power_w": p_dc
            }

    metadata["dc_analytics"]["total_dc_power_w"] = round(total_dc_w, 2)

    # Calculate Inverter Conversion Efficiency
    if total_dc_w >= 10.0:
        eff = round((p_ac / total_dc_w) * 100.0, 2)
        metadata["dc_analytics"]["efficiency_pct"] = min(100.0, max(0.0, eff))
    else:
        metadata["dc_analytics"]["efficiency_pct"] = 0.0
        if p_ac <= 5.0:
            metadata["dc_analytics"]["operating_mode"] = "Standby / Night-time"

    # 3. Build Standardized AC Operational Row
    standard_row = {
        "voltage_V": v_ac,
        "current_A": i_ac,
        "power_W": p_ac,
        "frequency_Hz": f_ac,
        "power_factor": pf
    }
    
    df_raw = pd.DataFrame([standard_row])
    df_engineered = add_electrical_features(df_raw)
    
    # Verify that all 16 operational columns exist
    df_operational = df_engineered[OPERATIONAL_FEATURE_COLS].copy()
    
    return df_operational, metadata
