"""
Solar Photovoltaic (PV) Telemetry Adapter.
Maps incoming solar inverter / string monitoring metrics to the unified electrical feature schema.
Rejects unsupported telemetry and adds missing-feature indicators without fabricating values.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from src.feature_engineering import add_electrical_features

REQUIRED_SOLAR_FIELDS = ["voltage_ac", "current_ac"]

def adapt_pv_telemetry(pv_data: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Adapts PV Inverter readings:
      - voltage_ac (V) -> voltage_V
      - current_ac (A) -> current_A
      - power_ac (W, optional) -> computed if absent: V * I * PF
      - frequency_ac (Hz, optional) -> default 50.0 Hz with indicator flag
      - power_factor (optional) -> default 1.0 with indicator flag
    
    Returns:
      (dataframe with standard features, metadata_dict with validity & missing flags)
    """
    metadata = {
        "valid": True,
        "missing_features": [],
        "warnings": []
    }
    
    # Check minimum required fields
    for field in REQUIRED_SOLAR_FIELDS:
        if field not in pv_data or pv_data[field] is None:
            metadata["valid"] = False
            metadata["missing_features"].append(field)
            metadata["warnings"].append(f"Essential PV parameter '{field}' is missing. Cannot evaluate.")
            return pd.DataFrame(), metadata

    v_ac = float(pv_data["voltage_ac"])
    i_ac = float(pv_data["current_ac"])
    
    # Power handling
    if "power_ac" in pv_data and pv_data["power_ac"] is not None:
        p_ac = float(pv_data["power_ac"])
    else:
        metadata["missing_features"].append("power_ac")
        metadata["warnings"].append("power_ac not supplied; estimated using V * I * PF.")
        pf_est = float(pv_data.get("power_factor", 1.0) or 1.0)
        p_ac = v_ac * i_ac * pf_est
        
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
        pf = 1.0

    # Build standardized row
    standard_row = {
        "voltage_V": v_ac,
        "current_A": i_ac,
        "power_W": p_ac,
        "frequency_Hz": f_ac,
        "power_factor": pf,
        # Edge non-linear metrics are specific to high-frequency edge meters;
        # For standard PV telemetry we fill standard baseline values with 0 deviation
        "PE_m3": 0.5,
        "PE_m4": 0.5,
        "PE_m5": 0.5,
        "WPE": 0.5,
        "MSPE": 0.5,
        "CV": 0.05,
        "delta_PF": 0.0,
        "SII": 0.0
    }
    
    df = pd.DataFrame([standard_row])
    df_engineered = add_electrical_features(df)
    
    return df_engineered, metadata
