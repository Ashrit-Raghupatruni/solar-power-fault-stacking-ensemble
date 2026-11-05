"""
FastAPI Real-Time Model Inference Service.
Serves:
  1. POST /predict/two-stage: Live two-stage inference with physics-guided attribution.
  2. GET /stream/step: 1 Hz telemetry tick simulator feeding real ML pipeline.
  3. GET /health: Model discovery and health check status.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any, List

from src.config import (
    MODELS_DIR, FEATURE_COLS,
    STAGE2_FAULT_CATEGORIES, STAGE2_FAULT_SEVERITIES
)
from src.feature_engineering import add_electrical_features
from src.explainability import explain_prediction_heuristic
from src.model_registry import MODEL_REGISTRY
from src.train import align_probabilities

app = FastAPI(title="Solar & Power System Fault Detection API", version="2.0")

# Enable CORS for browser dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
pipeline_cache = {}

def get_pipeline():
    if "stage1_model" not in pipeline_cache:
        scaler_p = MODELS_DIR / "preprocessors" / "scaler.pkl"
        s1_p = MODELS_DIR / "stage1_binary" / "binary_detector.pkl"
        meta_p = MODELS_DIR / "stage1_binary" / "meta.pkl"
        
        pipeline_cache["scaler"] = joblib.load(scaler_p) if scaler_p.exists() else None
        pipeline_cache["stage1_model"] = joblib.load(s1_p) if s1_p.exists() else None
        pipeline_cache["threshold"] = joblib.load(meta_p).get("threshold", 0.35) if meta_p.exists() else 0.35

        # Stage 2 Category models
        cat_dir = MODELS_DIR / "stage2_fault_category"
        pipeline_cache["cat_models"] = {}
        if cat_dir.exists():
            for m_key, m_info in MODEL_REGISTRY.items():
                p = cat_dir / m_info["filename"]
                if p.exists():
                    pipeline_cache["cat_models"][m_key] = joblib.load(p)
            meta_p = cat_dir / "residual_stacking_meta_learner.pkl"
            pipeline_cache["cat_meta"] = joblib.load(meta_p) if meta_p.exists() else None
        else:
            pipeline_cache["cat_meta"] = None

        # Stage 2 Severity models
        sev_dir = MODELS_DIR / "stage2_fault_severity"
        pipeline_cache["sev_models"] = {}
        if sev_dir.exists():
            for m_key, m_info in MODEL_REGISTRY.items():
                p = sev_dir / m_info["filename"]
                if p.exists():
                    pipeline_cache["sev_models"][m_key] = joblib.load(p)
            meta_p = sev_dir / "residual_stacking_meta_learner.pkl"
            pipeline_cache["sev_meta"] = joblib.load(meta_p) if meta_p.exists() else None
        else:
            pipeline_cache["sev_meta"] = None

    return pipeline_cache

class TelemetryPayload(BaseModel):
    voltage_V: float
    current_A: float
    frequency_Hz: float
    power_W: float
    power_factor: float
    thd_V: float
    thd_I: float
    phase_angle_deg: float
    unbalance_V: float
    wifi_rssi_dBm: Optional[float] = -65.0
    station_id: Optional[int] = 1

# Stateful stream buffer for 1 Hz simulator
stream_state = {
    "voltage_V": 220.0,
    "current_A": 5.0,
    "power_W": 1100.0,
    "fault_mode": "normal"
}

@app.get("/health")
def health():
    pipe = get_pipeline()
    return {
        "status": "healthy",
        "has_stage1": pipe["stage1_model"] is not None,
        "has_scaler": pipe["scaler"] is not None,
        "num_cat_models": len(pipe.get("cat_models", {})),
        "num_sev_models": len(pipe.get("sev_models", {})),
        "threshold": pipe.get("threshold", 0.35)
    }

@app.post("/predict/two-stage")
def predict_two_stage(payload: TelemetryPayload):
    pipe = get_pipeline()
    
    # 1. Feature Engineering
    raw_dict = payload.dict()
    df_single = pd.DataFrame([raw_dict])
    df_feat = add_electrical_features(df_single, previous_state=stream_state)
    
    # Fill any missing feature columns with default zeros
    for c in FEATURE_COLS:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
            
    X_raw = df_feat[FEATURE_COLS].values.astype(np.float32)
    
    # Physical explainability triggers
    shap_triggers = explain_prediction_heuristic(raw_dict)
    
    # Check if models are available; if not, return fallback diagnosis
    if pipe["stage1_model"] is None:
        # Heuristic fallback if models not loaded
        is_anom = payload.voltage_V < 20.0 or payload.voltage_V < 198.0 or payload.current_A > 15.0 or payload.power_factor < 0.70
        return {
            "source": "heuristic_fallback",
            "is_anomaly": is_anom,
            "anomaly_probability": 0.85 if is_anom else 0.05,
            "fault_category": "Voltage Sag / Dip" if payload.voltage_V < 198.0 else ("Outage / Blackout" if payload.voltage_V < 20.0 else "Normal Baseline"),
            "fault_severity": "Critical" if payload.voltage_V < 20.0 else ("Medium" if is_anom else "None"),
            "shap_triggers": shap_triggers
        }
        
    # Real Model Inference
    # Stage 1: Binary Filter
    s1_prob = float(pipe["stage1_model"].predict_proba(X_raw)[0, 1])
    is_anomaly = s1_prob >= pipe["threshold"]
    
    pred_category = "Normal Operation"
    pred_severity = "None"
    cat_conf = 1.0 - s1_prob
    sev_conf = 1.0 - s1_prob
    
    if is_anomaly:
        X_scaled = pipe["scaler"].transform(X_raw).astype(np.float32) if pipe["scaler"] else X_raw
        
        # Stage 2A: Fault Category
        if len(pipe["cat_models"]) > 0:
            cat_probs = []
            for m_key, m_info in MODEL_REGISTRY.items():
                if m_key in pipe["cat_models"]:
                    m = pipe["cat_models"][m_key]
                    X_in = X_scaled if m_info["needs_scaling"] else X_raw
                    raw_p = m.predict_proba(X_in)
                    aligned_p = align_probabilities(raw_p, m.classes_, len(STAGE2_FAULT_CATEGORIES))
                    cat_probs.append(aligned_p)
            soft_vote = np.mean(cat_probs, axis=0)
            if pipe["cat_meta"] is not None:
                meta_feats = np.hstack(cat_probs + [soft_vote])
                cat_idx = int(pipe["cat_meta"].predict(meta_feats)[0])
            else:
                cat_idx = int(np.argmax(soft_vote[0]))
            pred_category = STAGE2_FAULT_CATEGORIES[cat_idx].replace("_", " ").title()
            cat_conf = float(np.max(soft_vote[0]))
        else:
            pred_category = "Electrical Anomaly"
            
        # Stage 2B: Fault Severity
        if len(pipe["sev_models"]) > 0:
            sev_probs = []
            for m_key, m_info in MODEL_REGISTRY.items():
                if m_key in pipe["sev_models"]:
                    m = pipe["sev_models"][m_key]
                    X_in = X_scaled if m_info["needs_scaling"] else X_raw
                    raw_p = m.predict_proba(X_in)
                    aligned_p = align_probabilities(raw_p, m.classes_, len(STAGE2_FAULT_SEVERITIES))
                    sev_probs.append(aligned_p)
            soft_vote_s = np.mean(sev_probs, axis=0)
            if pipe["sev_meta"] is not None:
                meta_feats_s = np.hstack(sev_probs + [soft_vote_s])
                sev_idx = int(pipe["sev_meta"].predict(meta_feats_s)[0])
            else:
                sev_idx = int(np.argmax(soft_vote_s[0]))
            pred_severity = STAGE2_FAULT_SEVERITIES[sev_idx].title()
            sev_conf = float(np.max(soft_vote_s[0]))
        else:
            pred_severity = "High"

    return {
        "source": "two_stage_ensemble",
        "is_anomaly": bool(is_anomaly),
        "stage1_probability": round(s1_prob, 4),
        "stage1_threshold": float(pipe["threshold"]),
        "fault_category": pred_category,
        "category_confidence": round(cat_conf, 4),
        "fault_severity": pred_severity,
        "severity_confidence": round(sev_conf, 4),
        "shap_triggers": shap_triggers
    }

@app.get("/stream/inject")
def inject_stream_fault(fault_type: str):
    """Injects a simulated fault into the 1 Hz stream generator."""
    stream_state["fault_mode"] = fault_type
    return {"status": "injected", "fault_mode": fault_type}

@app.get("/stream/step")
def stream_step():
    """Generates the next 1 Hz continuous telemetry reading and evaluates it through ML."""
    mode = stream_state.get("fault_mode", "normal")
    
    # Base nominal fluctuation
    v = 220.0 + np.random.uniform(-1.0, 1.0)
    i = 5.0 + np.random.uniform(-0.1, 0.1)
    f = 50.0 + np.random.uniform(-0.02, 0.02)
    pf = 0.95 + np.random.uniform(-0.01, 0.01)
    
    # Fault overrides
    if mode == "sag":
        v = 184.0 + np.random.uniform(-2.0, 2.0)
    elif mode == "overload":
        i = 19.5 + np.random.uniform(-0.5, 0.5)
        v = 208.0 + np.random.uniform(-2.0, 1.0)
    elif mode == "outage":
        v = 0.0
        i = 0.0
        pf = 0.0
        
    p = v * i * pf
    
    payload = TelemetryPayload(
        voltage_V=round(v, 2),
        current_A=round(i, 2),
        frequency_Hz=round(f, 2),
        power_W=round(p, 2),
        power_factor=round(pf, 3),
        thd_V=1.2,
        thd_I=2.1,
        phase_angle_deg=18.0,
        unbalance_V=0.5
    )
    
    # Update state buffer
    stream_state["voltage_V"] = v
    stream_state["current_A"] = i
    stream_state["power_W"] = p
    
    ml_result = predict_two_stage(payload)
    return {
        "telemetry": payload.dict(),
        "diagnosis": ml_result
    }
