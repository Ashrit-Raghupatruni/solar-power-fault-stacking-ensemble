"""
FastAPI Real-Time Model Inference Service for PowerGuard AI.
Serves:
  1. POST /predict/two-stage: Live two-stage inference (16 operational features) with physics-guided attribution.
  2. POST /predict/solar-pv: Solar Inverter (AhoyDTU / OpenDTU) ingestion, DC string analytics & AC diagnosis.
  3. GET /stream/step: 1 Hz telemetry tick simulator feeding real ML pipeline.
  4. POST /stream/inject: Controls fault injection state for simulator.
  5. GET /health: Model discovery, hashes, and health check status.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any, List

from src.config import (
    MODELS_DIR, OPERATIONAL_FEATURE_COLS,
    STAGE2_FAULT_CATEGORIES, STAGE2_FAULT_SEVERITIES
)
from src.feature_engineering import add_electrical_features
from src.explainability import explain_prediction_heuristic
from src.model_registry import MODEL_REGISTRY
from src.train import align_probabilities
from src.pv_adapter import adapt_pv_telemetry

app = FastAPI(title="PowerGuard AI - Electrical & Solar Intelligence API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_cache = {}

def get_pipeline():
    if "stage1_model" not in pipeline_cache:
        scaler_p = MODELS_DIR / "preprocessors" / "scaler.pkl"
        s1_p = MODELS_DIR / "stage1_binary" / "binary_detector.pkl"
        meta_p = MODELS_DIR / "stage1_binary" / "meta.pkl"
        
        pipeline_cache["scaler"] = joblib.load(scaler_p) if scaler_p.exists() else None
        pipeline_cache["stage1_model"] = joblib.load(s1_p) if s1_p.exists() else None
        pipeline_cache["threshold"] = joblib.load(meta_p).get("threshold", 0.125) if meta_p.exists() else 0.125

        # Stage 2 Category models
        cat_dir = MODELS_DIR / "stage2_fault_category"
        pipeline_cache["cat_models"] = {}
        if cat_dir.exists():
            for m_key, m_info in MODEL_REGISTRY.items():
                p = cat_dir / m_info["filename"]
                if p.exists():
                    pipeline_cache["cat_models"][m_key] = joblib.load(p)
            meta_p = cat_dir / "stacking_meta_learner.pkl"
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
            meta_p = sev_dir / "stacking_meta_learner.pkl"
            pipeline_cache["sev_meta"] = joblib.load(meta_p) if meta_p.exists() else None
        else:
            pipeline_cache["sev_meta"] = None

    return pipeline_cache

class TelemetryPayload(BaseModel):
    voltage_V: float = Field(220.0, ge=0.0, le=500.0, description="AC Voltage in Volts")
    current_A: float = Field(5.0, ge=0.0, le=200.0, description="AC Current in Amperes")
    frequency_Hz: float = Field(50.0, ge=35.0, le=75.0, description="AC Frequency in Hertz")
    power_W: Optional[float] = Field(None, description="Active Power in Watts")
    power_factor: Optional[float] = Field(0.98, ge=-1.0, le=1.0, description="Power factor")

class SolarPVPayload(BaseModel):
    voltage_ac: float = Field(..., ge=0.0, le=500.0)
    current_ac: float = Field(..., ge=0.0, le=100.0)
    power_ac: Optional[float] = None
    frequency_ac: Optional[float] = 50.0
    power_factor: Optional[float] = 0.98
    ch1_v: Optional[float] = None
    ch1_i: Optional[float] = None
    ch2_v: Optional[float] = None
    ch2_i: Optional[float] = None
    ch3_v: Optional[float] = None
    ch3_i: Optional[float] = None
    ch4_v: Optional[float] = None
    ch4_i: Optional[float] = None

class StreamInjectPayload(BaseModel):
    fault_type: str = Field("normal", description="Fault mode: normal, sag, overload, outage")

# Stateful circular stream buffer for 1 Hz simulator
stream_state = {
    "voltage_V": 220.0,
    "current_A": 5.0,
    "power_W": 1100.0,
    "fault_mode": "normal",
    "step_count": 0
}

@app.get("/health")
def health():
    pipe = get_pipeline()
    return {
        "status": "healthy",
        "service": "PowerGuard AI Inference Service",
        "operational_features_count": len(OPERATIONAL_FEATURE_COLS),
        "stage1_loaded": pipe["stage1_model"] is not None,
        "stage1_threshold": pipe.get("threshold", 0.125),
        "stage2_cat_models_count": len(pipe.get("cat_models", {})),
        "stage2_sev_models_count": len(pipe.get("sev_models", {})),
        "stage2_cat_meta_loaded": pipe["cat_meta"] is not None,
        "stage2_sev_meta_loaded": pipe["sev_meta"] is not None
    }

@app.get("/")
def root():
    return {
        "name": "PowerGuard AI",
        "description": "Solar & Electrical Fault Detection Engine",
        "docs_url": "/docs",
        "health_url": "/health"
    }

def _run_cascaded_inference(df_feat: pd.DataFrame, raw_dict: dict) -> dict:
    pipe = get_pipeline()
    if pipe["stage1_model"] is None:
        raise HTTPException(status_code=503, detail="ML models are not loaded on server.")
        
    X_raw = df_feat[OPERATIONAL_FEATURE_COLS].values.astype(np.float32)
    s1_prob = float(pipe["stage1_model"].predict_proba(X_raw)[0, 1])
    threshold = float(pipe["threshold"])
    is_anomaly = s1_prob >= threshold

    pred_category = "Normal Operation"
    pred_severity = "None"
    cat_conf = 1.0 - s1_prob
    sev_conf = 1.0 - s1_prob
    
    cat_probs_dist = {"normal": round(1.0 - s1_prob, 4)}
    sev_probs_dist = {"none": round(1.0 - s1_prob, 4)}

    if is_anomaly:
        X_scaled = pipe["scaler"].transform(X_raw).astype(np.float32) if pipe["scaler"] else X_raw
        
        # Stage 2A: Fault Category Stacking
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
            cat_conf = float(soft_vote[0, cat_idx])
            for i, cat_name in enumerate(STAGE2_FAULT_CATEGORIES):
                cat_probs_dist[cat_name] = round(float(soft_vote[0, i]), 4)
                
        # Stage 2B: Fault Severity Stacking
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
                
            raw_sev = STAGE2_FAULT_SEVERITIES[sev_idx]
            pred_severity = "Medium" if raw_sev == "med" else raw_sev.title()
            sev_conf = float(soft_vote_s[0, sev_idx])
            for i, sev_name in enumerate(STAGE2_FAULT_SEVERITIES):
                d_name = "medium" if sev_name == "med" else sev_name
                sev_probs_dist[d_name] = round(float(soft_vote_s[0, i]), 4)

    shap_triggers = explain_prediction_heuristic(raw_dict)

    return {
        "is_anomaly": bool(is_anomaly),
        "stage1_probability": round(s1_prob, 4),
        "stage1_threshold": threshold,
        "fault_category": pred_category,
        "category_confidence": round(cat_conf, 4),
        "category_probabilities": cat_probs_dist,
        "fault_severity": pred_severity,
        "severity_confidence": round(sev_conf, 4),
        "severity_probabilities": sev_probs_dist,
        "physics_attributions": shap_triggers
    }

@app.post("/predict/two-stage")
def predict_two_stage(payload: TelemetryPayload):
    raw_dict = payload.dict()
    if raw_dict.get("power_W") is None:
        pf_val = raw_dict.get("power_factor", 0.98) or 0.98
        raw_dict["power_W"] = round(raw_dict["voltage_V"] * raw_dict["current_A"] * pf_val, 2)
        
    df_single = pd.DataFrame([raw_dict])
    df_feat = add_electrical_features(df_single, previous_state=stream_state)
    return _run_cascaded_inference(df_feat, raw_dict)

@app.post("/predict/solar-pv")
def predict_solar_pv(payload: SolarPVPayload):
    pv_data = payload.dict()
    df_operational, meta = adapt_pv_telemetry(pv_data)
    
    if not meta["valid"]:
        raise HTTPException(status_code=422, detail=meta["warnings"])
        
    raw_dict = {
        "voltage_V": float(df_operational["voltage_V"].iloc[0]),
        "current_A": float(df_operational["current_A"].iloc[0]),
        "power_W": float(df_operational["power_W"].iloc[0]),
        "frequency_Hz": float(df_operational["frequency_Hz"].iloc[0]),
        "power_factor": float(df_operational["power_factor"].iloc[0])
    }
    
    inference_result = _run_cascaded_inference(df_operational, raw_dict)
    return {
        "pv_dc_analytics": meta["dc_analytics"],
        "ac_grid_diagnosis": inference_result
    }

@app.post("/stream/inject")
def inject_fault(payload: StreamInjectPayload):
    stream_state["fault_mode"] = payload.fault_type
    return {"status": "success", "active_fault_mode": payload.fault_type}

@app.get("/stream/step")
def stream_step():
    stream_state["step_count"] += 1
    mode = stream_state.get("fault_mode", "normal")
    
    # 1. Base grid sinusoidal nominal
    t = stream_state["step_count"]
    v = 220.0 + 1.8 * np.sin(t * 0.1) + np.random.uniform(-0.5, 0.5)
    i = 5.0 + 0.2 * np.cos(t * 0.05) + np.random.uniform(-0.1, 0.1)
    f = 50.0 + np.random.uniform(-0.03, 0.03)
    pf = 0.98 + np.random.uniform(-0.01, 0.01)
    
    # 2. Inject fault dynamics if active
    if mode == "sag":
        v = 175.0 + np.random.uniform(-3.0, 3.0)
    elif mode == "overload":
        i = 24.5 + np.random.uniform(-1.0, 1.0)
        v = 205.0 + np.random.uniform(-2.0, 1.0)
    elif mode == "outage":
        v = 0.0
        i = 0.0
        pf = 0.0
        
    p = round(v * i * pf, 2)
    
    telemetry = {
        "voltage_V": round(v, 2),
        "current_A": round(i, 2),
        "power_W": p,
        "frequency_Hz": round(f, 2),
        "power_factor": round(pf, 3)
    }
    
    df_single = pd.DataFrame([telemetry])
    df_feat = add_electrical_features(df_single, previous_state=stream_state)
    
    # Update rolling state
    stream_state["voltage_V"] = v
    stream_state["current_A"] = i
    stream_state["power_W"] = p
    
    diagnosis = _run_cascaded_inference(df_feat, telemetry)
    
    return {
        "timestamp_tick": stream_state["step_count"],
        "telemetry": telemetry,
        "diagnosis": diagnosis
    }
