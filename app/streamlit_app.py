import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
try:
    import joblib
except ImportError:
    import pickle as joblib

from src.config import (
    MODELS_DIR, FEATURE_COLS, CATEGORY_LABELS, SEVERITY_LABELS,
    IDX_TO_CATEGORY, IDX_TO_SEVERITY, METRICS_DIR
)
from src.feature_engineering import add_electrical_features
from src.pv_adapter import adapt_pv_telemetry

st.set_page_config(
    page_title="Solar & Electrical Power Fault Detection",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Solar & Electrical Power Fault Detection System")
st.markdown("### Stacking Ensemble Learning for IoT Edge & Photovoltaic Inverter Monitoring")

@st.cache_resource
def load_trained_models():
    """Loads actual serialized base models and the stacking meta-learner."""
    cat_dir = MODELS_DIR / "fault_category"
    scaler_path = MODELS_DIR / "preprocessors" / "scaler.pkl"
    
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    
    base_names = [
        "random_forest", "extra_trees", "gradient_boosting",
        "k-nearest_neighbors", "support_vector_machine"
    ]
    base_models = {}
    for name in base_names:
        p = cat_dir / f"{name}.pkl"
        if p.exists():
            base_models[name] = joblib.load(p)
            
    meta_path = cat_dir / "stacking_meta_learner.pkl"
    meta_learner = joblib.load(meta_path) if meta_path.exists() else None
    
    return scaler, base_models, meta_learner

scaler, base_models, meta_learner = load_trained_models()

tabs = st.tabs(["🔍 Real-Time Diagnosis", "☀️ Solar PV Adapter", "📂 Batch CSV Diagnosis", "📊 Model Performance Explorer"])

# --- TAB 1: Real-Time Single Sensor Diagnosis ---
with tabs[0]:
    st.header("Real-Time Telemetry Diagnosis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        v_in = st.number_input("AC Voltage (V)", min_value=0.0, max_value=400.0, value=220.0, step=0.5)
        i_in = st.number_input("AC Current (A)", min_value=0.0, max_value=150.0, value=5.2, step=0.1)
    with col2:
        p_in = st.number_input("Active Power (W)", min_value=0.0, max_value=50000.0, value=1144.0, step=10.0)
        f_in = st.number_input("AC Frequency (Hz)", min_value=40.0, max_value=65.0, value=50.0, step=0.1)
    with col3:
        pf_in = st.slider("Power Factor (cos φ)", min_value=-1.0, max_value=1.0, value=0.98, step=0.01)

    if st.button("Diagnose Electrical State", type="primary"):
        # Construct raw row
        raw_row = {
            "voltage_V": v_in, "current_A": i_in, "power_W": p_in,
            "frequency_Hz": f_in, "power_factor": pf_in,
            "PE_m3": 0.5, "PE_m4": 0.5, "PE_m5": 0.5, "WPE": 0.5, "MSPE": 0.5,
            "CV": 0.05, "delta_PF": 0.0, "SII": 0.0
        }
        df_single = pd.DataFrame([raw_row])
        df_feat = add_electrical_features(df_single)
        X_raw = df_feat[FEATURE_COLS].values
        
        st.subheader("Diagnostic Results")
        
        if scaler and base_models and meta_learner:
            X_scaled = scaler.transform(X_raw)
            base_probs = []
            
            for name, model in base_models.items():
                is_scaled = name in ["k-nearest_neighbors", "support_vector_machine"]
                probs = model.predict_proba(X_scaled if is_scaled else X_raw)
                base_probs.append(probs)
                
            meta_X = np.hstack(base_probs)
            pred_idx = meta_learner.predict(meta_X)[0]
            pred_probs = meta_learner.predict_proba(meta_X)[0]
            confidence = float(np.max(pred_probs)) * 100.0
            
            predicted_category = IDX_TO_CATEGORY.get(pred_idx, f"Class {pred_idx}")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Predicted Fault Category (Stacking)", predicted_category)
            with res_col2:
                st.metric("Meta-Model Confidence", f"{confidence:.1f}%")
                
            # Class probability distribution chart
            prob_df = pd.DataFrame({
                "Category": [IDX_TO_CATEGORY[i] for i in range(len(pred_probs))],
                "Probability": pred_probs
            }).sort_values(by="Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Category"))
        else:
            st.warning("Model files not found in models/fault_category/. Please train the models to enable live inference.")

# --- TAB 2: Solar PV Telemetry Adapter ---
with tabs[1]:
    st.header("☀️ Solar Photovoltaic Inverter Telemetry Adapter")
    st.markdown("Accepts solar inverter and string parameters and transforms them into standardized electrical fault features without fabricating values.")
    
    pv_col1, pv_col2 = st.columns(2)
    with pv_col1:
        pv_v = st.number_input("PV Inverter AC Voltage (V)", value=228.0)
        pv_i = st.number_input("PV Inverter AC Current (A)", value=14.5)
    with pv_col2:
        pv_p = st.number_input("Inverter Output Active Power (W)", value=3250.0)
        pv_pf = st.number_input("Inverter Power Factor", value=0.99)
        
    if st.button("Run Solar PV Telemetry Analysis"):
        adapted_df, meta = adapt_pv_telemetry({
            "voltage_ac": pv_v,
            "current_ac": pv_i,
            "power_ac": pv_p,
            "power_factor": pv_pf
        })
        if not meta["valid"]:
            st.error(f"Cannot process telemetry: {', '.join(meta['warnings'])}")
        else:
            st.write("Engineered Telemetry Feature Vector:", adapted_df[FEATURE_COLS[:8]])
            if meta["warnings"]:
                for w in meta["warnings"]:
                    st.info(f"Notice: {w}")
            
            if scaler and base_models and meta_learner:
                X_pv = adapted_df[FEATURE_COLS].values
                X_pv_scaled = scaler.transform(X_pv)
                base_probs = [m.predict_proba(X_pv_scaled if n in ["k-nearest_neighbors", "support_vector_machine"] else X_pv) for n, m in base_models.items()]
                meta_X = np.hstack(base_probs)
                pv_cat = IDX_TO_CATEGORY.get(meta_learner.predict(meta_X)[0], "Unknown")
                st.success(f"Inverter Diagnosis from Stacking Ensemble: **{pv_cat}**")

# --- TAB 3: Batch CSV Telemetry Diagnosis ---
with tabs[2]:
    st.header("📂 Batch CSV Telemetry Upload & Fault Timeline")
    uploaded_file = st.file_uploader("Upload Time-Series Sensor CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file, nrows=1000)
        st.write(f"Loaded {len(batch_df)} telemetry samples from file:")
        st.dataframe(batch_df.head(10))
        available_cols = [c for c in ["voltage_V", "current_A", "power_W"] if c in batch_df.columns]
        if available_cols:
            st.line_chart(batch_df[available_cols].dropna().head(200))

# --- TAB 4: Model Performance Explorer ---
with tabs[3]:
    st.header("📊 Model Performance Benchmark (Real Evaluation Results)")
    st.markdown("Metrics computed on the 15,946 held-out test records with **zero event leakage**:")
    
    metrics_path = METRICS_DIR / "fault_category_comparison.csv"
    if metrics_path.exists():
        real_metrics_df = pd.read_csv(metrics_path)
        # Format percentage columns
        for c in ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]:
            real_metrics_df[c] = (real_metrics_df[c] * 100.0).round(2).astype(str) + "%"
        st.dataframe(real_metrics_df, use_container_width=True)
    else:
        st.info("Run `python -m src.evaluate` to generate the real benchmark CSV.")
