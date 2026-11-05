"""
Joint End-to-End Cascaded Evaluator.
Evaluates:
  1. Stage 1: Binary Anomaly Filter (Normal vs Anomaly)
  2. Stage 2A: Fault Category Classification (8 fault classes)
  3. Stage 2B: Fault Severity Classification (5 fault grades)
  4. Cascaded End-to-End System: Propagates Stage 1 filtering to Stage 2.
     (When Stage 1 predicts Normal, category defaults to 'normal' and severity to 'none').
Generates full confusion matrices, Macro-F1, Balanced Accuracy, and critical False Negative Rate (FNR).
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix
)
from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    FEATURE_COLS, STAGE2_FAULT_CATEGORIES, STAGE2_FAULT_SEVERITIES
)
from src.feature_engineering import add_electrical_features
from src.model_registry import MODEL_REGISTRY
from src.train import align_probabilities

def ensure_features_present(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        df = add_electrical_features(df)
    return df

def load_two_stage_pipeline():
    """Loads all components required for end-to-end inference."""
    scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    stage1_model = joblib.load(MODELS_DIR / "stage1_binary" / "binary_detector.pkl")
    stage1_meta = joblib.load(MODELS_DIR / "stage1_binary" / "meta.pkl")
    threshold = stage1_meta.get("threshold", 0.35)
    
    # Load Stage 2 models
    cat_models = {}
    cat_dir = MODELS_DIR / "stage2_fault_category"
    for m_key, m_info in MODEL_REGISTRY.items():
        p = cat_dir / m_info["filename"]
        if p.exists():
            cat_models[m_key] = joblib.load(p)
    cat_meta = joblib.load(cat_dir / "residual_stacking_meta_learner.pkl") if (cat_dir / "residual_stacking_meta_learner.pkl").exists() else None

    sev_models = {}
    sev_dir = MODELS_DIR / "stage2_fault_severity"
    for m_key, m_info in MODEL_REGISTRY.items():
        p = sev_dir / m_info["filename"]
        if p.exists():
            sev_models[m_key] = joblib.load(p)
    sev_meta = joblib.load(sev_dir / "residual_stacking_meta_learner.pkl") if (sev_dir / "residual_stacking_meta_learner.pkl").exists() else None

    return {
        "scaler": scaler,
        "stage1_model": stage1_model,
        "threshold": threshold,
        "cat_models": cat_models,
        "cat_meta": cat_meta,
        "sev_models": sev_models,
        "sev_meta": sev_meta
    }

def evaluate_cascaded_system():
    """Evaluates the full end-to-end two-stage architecture on test.csv."""
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if not test_path.exists():
        print(f"Error: {test_path} not found. Run preprocessing first.")
        return
        
    print("\n" + "="*80)
    print("EVALUATING TWO-STAGE CASCADED SYSTEM ON HELD-OUT TEST PARTITION")
    print("="*80)
    
    test_df = ensure_features_present(pd.read_csv(test_path))
    pipe = load_two_stage_pipeline()
    scaler = pipe["scaler"]
    
    X_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    
    # Ground truth
    y_true_binary = (test_df["anomaly"] == 1).astype(int).values
    y_true_cat_raw = test_df["fault_category"].values
    y_true_sev_raw = test_df["severity"].values
    
    # ----------------------------------------------------
    # Step 1: Stage 1 Binary Inference
    # ----------------------------------------------------
    print("\n[Step 1] Running Stage 1 Binary Anomaly Filter...")
    stage1_probs = pipe["stage1_model"].predict_proba(X_raw)[:, 1]
    stage1_preds = (stage1_probs >= pipe["threshold"]).astype(int)
    
    s1_rec = recall_score(y_true_binary, stage1_preds)
    s1_prec = precision_score(y_true_binary, stage1_preds)
    s1_f1 = f1_score(y_true_binary, stage1_preds)
    s1_fnr = 1.0 - s1_rec
    
    print(f"  Stage 1 Decision Threshold: {pipe['threshold']:.2f}")
    print(f"  Anomaly Recall (Sensitivity): {s1_rec*100:.2f}%")
    print(f"  Anomaly Precision:           {s1_prec*100:.2f}%")
    print(f"  Anomaly False Negative Rate: {s1_fnr*100:.2f}%")
    print(f"  Stage 1 F1-Score:             {s1_f1*100:.2f}%")
    
    # ----------------------------------------------------
    # Step 2: Stage 2 Inference for Flagged Records
    # ----------------------------------------------------
    anom_indices = np.where(stage1_preds == 1)[0]
    print(f"\n[Step 2] Stage 1 Flagged {len(anom_indices):,} samples ({len(anom_indices)/len(test_df)*100:.2f}%) for Stage 2 Inspection.")
    
    # Cascaded output containers defaulting to nominal
    final_pred_cat = np.array(["normal"] * len(test_df), dtype=object)
    final_pred_sev = np.array(["none"] * len(test_df), dtype=object)
    
    if len(anom_indices) > 0 and len(pipe["cat_models"]) > 0:
        X_anom_raw = X_raw[anom_indices]
        X_anom_scaled = X_scaled[anom_indices]
        
        # Category residual stacking / soft voting
        cat_probs_list = []
        for m_key, m_info in MODEL_REGISTRY.items():
            if m_key in pipe["cat_models"]:
                m = pipe["cat_models"][m_key]
                X_in = X_anom_scaled if m_info["needs_scaling"] else X_anom_raw
                raw_p = m.predict_proba(X_in)
                aligned_p = align_probabilities(raw_p, m.classes_, len(STAGE2_FAULT_CATEGORIES))
                cat_probs_list.append(aligned_p)
                
        soft_vote_cat = np.mean(cat_probs_list, axis=0)
        
        if pipe["cat_meta"] is not None:
            # Construct residual stacking feature block
            stacked_feats = np.hstack(cat_probs_list + [soft_vote_cat])
            cat_preds_idx = pipe["cat_meta"].predict(stacked_feats)
        else:
            cat_preds_idx = np.argmax(soft_vote_cat, axis=1)
            
        final_pred_cat[anom_indices] = [STAGE2_FAULT_CATEGORIES[idx] for idx in cat_preds_idx]

    if len(anom_indices) > 0 and len(pipe["sev_models"]) > 0:
        # Severity residual stacking / soft voting
        sev_probs_list = []
        for m_key, m_info in MODEL_REGISTRY.items():
            if m_key in pipe["sev_models"]:
                m = pipe["sev_models"][m_key]
                X_in = X_anom_scaled if m_info["needs_scaling"] else X_anom_raw
                raw_p = m.predict_proba(X_in)
                aligned_p = align_probabilities(raw_p, m.classes_, len(STAGE2_FAULT_SEVERITIES))
                sev_probs_list.append(aligned_p)
                
        soft_vote_sev = np.mean(sev_probs_list, axis=0)
        
        if pipe["sev_meta"] is not None:
            stacked_feats_sev = np.hstack(sev_probs_list + [soft_vote_sev])
            sev_preds_idx = pipe["sev_meta"].predict(stacked_feats_sev)
        else:
            sev_preds_idx = np.argmax(soft_vote_sev, axis=1)
            
        final_pred_sev[anom_indices] = [STAGE2_FAULT_SEVERITIES[idx] for idx in sev_preds_idx]

    # ----------------------------------------------------
    # Step 3: Global Holistic Cascaded Metrics
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("GLOBAL CASCADED END-TO-END PERFORMANCE")
    print("="*80)
    
    # Fault Category (Overall across all 348,899 test samples)
    all_cat_labels = ["normal"] + STAGE2_FAULT_CATEGORIES
    cat_acc = accuracy_score(y_true_cat_raw, final_pred_cat)
    cat_macro_f1 = f1_score(y_true_cat_raw, final_pred_cat, average="macro", zero_division=0)
    cat_b_acc = balanced_accuracy_score(y_true_cat_raw, final_pred_cat)
    
    print(f"Cascaded Fault Category Classification:")
    print(f"  Raw Overall Accuracy:  {cat_acc*100:.2f}%")
    print(f"  Macro-F1 Score:        {cat_macro_f1*100:.2f}%")
    print(f"  Balanced Accuracy:     {cat_b_acc*100:.2f}%")
    
    # Fault Severity (Overall)
    all_sev_labels = ["none"] + STAGE2_FAULT_SEVERITIES
    sev_acc = accuracy_score(y_true_sev_raw, final_pred_sev)
    sev_macro_f1 = f1_score(y_true_sev_raw, final_pred_sev, average="macro", zero_division=0)
    sev_b_acc = balanced_accuracy_score(y_true_sev_raw, final_pred_sev)
    
    # Critical fault safety metric: strictly requires prediction == 'critical'
    critical_mask = (y_true_sev_raw == "critical")
    if np.sum(critical_mask) > 0:
        crit_tp = np.sum(final_pred_sev[critical_mask] == "critical")
        crit_fn = np.sum(final_pred_sev[critical_mask] != "critical")
        crit_recall = crit_tp / np.sum(critical_mask)
        crit_fnr = (1.0 - crit_recall) * 100.0
    else:
        crit_fnr = 0.0

    print(f"\nCascaded Fault Severity Estimation:")
    print(f"  Raw Overall Accuracy:        {sev_acc*100:.2f}%")
    print(f"  Macro-F1 Score:              {sev_macro_f1*100:.2f}%")
    print(f"  Balanced Accuracy:           {sev_b_acc*100:.2f}%")
    print(f"  Critical Fault Miss Rate (Strict FNR): {crit_fnr:.2f}%")

    # Generate and export normalized confusion matrices
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Category Confusion Matrix
    cm_cat = confusion_matrix(y_true_cat_raw, final_pred_cat, labels=all_cat_labels, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_cat, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=all_cat_labels, yticklabels=all_cat_labels)
    plt.title("Cascaded End-to-End Fault Category Normalized Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cascaded_category_confusion_matrix.png", dpi=200)
    plt.close()

    # 2. Severity Confusion Matrix
    cm_sev = confusion_matrix(y_true_sev_raw, final_pred_sev, labels=all_sev_labels, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_sev, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=all_sev_labels, yticklabels=all_sev_labels)
    plt.title("Cascaded End-to-End Fault Severity Normalized Confusion Matrix")
    plt.xlabel("Predicted Severity")
    plt.ylabel("True Severity")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cascaded_severity_confusion_matrix.png", dpi=200)
    plt.close()
    print(f"Saved normalized confusion matrices to {FIGURES_DIR}")
    
    # Save metrics summary
    metrics_summary = pd.DataFrame([{
        "architecture": "Two-Stage Cascaded",
        "stage1_threshold": pipe["threshold"],
        "stage1_recall": round(s1_rec * 100, 2),
        "stage1_fnr": round(s1_fnr * 100, 2),
        "cat_overall_acc": round(cat_acc * 100, 2),
        "cat_macro_f1": round(cat_macro_f1 * 100, 2),
        "cat_balanced_acc": round(cat_b_acc * 100, 2),
        "sev_overall_acc": round(sev_acc * 100, 2),
        "sev_macro_f1": round(sev_macro_f1 * 100, 2),
        "sev_balanced_acc": round(sev_b_acc * 100, 2),
        "critical_fnr": round(crit_fnr, 2)
    }])
    
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_summary.to_csv(METRICS_DIR / "cascaded_two_stage_evaluation.csv", index=False)
    print(f"\nSaved cascaded evaluation report to {METRICS_DIR / 'cascaded_two_stage_evaluation.csv'}")

if __name__ == "__main__":
    evaluate_cascaded_system()
