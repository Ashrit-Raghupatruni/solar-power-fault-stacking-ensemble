"""
Comprehensive, Imbalance-Aware Evaluation Module.
Coupled with MODEL_REGISTRY for guaranteed model discovery and dimension alignment.
Computes:
1. Multiclass Metrics: Balanced Accuracy, Raw Accuracy, Macro-F1, Precision, Recall, MCC, Cohen Kappa
2. Anomaly-Only Metrics: Accuracy and F1 strictly evaluated on anomalies (excluding normal baseline)
3. Severity Metrics: False Negative Rate (FNR) on critical faults
4. Binary Anomaly Detection: PR-AUC, ROC-AUC, Balanced Accuracy, F1
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, precision_recall_curve, auc, roc_auc_score
)
from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    FEATURE_COLS, CATEGORY_LABELS, SEVERITY_LABELS
)
from src.model_registry import MODEL_REGISTRY
from src.train import align_probabilities

def evaluate_all_models(target_col: str, model_subfolder: str) -> pd.DataFrame:
    """Evaluates all registered models and ensembles using imbalance-aware metrics."""
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    
    save_dir = MODELS_DIR / model_subfolder
    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    y_test = test_df[target_col].values.astype(np.int64)
    
    labels = CATEGORY_LABELS if "category" in target_col else SEVERITY_LABELS
    total_classes = len(labels)
    
    loaded_models = {}
    aligned_base_probs = []
    
    # Load all models using the canonical MODEL_REGISTRY
    for m_key, m_info in MODEL_REGISTRY.items():
        m_path = save_dir / m_info["filename"]
        if m_path.exists():
            model = joblib.load(m_path)
            loaded_models[m_key] = model
            needs_scaling = m_info["needs_scaling"]
            raw_p = model.predict_proba(X_test_scaled if needs_scaling else X_test_raw)
            aligned_p = align_probabilities(raw_p, model.classes_, total_classes)
            aligned_base_probs.append(aligned_p)
            print(f"Loaded {m_info['name']} from {m_info['filename']}")
        else:
            print(f"Warning: {m_info['filename']} not found in {save_dir}!")

    if not loaded_models:
        print(f"No trained models found in {save_dir}!")
        return pd.DataFrame()

    results = []
    preds_dict = {}
    
    # 1. Base Models
    for i, (m_key, m_info) in enumerate(MODEL_REGISTRY.items()):
        if m_key in loaded_models:
            probs = aligned_base_probs[i]
            preds = np.argmax(probs, axis=1)
            preds_dict[m_key] = preds
            results.append({
                "Model": m_info["name"],
                "Raw Accuracy": accuracy_score(y_test, preds),
                "Balanced Accuracy": balanced_accuracy_score(y_test, preds),
                "Macro-F1": f1_score(y_test, preds, average="macro", zero_division=0),
                "Macro-Precision": precision_score(y_test, preds, average="macro", zero_division=0),
                "Macro-Recall": recall_score(y_test, preds, average="macro", zero_division=0),
                "MCC": matthews_corrcoef(y_test, preds),
                "Cohen Kappa": cohen_kappa_score(y_test, preds)
            })

    # 2. Soft Voting Ensemble
    if len(aligned_base_probs) > 1:
        soft_vote_probs = np.mean(aligned_base_probs, axis=0)
        soft_vote_preds = np.argmax(soft_vote_probs, axis=1)
        preds_dict["soft_voting"] = soft_vote_preds
        results.append({
            "Model": "Soft Voting Ensemble",
            "Raw Accuracy": accuracy_score(y_test, soft_vote_preds),
            "Balanced Accuracy": balanced_accuracy_score(y_test, soft_vote_preds),
            "Macro-F1": f1_score(y_test, soft_vote_preds, average="macro", zero_division=0),
            "Macro-Precision": precision_score(y_test, soft_vote_preds, average="macro", zero_division=0),
            "Macro-Recall": recall_score(y_test, soft_vote_preds, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, soft_vote_preds),
            "Cohen Kappa": cohen_kappa_score(y_test, soft_vote_preds)
        })

    # 3. Stacking Classifier
    meta_path = save_dir / "stacking_meta_learner.pkl"
    if meta_path.exists():
        meta_learner = joblib.load(meta_path)
        meta_X_test = np.hstack(aligned_base_probs)
        stack_preds = meta_learner.predict(meta_X_test)
        preds_dict["stacking_ensemble"] = stack_preds
        results.append({
            "Model": "Stacking Classifier (Ensemble)",
            "Raw Accuracy": accuracy_score(y_test, stack_preds),
            "Balanced Accuracy": balanced_accuracy_score(y_test, stack_preds),
            "Macro-F1": f1_score(y_test, stack_preds, average="macro", zero_division=0),
            "Macro-Precision": precision_score(y_test, stack_preds, average="macro", zero_division=0),
            "Macro-Recall": recall_score(y_test, stack_preds, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, stack_preds),
            "Cohen Kappa": cohen_kappa_score(y_test, stack_preds)
        })

    results_df = pd.DataFrame(results).sort_values(by="Macro-F1", ascending=False)
    metrics_file = METRICS_DIR / f"{model_subfolder}_comparison.csv"
    results_df.to_csv(metrics_file, index=False)
    print(f"\nSaved evaluation metrics to {metrics_file}")

    # 4. Anomaly-Only Subset Performance
    anom_mask = y_test != 0
    if anom_mask.sum() > 0:
        anom_results = []
        for name, preds in preds_dict.items():
            anom_y = y_test[anom_mask]
            anom_p = preds[anom_mask]
            anom_results.append({
                "Model": name.replace("_", " ").title(),
                "Anomaly-Only Accuracy": accuracy_score(anom_y, anom_p),
                "Anomaly-Only Macro-F1": f1_score(anom_y, anom_p, average="macro", zero_division=0),
                "Anomaly-Only Balanced Acc": balanced_accuracy_score(anom_y, anom_p)
            })
        anom_df = pd.DataFrame(anom_results).sort_values(by="Anomaly-Only Macro-F1", ascending=False)
        anom_file = METRICS_DIR / f"{model_subfolder}_anomalies_only_comparison.csv"
        anom_df.to_csv(anom_file, index=False)
        print(f"Saved anomaly-only metrics to {anom_file}")

    # 5. False Negative Rate on Critical Faults
    if "severity" in target_col and "critical" in labels:
        crit_idx = labels.index("critical")
        print("\n--- Critical Fault False Negative Rate (FNR) ---")
        for name, preds in preds_dict.items():
            if (y_test == crit_idx).sum() > 0:
                fn = ((y_test == crit_idx) & (preds != crit_idx)).sum()
                tp = ((y_test == crit_idx) & (preds == crit_idx)).sum()
                fnr = (fn / (fn + tp)) * 100.0
                print(f"  {name:25s} | Critical FNR: {fnr:.2f}% (Captured: {100-fnr:.2f}%)")

    # Save Confusion Matrix Figure
    top_key = "stacking_ensemble" if "stacking_ensemble" in preds_dict else "soft_voting"
    cm = confusion_matrix(y_test, preds_dict[top_key], normalize="true")
    plt.figure(figsize=(10, 8))
    present_labels = [labels[i] for i in sorted(np.unique(y_test))]
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=present_labels, yticklabels=present_labels)
    plt.title(f"Normalized Confusion Matrix ({top_key.replace('_', ' ').title()})")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"{model_subfolder}_stacking_confusion_matrix.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return results_df

def evaluate_binary_anomaly_detection():
    """Evaluates binary anomaly detection: Normal (0) vs Anomaly (1)."""
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    cat_dir = MODELS_DIR / "fault_category"
    
    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    y_binary = (test_df["fault_category"] != "normal").astype(int).values
    
    print("\n" + "="*70 + "\nEvaluating Binary Anomaly Detection (Normal vs Anomaly)\n" + "="*70)
    
    all_probs = []
    for m_key, m_info in MODEL_REGISTRY.items():
        p = cat_dir / m_info["filename"]
        if p.exists():
            m = joblib.load(p)
            needs_scaling = m_info["needs_scaling"]
            raw_p = m.predict_proba(X_test_scaled if needs_scaling else X_test_raw)
            aligned_p = align_probabilities(raw_p, m.classes_, 9)
            all_probs.append(aligned_p)
            
    if all_probs:
        avg_probs = np.mean(all_probs, axis=0)
        p_anomaly = 1.0 - avg_probs[:, 0]
        preds_binary = (p_anomaly >= 0.5).astype(int)
        
        prec, rec, _ = precision_recall_curve(y_binary, p_anomaly)
        pr_auc = auc(rec, prec)
        roc_auc = roc_auc_score(y_binary, p_anomaly)
        b_acc = balanced_accuracy_score(y_binary, preds_binary)
        f1 = f1_score(y_binary, preds_binary)
        mcc = matthews_corrcoef(y_binary, preds_binary)
        
        bin_df = pd.DataFrame([{
            "Task": "Binary Anomaly Detection (Ensemble)",
            "PR-AUC": round(pr_auc, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Balanced Accuracy": round(b_acc * 100, 2),
            "F1-Score": round(f1 * 100, 2),
            "MCC": round(mcc, 4)
        }])
        
        bin_file = METRICS_DIR / "binary_anomaly_detection_benchmark.csv"
        bin_df.to_csv(bin_file, index=False)
        print(f"Saved binary anomaly detection metrics to {bin_file}")
        print(bin_df.T)

if __name__ == "__main__":
    evaluate_all_models("target_category_idx", "fault_category")
    evaluate_all_models("target_severity_idx", "fault_severity")
    evaluate_binary_anomaly_detection()
