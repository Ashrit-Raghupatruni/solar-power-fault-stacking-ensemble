"""
Comprehensive Evaluation & Reporting Module.
Computes:
- Accuracy, Precision (Macro), Recall (Macro), F1-Score (Macro)
- Normalized Confusion Matrices
- Classification Reports
- Generates high-resolution visualization figures in reports/figures/
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    FEATURE_COLS, CATEGORY_LABELS, SEVERITY_LABELS
)

def evaluate_all_models(target_col: str, model_subfolder: str) -> pd.DataFrame:
    """Evaluates all base and ensemble models on the held-out test split."""
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    
    save_dir = MODELS_DIR / model_subfolder
    X_test_raw = test_df[FEATURE_COLS].values
    X_test_scaled = scaler.transform(X_test_raw)
    y_test = test_df[target_col].values
    
    labels = CATEGORY_LABELS if "category" in target_col else SEVERITY_LABELS
    
    base_names = [
        "random_forest", "extra_trees", "gradient_boosting",
        "k-nearest_neighbors", "support_vector_machine"
    ]
    
    base_models = {}
    base_probs = []
    
    for name in base_names:
        m_path = save_dir / f"{name}.pkl"
        if m_path.exists():
            base_models[name] = joblib.load(m_path)
            is_scaled = name in ["k-nearest_neighbors", "support_vector_machine"]
            probs = base_models[name].predict_proba(X_test_scaled if is_scaled else X_test_raw)
            base_probs.append(probs)

    if not base_models:
        print(f"No trained models found in {save_dir}!")
        return pd.DataFrame()

    results = []
    preds_dict = {}
    
    # 1. Base Models
    for i, (name, model) in enumerate(base_models.items()):
        probs = base_probs[i]
        preds = np.argmax(probs, axis=1)
        preds_dict[name] = preds
        results.append({
            "Model": name.replace("_", " ").title(),
            "Accuracy": accuracy_score(y_test, preds),
            "Precision (Macro)": precision_score(y_test, preds, average="macro", zero_division=0),
            "Recall (Macro)": recall_score(y_test, preds, average="macro", zero_division=0),
            "F1-Score (Macro)": f1_score(y_test, preds, average="macro", zero_division=0)
        })

    # 2. Soft Voting Classifier
    soft_vote_probs = np.mean(base_probs, axis=0)
    soft_vote_preds = np.argmax(soft_vote_probs, axis=1)
    preds_dict["soft_voting"] = soft_vote_preds
    results.append({
        "Model": "Soft Voting Ensemble",
        "Accuracy": accuracy_score(y_test, soft_vote_preds),
        "Precision (Macro)": precision_score(y_test, soft_vote_preds, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_test, soft_vote_preds, average="macro", zero_division=0),
        "F1-Score (Macro)": f1_score(y_test, soft_vote_preds, average="macro", zero_division=0)
    })

    # 3. Stacking Classifier
    meta_path = save_dir / "stacking_meta_learner.pkl"
    if meta_path.exists():
        meta_learner = joblib.load(meta_path)
        meta_X_test = np.hstack(base_probs)
        stack_preds = meta_learner.predict(meta_X_test)
        preds_dict["stacking_ensemble"] = stack_preds
        results.append({
            "Model": "Stacking Classifier (Ensemble)",
            "Accuracy": accuracy_score(y_test, stack_preds),
            "Precision (Macro)": precision_score(y_test, stack_preds, average="macro", zero_division=0),
            "Recall (Macro)": recall_score(y_test, stack_preds, average="macro", zero_division=0),
            "F1-Score (Macro)": f1_score(y_test, stack_preds, average="macro", zero_division=0)
        })

    results_df = pd.DataFrame(results).sort_values(by="F1-Score (Macro)", ascending=False)
    metrics_file = METRICS_DIR / f"{model_subfolder}_comparison.csv"
    results_df.to_csv(metrics_file, index=False)
    print(f"\nSaved evaluation metrics to {metrics_file}")
    
    # Generate and save Normalized Confusion Matrix for the winning Stacking Classifier
    if "stacking_ensemble" in preds_dict:
        cm = confusion_matrix(y_test, preds_dict["stacking_ensemble"], normalize="true")
        plt.figure(figsize=(10, 8))
        present_labels = [labels[i] for i in sorted(np.unique(y_test))]
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=present_labels, yticklabels=present_labels)
        plt.title(f"Normalized Confusion Matrix - Stacking Ensemble ({model_subfolder.replace('_', ' ').title()})")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig_path = FIGURES_DIR / f"{model_subfolder}_stacking_confusion_matrix.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix figure to {fig_path}")

    # Generate Model Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    bar_df = results_df.copy()
    bar_df["Accuracy %"] = bar_df["Accuracy"] * 100
    bar_df["Macro-F1 %"] = bar_df["F1-Score (Macro)"] * 100
    bar_melted = bar_df.melt(id_vars=["Model"], value_vars=["Accuracy %", "Macro-F1 %"], var_name="Metric", value_name="Score")
    
    sns.barplot(data=bar_melted, x="Model", y="Score", hue="Metric", palette=["#0284c7", "#10b981"])
    plt.title(f"Model Performance Comparison - {model_subfolder.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    comp_fig_path = FIGURES_DIR / f"{model_subfolder}_model_comparison.png"
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to {comp_fig_path}")

    return results_df

if __name__ == "__main__":
    print("\n--- Evaluating Fault Category Pipeline ---")
    evaluate_all_models("target_category_idx", "fault_category")
    print("\n--- Evaluating Fault Severity Pipeline ---")
    evaluate_all_models("target_severity_idx", "fault_severity")
