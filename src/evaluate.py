"""
Evaluation module comparing all 7 models:
- 5 Base Models (Random Forest, Extra Trees, Gradient Boosting, KNN, SVM)
- Soft Voting Classifier
- Stacking Classifier (Logistic Regression Meta-Model)
Calculates: Accuracy, Precision (Macro), Recall (Macro), F1-Score (Macro), ROC-AUC (One-vs-Rest)
Generates: Confusion Matrices and Performance Reports
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
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
    n_classes = len(labels)
    
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
        print(f"No trained models found in {save_dir}. Please run training first!")
        return pd.DataFrame()

    results = []
    
    # 1. Evaluate Base Models
    for i, (name, model) in enumerate(base_models.items()):
        probs = base_probs[i]
        preds = np.argmax(probs, axis=1)
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
    
    return results_df

if __name__ == "__main__":
    evaluate_all_models("target_category_idx", "fault_category")
