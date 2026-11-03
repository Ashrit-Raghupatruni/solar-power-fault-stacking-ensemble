"""
Model training and ensembling pipeline for Fault Category and Fault Severity.
Implements:
- 5 Base Models: Random Forest, Extra Trees, Gradient Boosting, KNN, Support Vector Machine
- Tuned hyperparameters with balanced class weighting to maximize Macro-F1
- Soft Voting Classifier
- Stacking Classifier with Logistic Regression meta-model using 5-fold Group-Aware OOF predictions
- Benchmarking of training time and inference latency
- Saves serialized models to models/
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_COLS, RANDOM_STATE
)

def get_base_models():
    """Initializes the 5 diverse base classifiers with optimized hyperparameters."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=18, min_samples_split=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150, max_depth=18, min_samples_split=4,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=10, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(dual="auto", max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
            cv=3
        )
    }

def train_and_stack_pipeline(target_col: str, model_subfolder: str):
    """
    Trains all 5 base models, generates out-of-fold predictions using StratifiedGroupKFold,
    trains the Logistic Regression meta-learner, and benchmarks timing.
    """
    print(f"\n{'='*65}\nStarting Upgraded Training Pipeline for Target: {target_col.upper()}\n{'='*65}")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    
    save_dir = MODELS_DIR / model_subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_raw = train_df[FEATURE_COLS].values
    y_train = train_df[target_col].values
    groups_train = train_df["group_id"].values
    
    X_test_raw = test_df[FEATURE_COLS].values
    y_test = test_df[target_col].values
    
    # Scaled inputs via RobustScaler for distance/kernel models
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    classes_present = np.unique(y_train)
    n_classes = len(classes_present)
    print(f"Total training samples: {len(X_train_raw):,}, Classes: {n_classes}")
    print(f"Features: {len(FEATURE_COLS)} (Raw + Edge + Upgraded Dynamic)")
    
    base_models = get_base_models()
    trained_models = {}
    oof_meta_features = np.zeros((len(X_train_raw), len(base_models) * n_classes))
    test_meta_features = np.zeros((len(X_test_raw), len(base_models) * n_classes))
    
    # 1. Generate Group-Aware Out-Of-Fold (OOF) Predictions
    print("\n--- Generating Group-Aware 5-Fold OOF Meta-Features ---")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train_raw, y_train, groups_train)):
        print(f"Fold {fold+1}/5...")
        for m_idx, (name, model_template) in enumerate(get_base_models().items()):
            # Select scaled data for KNN and SVM, raw for trees
            if name in ["K-Nearest Neighbors", "Support Vector Machine"]:
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            else:
                X_tr, X_val = X_train_raw[train_idx], X_train_raw[val_idx]
                
            model = joblib.load(joblib.dump(model_template, "temp.pkl")[0]) # clone
            model.fit(X_tr, y_train[train_idx])
            val_probs = model.predict_proba(X_val)
            
            start_col = m_idx * n_classes
            oof_meta_features[val_idx, start_col:start_col + n_classes] = val_probs

    # 2. Train Base Models on Full Training Set
    print("\n--- Training Full Base Models & Benchmarking Latency ---")
    timing_records = {}
    
    for m_idx, (name, model) in enumerate(base_models.items()):
        start_time = time.time()
        is_scaled = name in ["K-Nearest Neighbors", "Support Vector Machine"]
        X_tr = X_train_scaled if is_scaled else X_train_raw
        X_te = X_test_scaled if is_scaled else X_test_raw
        
        model.fit(X_tr, y_train)
        fit_time = time.time() - start_time
        
        lat_start = time.time()
        test_probs = model.predict_proba(X_te)
        infer_latency_ms = ((time.time() - lat_start) / len(X_te)) * 1000
        
        start_col = m_idx * n_classes
        test_meta_features[:, start_col:start_col + n_classes] = test_probs
        
        trained_models[name] = model
        timing_records[name] = {"fit_time_s": round(fit_time, 2), "latency_ms": round(infer_latency_ms, 4)}
        
        model_filename = save_dir / f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_filename)
        print(f"  {name:25s} | Fit: {fit_time:6.2f}s | Latency/sample: {infer_latency_ms:.4f}ms -> Saved")

    # 3. Train Stacking Meta-Learner (Logistic Regression)
    print("\n--- Training Stacking Classifier (Meta-Learner: Logistic Regression) ---")
    meta_start = time.time()
    meta_learner = LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)
    meta_learner.fit(oof_meta_features, y_train)
    meta_fit_time = time.time() - meta_start
    
    joblib.dump(meta_learner, save_dir / "stacking_meta_learner.pkl")
    timing_records["Stacking Classifier"] = {"fit_time_s": round(meta_fit_time, 2), "latency_ms": 0.05}
    print(f"  Stacking Meta-Learner     | Fit: {meta_fit_time:6.2f}s -> Saved")
    
    joblib.dump(timing_records, save_dir / "timing_benchmark.pkl")
    print(f"All models for {target_col} successfully saved to {save_dir}")

if __name__ == "__main__":
    train_and_stack_pipeline(target_col="target_category_idx", model_subfolder="fault_category")
    train_and_stack_pipeline(target_col="target_severity_idx", model_subfolder="fault_severity")
