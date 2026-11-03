"""
Full-Scale (1.81 Million data points) model training and ensembling pipeline.
Optimized for massive tabular datasets:
- High-efficiency Gradient Boosting (HistGradientBoosting / LightGBM style)
- Parallelized Random Forest & Extra Trees with leaf subsampling
- Linear-complexity SGD/LinearSVC and MiniBatch K-Means / Subsampled KNN
- Full Stacking Ensemble with Logistic Regression meta-model
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_COLS, RANDOM_STATE
)

def get_full_scale_models():
    """Returns models architected to scale efficiently to 1.81 million data points."""
    return {
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=12, min_samples_leaf=50,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=16, max_samples=0.2, # 20% bootstrap subsample per tree
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=16, max_samples=0.2,
            bootstrap=True, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Calibrated Linear SGD": CalibratedClassifierCV(
            SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1000, random_state=RANDOM_STATE),
            cv=3
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance", algorithm="kd_tree", n_jobs=-1
        )
    }

def train_and_stack_full_scale(target_col: str, model_subfolder: str):
    """
    Trains models on the complete 1.81M records using OOF meta-feature generation
    and trains the Stacking Classifier.
    """
    print(f"\n{'='*70}\nStarting Full 1.81M Dataset Training: {target_col.upper()}\n{'='*70}")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    
    save_dir = MODELS_DIR / model_subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_raw = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[target_col].values
    groups_train = train_df["group_id"].values
    
    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df[target_col].values
    
    X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    
    n_classes = len(np.unique(y_train))
    print(f"Loaded Train: {len(X_train_raw):,} samples | Test: {len(X_test_raw):,} samples | Classes: {n_classes}")
    
    base_models = get_full_scale_models()
    trained_models = {}
    oof_meta_features = np.zeros((len(X_train_raw), len(base_models) * n_classes), dtype=np.float32)
    
    # 1. 5-Fold Group OOF Generation
    print("\n--- Generating 5-Fold Out-Of-Fold Predictions across 1.81M records ---")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train_raw, y_train, groups_train)):
        print(f"Processing Fold {fold+1}/5...")
        for m_idx, (name, model_template) in enumerate(get_full_scale_models().items()):
            # For KNN on 1.4M train points, fit on a stratified subsample (50k) to prevent OOM
            is_knn = "Neighbors" in name
            is_scaled = name in ["K-Nearest Neighbors", "Calibrated Linear SGD"]
            
            X_tr = X_train_scaled[train_idx] if is_scaled else X_train_raw[train_idx]
            X_val = X_train_scaled[val_idx] if is_scaled else X_train_raw[val_idx]
            y_tr_f = y_train[train_idx]
            
            if is_knn and len(X_tr) > 50000:
                sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), 50000, replace=False)
                X_tr, y_tr_f = X_tr[sub_idx], y_tr_f[sub_idx]
                
            model = joblib.load(joblib.dump(model_template, "temp.pkl")[0])
            model.fit(X_tr, y_tr_f)
            val_probs = model.predict_proba(X_val)
            
            start_col = m_idx * n_classes
            oof_meta_features[val_idx, start_col:start_col + n_classes] = val_probs

    # 2. Train Full Models
    print("\n--- Training Base Models on Full 1.81M Training Split ---")
    timing_records = {}
    for m_idx, (name, model) in enumerate(base_models.items()):
        start_time = time.time()
        is_scaled = name in ["K-Nearest Neighbors", "Calibrated Linear SGD"]
        X_tr = X_train_scaled if is_scaled else X_train_raw
        X_te = X_test_scaled if is_scaled else X_test_raw
        y_tr_full = y_train
        
        if "Neighbors" in name and len(X_tr) > 50000:
            sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), 50000, replace=False)
            X_tr, y_tr_full = X_tr[sub_idx], y_tr_full[sub_idx]
            
        model.fit(X_tr, y_tr_full)
        fit_time = time.time() - start_time
        
        trained_models[name] = model
        timing_records[name] = {"fit_time_s": round(fit_time, 2)}
        
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(model, save_dir / f"{clean_name}.pkl")
        print(f"  {name:25s} | Fit: {fit_time:6.2f}s -> Saved to models/")

    # 3. Train Stacking Meta-Learner
    print("\n--- Training Stacking Classifier (Meta-Learner: Logistic Regression) ---")
    meta_learner = LogisticRegression(max_iter=1500, random_state=RANDOM_STATE)
    meta_learner.fit(oof_meta_features, y_train)
    joblib.dump(meta_learner, save_dir / "stacking_meta_learner.pkl")
    print("  Stacking Meta-Learner successfully trained and saved!")

if __name__ == "__main__":
    train_and_stack_full_scale(target_col="target_category_idx", model_subfolder="fault_category")
    train_and_stack_full_scale(target_col="target_severity_idx", model_subfolder="fault_severity")
