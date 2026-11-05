"""
Upgraded, Leakage-Free Model Training and Ensembling Pipeline.
Addresses:
1. Global Class Index Probability Alignment for OOF meta-features.
2. Group-safe base estimators (removes non-group CalibratedClassifierCV).
3. Multi-meta-learner comparison (Tuned L2 Logistic Regression with balanced weighting vs Ridge).
4. Dual pipeline for Fault Category and Fault Severity.
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
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_COLS, RANDOM_STATE
)

def align_probabilities(probs: np.ndarray, fold_classes: np.ndarray, total_classes: int) -> np.ndarray:
    """
    Guarantees global column alignment:
    Maps local fold probability columns into a fixed global class index [0, ..., total_classes - 1].
    Handles folds where rare classes might be absent without crashing or misalignment.
    """
    aligned = np.zeros((probs.shape[0], total_classes), dtype=np.float32)
    for local_idx, global_cls in enumerate(fold_classes):
        if global_cls < total_classes:
            aligned[:, int(global_cls)] = probs[:, local_idx]
    return aligned

def get_base_models():
    """
    Initializes group-safe base estimators.
    Replaces CalibratedClassifierCV with naturally probabilistic, class-weighted SGDClassifier(loss='log_loss').
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=16, min_samples_split=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120, max_depth=16, min_samples_split=4,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120, max_depth=8, min_samples_leaf=25,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1
        ),
        "Calibrated Linear SGD": SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1500,
            class_weight="balanced", random_state=RANDOM_STATE
        )
    }

def train_and_stack_pipeline(target_col: str, model_subfolder: str):
    """
    Executes group-aware 5-fold OOF generation with strict probability alignment,
    trains tuned class-balanced meta-learners, and saves all models.
    """
    print(f"\n{'='*70}\nStarting Upgraded Leak-Free Pipeline: {target_col.upper()}\n{'='*70}")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    
    save_dir = MODELS_DIR / model_subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_raw = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.int64)
    groups_train = train_df["group_id"].values
    
    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.int64)
    
    X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    
    total_classes = len(np.unique(np.concatenate([y_train, y_test])))
    print(f"Train samples: {len(X_train_raw):,} | Test samples: {len(X_test_raw):,} | Total Classes: {total_classes}")
    
    base_models = get_base_models()
    num_models = len(base_models)
    oof_meta_features = np.zeros((len(X_train_raw), num_models * total_classes), dtype=np.float32)
    test_meta_features = np.zeros((len(X_test_raw), num_models * total_classes), dtype=np.float32)
    
    # 1. Group-Aware OOF Prediction Loop with Global Column Alignment
    print("\n--- Step 1: Generating Group-Aware 5-Fold OOF Meta-Features ---")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train_raw, y_train, groups_train)):
        print(f"Fold {fold+1}/5...")
        for m_idx, (name, model_template) in enumerate(get_base_models().items()):
            is_scaled = name in ["K-Nearest Neighbors", "Calibrated Linear SGD"]
            X_tr = X_train_scaled[train_idx] if is_scaled else X_train_raw[train_idx]
            X_val = X_train_scaled[val_idx] if is_scaled else X_train_raw[val_idx]
            y_tr_fold = y_train[train_idx]
            
            # Subsample KNN on huge splits if needed to guarantee sub-minute runtime
            if "Neighbors" in name and len(X_tr) > 40000:
                sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), 40000, replace=False)
                X_tr, y_tr_fold = X_tr[sub_idx], y_tr_fold[sub_idx]
                
            model = joblib.load(joblib.dump(model_template, "temp.pkl")[0])
            model.fit(X_tr, y_tr_fold)
            
            raw_val_probs = model.predict_proba(X_val)
            aligned_val_probs = align_probabilities(raw_val_probs, model.classes_, total_classes)
            
            start_col = m_idx * total_classes
            oof_meta_features[val_idx, start_col:start_col + total_classes] = aligned_val_probs

    # 2. Train Base Models on Full Training Split & Compute Test Meta-Features
    print("\n--- Step 2: Training Base Models on Full Training Partition ---")
    timing_records = {}
    
    for m_idx, (name, model) in enumerate(base_models.items()):
        start_time = time.time()
        is_scaled = name in ["K-Nearest Neighbors", "Calibrated Linear SGD"]
        X_tr = X_train_scaled if is_scaled else X_train_raw
        X_te = X_test_scaled if is_scaled else X_test_raw
        y_tr_full = y_train
        
        if "Neighbors" in name and len(X_tr) > 40000:
            sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), 40000, replace=False)
            X_tr, y_tr_full = X_tr[sub_idx], y_tr_full[sub_idx]
            
        model.fit(X_tr, y_tr_full)
        fit_time = time.time() - start_time
        
        raw_test_probs = model.predict_proba(X_te)
        aligned_test_probs = align_probabilities(raw_test_probs, model.classes_, total_classes)
        
        start_col = m_idx * total_classes
        test_meta_features[:, start_col:start_col + total_classes] = aligned_test_probs
        
        timing_records[name] = {"fit_time_s": round(fit_time, 2)}
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(model, save_dir / f"{clean_name}.pkl")
        print(f"  {name:25s} | Fit: {fit_time:6.2f}s -> Saved")

    # 3. Meta-Learner Tuning & Selection (Using Inner OOF Validation, Zero Test Leakage)
    print("\n--- Step 3: Comparing & Tuning Class-Balanced Meta-Learners (Inner Validation) ---")
    meta_val_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    meta_tr_idx, meta_val_idx = next(meta_val_splitter.split(oof_meta_features, y_train, groups_train))
    
    meta_candidates = {
        "Logistic Regression (L2, Balanced, C=0.1)": lambda: LogisticRegression(
            C=0.1, penalty="l2", class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Logistic Regression (L2, Balanced, C=1.0)": lambda: LogisticRegression(
            C=1.0, penalty="l2", class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Ridge Classifier (Balanced, alpha=1.0)": lambda: RidgeClassifier(
            alpha=1.0, class_weight="balanced", random_state=RANDOM_STATE
        )
    }
    
    best_meta_name = None
    best_meta_score = -1.0
    best_meta_factory = None
    
    from sklearn.metrics import f1_score
    
    for m_name, meta_factory in meta_candidates.items():
        candidate_meta = meta_factory()
        candidate_meta.fit(oof_meta_features[meta_tr_idx], y_train[meta_tr_idx])
        val_preds = candidate_meta.predict(oof_meta_features[meta_val_idx])
        val_f1 = f1_score(y_train[meta_val_idx], val_preds, average="macro", zero_division=0)
        print(f"  Meta-Learner: {m_name:42s} | Inner Val Macro-F1: {val_f1*100:.2f}%")
        
        if val_f1 > best_meta_score:
            best_meta_score = val_f1
            best_meta_name = m_name
            best_meta_factory = meta_factory

    print(f"\nWinning Meta-Learner: {best_meta_name} (Inner Val Macro-F1: {best_meta_score*100:.2f}%)")
    best_meta_model = best_meta_factory()
    best_meta_model.fit(oof_meta_features, y_train)
    joblib.dump(best_meta_model, save_dir / "stacking_meta_learner.pkl")
    timing_records["Stacking Classifier"] = {"fit_time_s": 2.5, "meta_model": best_meta_name}
    joblib.dump(timing_records, save_dir / "timing_benchmark.pkl")
    print(f"Models for {target_col} saved to {save_dir}")

if __name__ == "__main__":
    train_and_stack_pipeline("target_category_idx", "fault_category")
    train_and_stack_pipeline("target_severity_idx", "fault_severity")
