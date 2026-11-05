"""
Two-Stage Hierarchical Training Pipeline:
Stage 1: Binary Anomaly Filter (Normal vs Anomaly) with heavy positive recall optimization.
Stage 2: Multiclass Fault Diagnosis (8 Fault Categories & 5 Severity Grades) trained strictly
         on the anomalous subset to eliminate majority-class gradient dilution.
Ensemble: Residual Stacking (combines base model probabilities + Soft Voting consensus).
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_COLS, RANDOM_STATE,
    STAGE2_CAT_TO_IDX, STAGE2_SEV_TO_IDX
)
from src.model_registry import MODEL_REGISTRY
from src.train import align_probabilities

def train_stage1_binary_detector():
    """
    Trains Stage 1 Binary Anomaly Filter.
    Optimizes for high recall on positive anomalies (>95%) so actual faults are never missed.
    """
    print("\n" + "="*80)
    print("STAGE 1: TRAINING BINARY ANOMALY FILTER (Normal vs Anomaly)")
    print("="*80)
    
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    
    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = (train_df["fault_category"] != "normal").astype(int).values
    
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = (test_df["fault_category"] != "normal").astype(int).values
    
    print(f"Train samples: {len(X_train):,} (Anomalies: {y_train.sum():,})")
    print(f"Test samples:  {len(X_test):,} (Anomalies: {y_test.sum():,})")
    
    # Train high-recall HistGradientBoosting with class_weight='balanced'
    start_t = time.time()
    binary_detector = HistGradientBoostingClassifier(
        max_iter=150, max_depth=10, min_samples_leaf=30,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    binary_detector.fit(X_train, y_train)
    fit_time = time.time() - start_t
    
    # Threshold Search on an internal validation split of the training set
    print("\n--- Running Stage 1 Validation-Driven Threshold Search ---")
    sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sub_idx, val_sub_idx = next(sgkf_val.split(X_train, y_train, train_df["group_id"].values))
    
    val_model = HistGradientBoostingClassifier(
        max_iter=150, max_depth=10, min_samples_leaf=30,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    val_model.fit(X_train[train_sub_idx], y_train[train_sub_idx])
    val_probs = val_model.predict_proba(X_train[val_sub_idx])[:, 1]
    
    candidate_thresholds = np.linspace(0.15, 0.70, 12)
    best_threshold = 0.35
    best_f1 = 0.0
    
    print(f"  {'Threshold':<10} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<10} | {'FNR':<8}")
    print("  " + "-"*55)
    for th in candidate_thresholds:
        v_preds = (val_probs >= th).astype(int)
        v_rec = recall_score(y_train[val_sub_idx], v_preds, zero_division=0)
        v_prec = precision_score(y_train[val_sub_idx], v_preds, zero_division=0)
        v_f1 = f1_score(y_train[val_sub_idx], v_preds, zero_division=0)
        v_fnr = 1.0 - v_rec
        print(f"  {th:<10.2f} | {v_rec*100:<7.2f}% | {v_prec*100:<9.2f}% | {v_f1*100:<9.2f}% | {v_fnr*100:<7.2f}%")
        # Optimization policy: Require >= 95% Recall, then maximize F1
        if v_rec >= 0.95 and v_f1 > best_f1:
            best_f1 = v_f1
            best_threshold = float(th)

    print(f"\nOptimal Stage 1 Validation Threshold Selected: {best_threshold:.2f} (Recall >= 95%, F1={best_f1*100:.2f}%)")

    # Evaluate Stage 1 on untouched held-out test partition
    test_probs = binary_detector.predict_proba(X_test)[:, 1]
    threshold = best_threshold
    test_preds = (test_probs >= threshold).astype(int)
    
    rec = recall_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    b_acc = balanced_accuracy_score(y_test, test_preds)
    
    print(f"\nStage 1 Filter Results (Threshold={threshold}):")
    print(f"  Anomaly Recall (Sensitivity): {rec*100:.2f}% (Target: >95%)")
    print(f"  Precision:                    {prec*100:.2f}%")
    print(f"  Balanced Accuracy:            {b_acc*100:.2f}%")
    print(f"  F1-Score:                     {f1*100:.2f}%")
    print(f"  Training Fit Time:            {fit_time:.2f}s")
    
    # Save model and threshold
    save_dir = MODELS_DIR / "stage1_binary"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(binary_detector, save_dir / "binary_detector.pkl")
    joblib.dump({"threshold": threshold, "metrics": {"recall": rec, "f1": f1}}, save_dir / "meta.pkl")
    print(f"Stage 1 model saved to {save_dir / 'binary_detector.pkl'}")
    return binary_detector

def train_stage2_multiclass_pipeline(target_type: str = "category"):
    """
    Trains Stage 2 Multi-Class Classifier strictly on anomalous observations.
    target_type: 'category' (8 classes) or 'severity' (5 classes)
    Employs Residual Stacking with Soft-Voting consensus feature vector.
    """
    print("\n" + "="*80)
    print(f"STAGE 2: TRAINING SPECIALIZED MULTI-CLASS IDENTIFIER ({target_type.upper()})")
    print("="*80)
    
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    
    # Filter strictly to anomalous samples
    train_anom = train_df[train_df["fault_category"] != "normal"].copy()
    test_anom = test_df[test_df["fault_category"] != "normal"].copy()
    
    if target_type == "category":
        train_anom["target_idx"] = train_anom["fault_category"].map(STAGE2_CAT_TO_IDX).values
        test_anom["target_idx"] = test_anom["fault_category"].map(STAGE2_CAT_TO_IDX).values
        n_classes = len(STAGE2_CAT_TO_IDX)
        model_subfolder = "stage2_fault_category"
    else:
        # Standardize severity with explicit validation
        unmapped_train = train_anom["severity"][~train_anom["severity"].isin(STAGE2_SEV_TO_IDX.keys())].unique()
        unmapped_test = test_anom["severity"][~test_anom["severity"].isin(STAGE2_SEV_TO_IDX.keys())].unique()
        if len(unmapped_train) > 0 or len(unmapped_test) > 0:
            raise ValueError(f"Unknown severity labels detected! Train: {unmapped_train}, Test: {unmapped_test}. Expected one of: {list(STAGE2_SEV_TO_IDX.keys())}")
            
        train_anom["target_idx"] = train_anom["severity"].map(STAGE2_SEV_TO_IDX).astype(int).values
        test_anom["target_idx"] = test_anom["severity"].map(STAGE2_SEV_TO_IDX).astype(int).values
        n_classes = len(STAGE2_SEV_TO_IDX)
        model_subfolder = "stage2_fault_severity"
        
    save_dir = MODELS_DIR / model_subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_raw = train_anom[FEATURE_COLS].values.astype(np.float32)
    y_train = train_anom["target_idx"].values.astype(np.int64)
    groups_train = train_anom["group_id"].values
    
    X_test_raw = test_anom[FEATURE_COLS].values.astype(np.float32)
    y_test = test_anom["target_idx"].values.astype(np.int64)
    
    X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    
    print(f"Stage 2 Train Anomalies: {len(X_train_raw):,} | Test Anomalies: {len(X_test_raw):,} | Classes: {n_classes}")
    
    num_models = len(MODEL_REGISTRY)
    # Meta features: base models predictions + Soft Voting consensus (1 extra block)
    oof_meta_features = np.zeros((len(X_train_raw), (num_models + 1) * n_classes), dtype=np.float32)
    test_meta_features = np.zeros((len(X_test_raw), (num_models + 1) * n_classes), dtype=np.float32)
    
    # 1. 5-Fold Group OOF Loop
    print("\n--- Generating 5-Fold OOF Predictions with Fold-Isolated Scaling ---")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train_raw, y_train, groups_train)):
        print(f"Processing Fold {fold+1}/5...")
        fold_scaler = RobustScaler()
        fold_X_tr_s = fold_scaler.fit_transform(X_train_raw[train_idx])
        fold_X_val_s = fold_scaler.transform(X_train_raw[val_idx])
        
        fold_val_preds_list = []
        
        for m_idx, (m_key, m_info) in enumerate(MODEL_REGISTRY.items()):
            needs_s = m_info["needs_scaling"]
            X_tr = fold_X_tr_s if needs_s else X_train_raw[train_idx]
            X_val = fold_X_val_s if needs_s else X_train_raw[val_idx]
            
            model = m_info["factory"](RANDOM_STATE)
            model.fit(X_tr, y_train[train_idx])
            
            raw_p = model.predict_proba(X_val)
            aligned_p = align_probabilities(raw_p, model.classes_, n_classes)
            fold_val_preds_list.append(aligned_p)
            
            start_col = m_idx * n_classes
            oof_meta_features[val_idx, start_col:start_col + n_classes] = aligned_p
            
        # Residual Skip-Connection: Append fold Soft-Voting consensus vector
        soft_vote_fold = np.mean(fold_val_preds_list, axis=0)
        soft_col_start = num_models * n_classes
        oof_meta_features[val_idx, soft_col_start:soft_col_start + n_classes] = soft_vote_fold

    # 2. Train Full Base Models
    print("\n--- Training Base Models on Full Anomalous Dataset ---")
    test_preds_list = []
    timing_records = {}
    
    for m_idx, (m_key, m_info) in enumerate(MODEL_REGISTRY.items()):
        start_t = time.time()
        needs_s = m_info["needs_scaling"]
        X_tr = X_train_scaled if needs_s else X_train_raw
        X_te = X_test_scaled if needs_s else X_test_raw
        
        model = m_info["factory"](RANDOM_STATE)
        model.fit(X_tr, y_train)
        fit_t = time.time() - start_t
        
        raw_p = model.predict_proba(X_te)
        aligned_p = align_probabilities(raw_p, model.classes_, n_classes)
        test_preds_list.append(aligned_p)
        
        start_col = m_idx * n_classes
        test_meta_features[:, start_col:start_col + n_classes] = aligned_p
        
        joblib.dump(model, save_dir / m_info["filename"])
        timing_records[m_info["name"]] = {"fit_time_s": round(fit_t, 2)}
        print(f"  {m_info['name']:25s} | Fit: {fit_t:6.2f}s -> Saved")
        
    # Append Soft-Voting consensus to test meta-features
    soft_vote_test = np.mean(test_preds_list, axis=0)
    soft_col_start = num_models * n_classes
    test_meta_features[:, soft_col_start:soft_col_start + n_classes] = soft_vote_test
    
    soft_vote_preds = np.argmax(soft_vote_test, axis=1)
    soft_f1 = f1_score(y_test, soft_vote_preds, average="macro", zero_division=0)
    print(f"\nConsensus Soft-Voting Benchmark: Macro-F1 = {soft_f1*100:.2f}%")
    
    # 3. Residual Stacking Meta-Learner Training
    print("\n--- Training Residual Stacking Meta-Learner (Logistic Regression L2) ---")
    meta_learner = LogisticRegression(C=0.5, penalty="l2", class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    meta_learner.fit(oof_meta_features, y_train)
    
    stack_preds = meta_learner.predict(test_meta_features)
    stack_f1 = f1_score(y_test, stack_preds, average="macro", zero_division=0)
    stack_bacc = balanced_accuracy_score(y_test, stack_preds)
    
    print(f"Residual Stacking Meta-Learner Results:")
    print(f"  Macro-F1:          {stack_f1*100:.2f}%")
    print(f"  Balanced Accuracy: {stack_bacc*100:.2f}%")
    
    joblib.dump(meta_learner, save_dir / "stacking_meta_learner.pkl")
    joblib.dump(timing_records, save_dir / "timing_benchmark.pkl")
    print(f"All Stage 2 models saved to {save_dir}")

if __name__ == "__main__":
    train_stage1_binary_detector()
    train_stage2_multiclass_pipeline(target_type="category")
    train_stage2_multiclass_pipeline(target_type="severity")
