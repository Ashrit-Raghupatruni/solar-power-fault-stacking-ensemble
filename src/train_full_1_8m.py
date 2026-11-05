"""
Full-scale (1.81M data points) model training module.
Methodological Fixes Implemented:
1. Powered by MODEL_REGISTRY: Guaranteed shared naming, scaling flags, and order.
2. Fold-Isolated Preprocessing: Fits RobustScaler strictly inside each inner OOF training fold.
3. Strict Global Class Probability Alignment: Eliminates dimension and alignment mismatch.
4. Balanced Meta-Learners: Tuned L2 Logistic Regression & Ridge meta-learners.
"""
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_COLS, RANDOM_STATE
from src.model_registry import MODEL_REGISTRY, META_LEARNER_REGISTRY
from src.train import align_probabilities

def train_and_stack_full_scale(target_col: str, model_subfolder: str):
    """
    Trains all registry base models and meta-learners with inner fold-isolated scaling.
    """
    print(f"\n{'='*75}\nStarting Full 1.81M Registry-Aligned Training: {target_col.upper()}\n{'='*75}")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    save_dir = MODELS_DIR / model_subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_raw = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.int64)
    groups_train = train_df["group_id"].values
    
    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.int64)
    
    total_classes = len(np.unique(np.concatenate([y_train, y_test])))
    print(f"Loaded Train: {len(X_train_raw):,} samples | Test: {len(X_test_raw):,} samples | Classes: {total_classes}")
    
    num_models = len(MODEL_REGISTRY)
    oof_meta_features = np.zeros((len(X_train_raw), num_models * total_classes), dtype=np.float32)
    test_meta_features = np.zeros((len(X_test_raw), num_models * total_classes), dtype=np.float32)
    
    # 1. Inner 5-Fold Group OOF Loop with Fold-Isolated Scaling
    print("\n--- Step 1: 5-Fold OOF Predictions with Fold-Isolated Scaling ---")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train_raw, y_train, groups_train)):
        print(f"Processing Fold {fold+1}/5...")
        
        # SCIENTIFIC FIX: Fit fold scaler strictly on train_idx to eliminate look-ahead leakage
        fold_scaler = RobustScaler()
        fold_X_tr_scaled = fold_scaler.fit_transform(X_train_raw[train_idx])
        fold_X_val_scaled = fold_scaler.transform(X_train_raw[val_idx])
        
        for m_idx, (m_key, m_info) in enumerate(MODEL_REGISTRY.items()):
            needs_scaling = m_info["needs_scaling"]
            X_tr = fold_X_tr_scaled if needs_scaling else X_train_raw[train_idx]
            X_val = fold_X_val_scaled if needs_scaling else X_train_raw[val_idx]
            y_tr_fold = y_train[train_idx]
            
            # Subsample KNN on huge splits
            if "neighbors" in m_key and len(X_tr) > 40000:
                sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), 40000, replace=False)
                X_tr, y_tr_fold = X_tr[sub_idx], y_tr_fold[sub_idx]
                
            model = m_info["factory"](RANDOM_STATE)
            model.fit(X_tr, y_tr_fold)
            
            raw_val_probs = model.predict_proba(X_val)
            aligned_val_probs = align_probabilities(raw_val_probs, model.classes_, total_classes)
            
            start_col = m_idx * total_classes
            oof_meta_features[val_idx, start_col:start_col + total_classes] = aligned_val_probs

    # 2. Train Base Models on Full Training Partition
    print("\n--- Step 2: Training Full Base Models from MODEL_REGISTRY ---")
    final_scaler = joblib.load(MODELS_DIR / "preprocessors" / "scaler.pkl")
    X_train_scaled = final_scaler.transform(X_train_raw)
    X_test_scaled = final_scaler.transform(X_test_raw)
    
    timing_records = {}
    for m_idx, (m_key, m_info) in enumerate(MODEL_REGISTRY.items()):
        start_time = time.time()
        needs_scaling = m_info["needs_scaling"]
        X_tr = X_train_scaled if needs_scaling else X_train_raw
        X_te = X_test_scaled if needs_scaling else X_test_raw
        y_tr_full = y_train
        
        if "neighbors" in m_key and len(X_tr) > 40000:
            sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr), 40000, replace=False)
            X_tr, y_tr_full = X_tr[sub_idx], y_tr_full[sub_idx]
            
        model = m_info["factory"](RANDOM_STATE)
        model.fit(X_tr, y_tr_full)
        fit_time = time.time() - start_time
        
        raw_test_probs = model.predict_proba(X_te)
        aligned_test_probs = align_probabilities(raw_test_probs, model.classes_, total_classes)
        
        start_col = m_idx * total_classes
        test_meta_features[:, start_col:start_col + total_classes] = aligned_test_probs
        
        # Save using canonical filename from registry
        save_path = save_dir / m_info["filename"]
        joblib.dump(model, save_path)
        timing_records[m_info["name"]] = {"fit_time_s": round(fit_time, 2)}
        print(f"  {m_info['name']:25s} | Fit: {fit_time:6.2f}s -> Saved as {m_info['filename']}")

    # 3. Train Stacking Meta-Learner (Selected on Internal OOF Validation, NEVER on Test)
    print("\n--- Step 3: Meta-Learner Tuning & Selection (Using Internal OOF Validation) ---")
    from sklearn.metrics import f1_score
    
    # Create internal validation split of the OOF meta-features
    meta_val_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    meta_tr_idx, meta_val_idx = next(meta_val_splitter.split(oof_meta_features, y_train, groups_train))
    
    best_meta_factory = None
    best_val_f1 = -1.0
    
    for meta_key, meta_info in META_LEARNER_REGISTRY.items():
        # Evaluate candidate meta-learner on inner validation split
        candidate_model = meta_info["factory"](RANDOM_STATE)
        candidate_model.fit(oof_meta_features[meta_tr_idx], y_train[meta_tr_idx])
        val_preds = candidate_model.predict(oof_meta_features[meta_val_idx])
        val_score = f1_score(y_train[meta_val_idx], val_preds, average="macro", zero_division=0)
        print(f"  {meta_info['name']:45s} | Inner Val Macro-F1: {val_score*100:.2f}%")
        if val_score > best_val_f1:
            best_val_f1 = val_score
            best_meta_factory = meta_info["factory"]
            
    # Retrain winning meta-learner on complete 100% OOF features
    print(f"\nWinning Meta-Learner Retrained on Full OOF Dataset (Val Macro-F1: {best_val_f1*100:.2f}%)")
    best_meta = best_meta_factory(RANDOM_STATE)
    best_meta.fit(oof_meta_features, y_train)
            
    joblib.dump(best_meta, save_dir / "stacking_meta_learner.pkl")
    joblib.dump(timing_records, save_dir / "timing_benchmark.pkl")
    print(f"Stacking Meta-Learner saved to {save_dir / 'stacking_meta_learner.pkl'}")

if __name__ == "__main__":
    train_and_stack_full_scale("target_category_idx", "fault_category")
    train_and_stack_full_scale("target_severity_idx", "fault_severity")
