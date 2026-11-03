# Solar & Electrical Power Fault and Anomaly Detection (Colab / Kaggle Edition)
# Run on Google Colab or Kaggle Notebooks with GPU/High-RAM

# --- CELL 1: Clone Repository & Setup ---
"""
!git clone https://github.com/Ashrit-Raghupatruni/solar-power-fault-stacking-ensemble.git
%cd solar-power-fault-stacking-ensemble
!pip install -r requirements.txt
"""

# --- CELL 2: Verify Processed Data ---
"""
import pandas as pd
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
print(f"Loaded Train: {train_df.shape}, Test: {test_df.shape}")
"""

# --- CELL 3: Train All 5 Base Models & Stacking Ensemble ---
"""
from src.train import train_and_stack_pipeline

# 1. Train Fault Category Pipeline (9 classes)
print("--- Training Fault Category Pipeline ---")
train_and_stack_pipeline(target_col="target_category_idx", model_subfolder="fault_category")

# 2. Train Fault Severity Pipeline (6 classes)
print("--- Training Fault Severity Pipeline ---")
train_and_stack_pipeline(target_col="target_severity_idx", model_subfolder="fault_severity")
"""

# --- CELL 4: Evaluate All 7 Models & View Performance ---
"""
from src.evaluate import evaluate_all_models
metrics_cat = evaluate_all_models("target_category_idx", "fault_category")
display(metrics_cat)
"""
