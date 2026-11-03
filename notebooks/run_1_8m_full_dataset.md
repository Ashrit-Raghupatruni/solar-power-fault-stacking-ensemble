# Full-Scale 1.81 Million Data Points Training & Preprocessing (Colab / Kaggle)

## Cell 1: Clone Repository & Setup
```bash
!git clone https://github.com/Ashrit-Raghupatruni/solar-power-fault-stacking-ensemble.git
%cd solar-power-fault-stacking-ensemble
!pip install -r requirements.txt
```

## Cell 2: Process the Entire 1.81 Million Raw Telemetry Dataset
*(Processes Station 1 (604,800) + Station 2 (604,800) + Station 3 (604,800) = 1,814,400 rows)*
```python
from src.preprocessing_full_1_8m import run_full_1_8m_preprocessing

print("=== Starting Full 1.81M Data Processing & Group Splitting ===")
run_full_1_8m_preprocessing()
```

## Cell 3: Train Fault Category Models on the Full 1.81M Dataset
```python
from src.train_full_1_8m import train_and_stack_full_scale

print("=== Training 1.81M Fault Category Pipeline ===")
train_and_stack_full_scale(target_col="target_category_idx", model_subfolder="fault_category")
```

## Cell 4: Train Fault Severity Models on the Full 1.81M Dataset
```python
print("=== Training 1.81M Fault Severity Pipeline ===")
train_and_stack_full_scale(target_col="target_severity_idx", model_subfolder="fault_severity")
```

## Cell 5: Evaluate All 7 Models on the 1.81M Held-Out Test Set
```python
from src.evaluate import evaluate_all_models

results_1_8m = evaluate_all_models("target_category_idx", "fault_category")
display(results_1_8m)
```
