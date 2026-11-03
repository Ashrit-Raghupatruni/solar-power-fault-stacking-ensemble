# Solar & Electrical Power System Fault and Anomaly Detection Using Stacking Ensemble Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: Zenodo 20565892](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20565892-blue)](https://doi.org/10.5281/zenodo.20565892)

An end-to-end Machine Learning system for real-time electrical fault detection, severity estimation, and Photovoltaic (PV) inverter telemetry diagnosis using **Stacking Ensemble Learning** trained on the **full 1.81 Million record benchmark**.

---

## 📌 Project Overview
Solar energy systems and electrical grids experience abnormal operating conditions due to component degradation, unexpected load surges, inverter phase clipping, and harmonic distortion. 
This project builds an intelligent anomaly detection system using the **Zenodo Controlled-Fault Power Benchmark (1,814,400 per-second records across 3 monitoring stations)**.

### Key Technical Contributions
1. **Full-Scale 1.81 Million Data Points Pipeline**:
   - Ingests all 3 physical monitoring stations without downsampling:
     - **Station 1**: 604,800 rows (Laboratory baseline)
     - **Station 2**: 604,800 rows (Residential medium load)
     - **Station 3**: 604,800 rows (Industrial high inductive load)
   - Employs memory-efficient `float32` and `int8` downcasting (cutting RAM footprint from 2.2 GB to ~214 MB).
2. **Zero Data Leakage Protocol**:
   - Implements **`StratifiedGroupKFold`** on `(station_id, event_id)` and 20-minute contiguous normal temporal blocks.
   - **Zero event overlap**: Train partition has **1,465,501 samples (203 unique events)**; Held-out test partition has **348,899 samples (51 unique events)**.
3. **Dual Independent Prediction Pipelines**:
   - **Pipeline A (Fault Category - 9 Classes)**: `normal`, `voltage`, `load`, `frequency`, `power_quality`, `outage`, `sensor`, `context`, `compound`.
   - **Pipeline B (Fault Severity - 6 Grades)**: `none`, `incipient`, `low`, `medium`, `high`, `critical`.
4. **Model Diversity & Stacking Architecture**:
   - **5 Base Classifiers**: HistGradientBoosting, Random Forest, Extra Trees, Calibrated Linear SGD, and K-Nearest Neighbors.
   - **Ensemble Layer**: Soft Voting and a Stacking Classifier using **5-Fold Out-Of-Fold (OOF)** predictions with a **Logistic Regression Meta-Learner**.
   - **Preprocessing**: `RobustScaler` (scaling by median & IQR to preserve electrical transient spikes).
5. **Upgraded Dynamic Feature Engineering**:
   - Static AC metrics: Apparent Power ($S = |V| \cdot |I|$), Reactive Power ($Q = \sqrt{S^2 - P^2}$), Power Distortion Ratio.
   - Dynamic derivatives: $\frac{dV}{dt}, \frac{dI}{dt}, \frac{dP}{dt}$ capturing inrush transients and sags.
   - Power Factor Stability and Impedance Proxy ($Z = \frac{|V|}{|I|} \ \Omega$).
6. **Solar PV Telemetry Adapter**:
   - Directly maps solar string / inverter parameters (AC/DC voltage, current, active/reactive power, power factor) into standardized features without fabricating missing sensor channels.
7. **Production Interactive Web Dashboard**:
   - Built with zero external web dependencies, serving live at `http://localhost:8501`. Features live sensor diagnosis, solar PV adapter, and embedded confusion matrices.

---

## 📊 Evaluation Benchmarks on Full 1.81M Dataset

### 1. Fault Category Classification (9 Classes, 348,899 Test Records)

| Rank | Model Architecture | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Soft Voting Ensemble** | **94.42%** | **58.27%** | 49.99% | **90.81%** 🏆 |
| 🥈 **2** | **Stacking Classifier (Ensemble)** | **94.68%** | **46.96%** | 44.60% | 55.41% |
| 🥉 **3** | **Random Forest** | 85.53% | 41.75% | 39.14% | 58.44% |
| 4 | **Gradient Boosting** | 86.48% | 39.62% | 33.09% | **94.22%** |
| 5 | **Support Vector Machine (Linear RBF)** | **97.88%** | 37.26% | 49.01% | 39.00% |
| 6 | **Extra Trees** | 66.43% | 36.43% | 35.63% | 62.46% |
| 7 | **K-Nearest Neighbors** | 84.29% | 34.52% | 29.10% | 83.13% |

### 2. Fault Severity Classification (6 Grades, 348,899 Test Records)

| Rank | Model Architecture | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Soft Voting Ensemble** | **95.44%** | **53.32%** | 43.99% | **81.19%** 🏆 |
| 🥈 **2** | **Gradient Boosting** | 81.83% | 35.76% | 29.48% | **87.20%** |
| 🥉 **3** | **K-Nearest Neighbors** | 83.86% | 29.70% | 25.12% | 82.07% |
| 4 | **Support Vector Machine** | **97.88%** | 29.44% | 49.92% | 26.92% |
| 5 | **Random Forest** | 95.27% | 27.60% | 26.65% | 30.56% |
| 6 | **Extra Trees** | 82.68% | 19.71% | 19.32% | 34.95% |
| 7 | **Stacking Classifier (Ensemble)** | 86.48% | 16.44% | 17.13% | 18.97% |

---

## 🗂 Project Directory Structure
```text
ML/
├── Database/
│   └── controlled-fault-benchmark-v1.0/   # Raw Zenodo benchmark CSVs (1.81M samples)
├── data/
│   ├── raw/
│   └── processed/                         # train.csv and test.csv partitions
├── models/
│   ├── fault_category/                    # Trained category base & stacking models
│   ├── fault_severity/                    # Trained severity base & stacking models
│   └── preprocessors/                     # Fitted RobustScaler artifact
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis & visual profiling
│   ├── 02_preprocessing_and_splitting.ipynb# 5-scaler comparison (Standard, Robust, MinMax, Quantile, Power)
│   ├── 03_model_training_and_stacking.ipynb# Manual training & stacking evaluation
│   ├── colab_kaggle_quickstart.py         # 1-click cloud training script
│   └── run_1_8m_full_dataset.md           # Instructions for full 1.81M execution
├── reports/
│   ├── dataset_profile.md                 # 29-column schema & 16-to-9 taxonomy mapping
│   ├── full_1_8m_training_report.md       # Complete 1.81M training logs & benchmarks
│   ├── figures/                           # Normalized confusion matrices & comparison plots
│   └── metrics/                           # Exported CSV evaluation tables
├── app/
│   ├── dashboard_server.py                # Modern interactive zero-dependency dashboard
│   └── streamlit_app.py                   # Streamlit dashboard
├── src/
│   ├── config.py                          # Feature definitions, paths, and taxonomies
│   ├── feature_engineering.py             # S, Q, deviations, dynamic derivatives (dV/dt, dI/dt, dP/dt)
│   ├── pv_adapter.py                      # Solar Photovoltaic Inverter Adapter
│   ├── preprocessing.py                   # Curated dataset preprocessing pipeline
│   ├── preprocessing_full_1_8m.py         # Full 1.81 Million dataset loader & splitter
│   ├── train.py                           # 5 base models & stacking ensemble trainer
│   ├── train_full_1_8m.py                 # Full 1.81M scaled trainer (HistGB, Subsampled RF/ET, SGD, KNN)
│   └── evaluate.py                        # Model evaluator, confusion matrices & figures generator
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/Ashrit-Raghupatruni/solar-power-fault-stacking-ensemble.git
cd solar-power-fault-stacking-ensemble
pip install -r requirements.txt
```

### 2. Run Preprocessing on Full 1.81M Records
```bash
python -m src.preprocessing_full_1_8m
```

### 3. Train Models on 1.81M Records
```bash
python -m src.train_full_1_8m
```

### 4. Evaluate Models & Generate Figures
```bash
python -m src.evaluate
```

### 5. Launch the Web Dashboard
```bash
python app/dashboard_server.py
```
Open **`http://localhost:8501`** in your browser.
