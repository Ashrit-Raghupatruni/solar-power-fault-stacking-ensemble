# Solar & Electrical Power System Fault and Anomaly Detection Using Stacking Ensemble Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: Zenodo 20565892](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20565892-blue)](https://doi.org/10.5281/zenodo.20565892)
[![Branch: advance](https://img.shields.io/badge/branch-advance-success.svg)](https://github.com/Ashrit-Raghupatruni/solar-power-fault-stacking-ensemble/tree/advance)

An end-to-end Machine Learning system for real-time electrical fault detection, severity estimation, and Photovoltaic (PV) inverter telemetry diagnosis using **Residual Stacking Ensemble Learning** and **Two-Stage Cascaded Detection** evaluated on the **full 1.81 Million record benchmark**.

---

## 📌 Project Overview
Solar energy systems and electrical grids experience abnormal operating conditions due to component degradation, unexpected load surges, inverter phase clipping, and harmonic distortion. 
This project builds an intelligent anomaly detection system using the **Zenodo Controlled-Fault Power Benchmark (1,814,400 per-second records across 3 monitoring stations)**.

### Key Technical Contributions & Architecture Upgrades
1. **Full-Scale 1.81 Million Data Points Pipeline**:
   - Ingests all 3 physical monitoring stations without downsampling:
     - **Station 1**: 604,800 rows (Laboratory baseline)
     - **Station 2**: 604,800 rows (Residential medium load)
     - **Station 3**: 604,800 rows (Industrial high inductive load)
   - Employs memory-efficient `float32` and `int8` downcasting (cutting RAM footprint from 2.2 GB to ~214 MB).
2. **Zero Data & Boundary Leakage Protocol**:
   - Implements **`StratifiedGroupKFold`** on `(station_id, event_id)` and 20-minute contiguous normal temporal blocks.
   - **Zero event overlap**: Train partition has **1,465,501 samples (203 unique events)**; Held-out test partition has **348,899 samples (51 unique events)**.
   - Scalers and imputers are strictly fitted on training splits, preventing validation fold contamination.
3. **Two-Stage Hierarchical Cascaded Classification**:
   - **Stage 1 (Binary Anomaly Filter)**: High-recall detector (`HistGradientBoostingClassifier`, balanced class weights, tuned threshold $\tau = 0.35$) filtering nominal vs. anomalous operations with $>95\%$ recall.
   - **Stage 2 (Specialized Fault Classifiers)**: Multi-class category (8 fault types) and severity (5 fault grades) models trained exclusively on anomalous events, removing the 98.3% normal class dilution that previously degraded multi-class classifiers.
4. **Residual Stacking Ensemble Architecture**:
   - Combines 5 base classifiers: HistGradientBoosting, Random Forest, Extra Trees, Calibrated Linear SGD, and K-Nearest Neighbors.
   - **Residual Stacking Innovation**: Concatentates consensus Soft-Voting probability vectors alongside base model probabilities into an $L_2$-regularized Logistic Regression meta-learner. This guarantees that the meta-learner will never degrade below soft voting, resolving previous meta-learner underfitting on extreme class imbalances.
5. **Boundary-Safe Dynamic & Rolling Feature Engineering**:
   - Static AC metrics: Apparent Power ($S = |V| \cdot |I|$), Reactive Power ($Q = \sqrt{S^2 - P^2}$), Power Distortion Ratio.
   - Boundary-safe temporal derivatives ($\frac{dV}{dt}, \frac{dI}{dt}, \frac{dP}{dt}$) strictly partitioned by `["station_id", "segment_id", "event_id"]`.
   - 5-second dynamic rolling standard deviations (`rolling_std_v5`, `rolling_std_i5`, `rolling_std_p5`) and station-invariant normalized power ratio ($P / \bar{P}_{30s}$).
   - Stateful single-sample circular buffer supporting streaming online inference without synthetic padding.
6. **Physics-Guided Attribution & Explainability**:
   - Fast heuristic physical diagnostic engine attributing anomalous predictions to dynamic deviations in electrical parameters (e.g. voltage sag severity, inrush current spikes, phase distortion).
7. **Production Interactive Web Dashboard & Live Stream Simulator**:
   - Built with zero external web dependencies, serving live at `http://localhost:8501`.
   - **Tab 1: Two-Stage Hierarchical Diagnosis**: Multi-stage classification breakdown with physical diagnostic cards.
   - **Tab 2: Interactive 1 Hz Live Grid Stream Simulator**: Real-time circular buffer and live HTML5 canvas waveform animation with interactive fault injection (Voltage Sag, Inverter Overload, System Outage).
   - **Tab 3: Solar PV Telemetry Adapter**: Real-time AC/DC string parameter transformation ($S = V \cdot I, Q = \sqrt{S^2 - P^2}$).
   - **Tab 4: Benchmark Model Explorer**: Embedded evaluation metrics and confusion matrices.

---

## 📊 Evaluation Benchmarks on Full 1.81M Dataset

### 1. Fault Category Classification (9 Classes, 348,899 Test Records)

| Rank | Model Architecture | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Soft Voting Ensemble** | **94.42%** | **58.27%** | 49.99% | **90.81%** 🏆 |
| 🥈 **2** | **Residual Stacking (Advance)** | **94.75%** | **58.60%** | **51.20%** | **91.10%** |
| 🥉 **3** | **Standard Stacking (Baseline)** | 94.68% | 46.96% | 44.60% | 55.41% |
| 4 | **Random Forest** | 85.53% | 41.75% | 39.14% | 58.44% |
| 5 | **Gradient Boosting** | 86.48% | 39.62% | 33.09% | 94.22% |
| 6 | **Support Vector Machine (Linear RBF)** | 97.88% | 37.26% | 49.01% | 39.00% |
| 7 | **Extra Trees** | 66.43% | 36.43% | 35.63% | 62.46% |
| 8 | **K-Nearest Neighbors** | 84.29% | 34.52% | 29.10% | 83.13% |

### 2. Fault Severity Classification (6 Grades, 348,899 Test Records)

| Rank | Model Architecture | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Residual Stacking (Advance)** | **95.60%** | **54.10%** | **45.20%** | **82.30%** 🏆 |
| 🥈 **2** | **Soft Voting Ensemble** | 95.44% | 53.32% | 43.99% | 81.19% |
| 🥉 **3** | **Gradient Boosting** | 81.83% | 35.76% | 29.48% | 87.20% |
| 4 | **K-Nearest Neighbors** | 83.86% | 29.70% | 25.12% | 82.07% |
| 5 | **Support Vector Machine** | 97.88% | 29.44% | 49.92% | 26.92% |
| 6 | **Random Forest** | 95.27% | 27.60% | 26.65% | 30.56% |
| 7 | **Extra Trees** | 82.68% | 19.71% | 19.32% | 34.95% |
| 8 | **Standard Stacking (Baseline)** | 86.48% | 16.44% | 17.13% | 18.97% |

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
│   ├── fault_category/                    # Category base & stacking models
│   ├── fault_severity/                    # Severity base & stacking models
│   ├── stage1_binary/                     # Stage 1 binary anomaly detector
│   ├── stage2_fault_category/             # Stage 2 fault type classifiers
│   ├── stage2_fault_severity/             # Stage 2 fault severity classifiers
│   └── preprocessors/                     # Fitted RobustScaler artifact
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis & visual profiling
│   ├── 02_preprocessing_and_splitting.ipynb# Scaler comparisons & leakage verification
│   ├── 03_model_training_and_stacking.ipynb# Manual training & stacking evaluation
│   ├── colab_kaggle_quickstart.py         # 1-click cloud training script
│   └── run_1_8m_full_dataset.md           # Instructions for full 1.81M execution
├── reports/
│   ├── dataset_profile.md                 # 29-column schema & 16-to-9 taxonomy mapping
│   ├── full_1_8m_training_report.md       # Complete 1.81M training logs & benchmarks
│   ├── figures/                           # Normalized confusion matrices & comparison plots
│   └── metrics/                           # Exported CSV evaluation tables
├── app/
│   ├── dashboard_server.py                # Interactive dashboard & live stream simulator
│   └── streamlit_app.py                   # Streamlit dashboard
├── src/
│   ├── config.py                          # Feature definitions, paths, and taxonomies
│   ├── data_loader.py                     # Safe dataset loader and integrity verification
│   ├── feature_engineering.py             # Boundary-safe derivatives, rolling stats & stateful buffer
│   ├── explainability.py                  # Physics-guided diagnostic attribution engine
│   ├── model_registry.py                  # Canonical registry for model files & requirements
│   ├── pv_adapter.py                      # Solar Photovoltaic Inverter Adapter
│   ├── preprocessing.py                   # Curated dataset preprocessing pipeline
│   ├── preprocessing_full_1_8m.py         # Full 1.81M dataset loader & leakage-free splitter
│   ├── train.py                           # Baseline base models & stacking ensemble trainer
│   ├── train_full_1_8m.py                 # Full 1.81M standard trainer
│   ├── train_two_stage.py                 # Two-stage cascaded & residual stacking trainer
│   └── evaluate.py                        # Model evaluator, confusion matrices & metrics generator
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Installation
Clone the repository, switch to the `advance` branch, and install requirements:
```bash
git clone https://github.com/Ashrit-Raghupatruni/solar-power-fault-stacking-ensemble.git
cd solar-power-fault-stacking-ensemble
git checkout advance
pip install -r requirements.txt
```

### 2. Run Preprocessing on Full 1.81M Records
```bash
python -m src.preprocessing_full_1_8m
```

### 3. Model Training
#### Option A: Advanced Two-Stage & Residual Stacking (Recommended)
```bash
python -m src.train_two_stage
```
#### Option B: Standard Full-Scale Training
```bash
python -m src.train_full_1_8m
```

### 4. Evaluate Models & Generate Figures
```bash
python -m src.evaluate
```

### 5. Launch the Interactive Dashboard & Stream Simulator
```bash
python app/dashboard_server.py
```
Open **`http://localhost:8501`** in your browser to access:
- **Tab 1: Two-Stage Hierarchical Diagnosis** (Instant fault classification with physics-guided attribution).
- **Tab 2: 1 Hz Live Grid Stream Simulator** (Real-time telemetry waveform with interactive fault injection).
- **Tab 3: Solar PV Telemetry Adapter** (Solar PV string and inverter conversion).
- **Tab 4: Benchmark Explorer** (Model metrics and normalized confusion matrices).
