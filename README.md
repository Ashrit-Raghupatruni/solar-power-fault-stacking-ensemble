# Solar & Electrical Power System Fault and Anomaly Detection Using Stacking Ensemble Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end Machine Learning system for real-time electrical fault detection, severity estimation, and photovoltaic (PV) inverter monitoring using **Stacking Ensemble Learning**.

---

## 📌 Project Overview
Solar energy systems and electrical grids experience abnormal operating conditions due to wiring degradation, load surges, inverter malfunctions, and harmonic distortion. 
This project builds an intelligent anomaly detection system using the **Zenodo Controlled-Fault Power Benchmark (1.81M records across 3 monitoring stations)**.

### Key Technical Contributions
1. **Zero Data Leakage**: Utilizes **Stratified Group K-Fold** splitting on `(station_id, event_id)` so complete contiguous fault intervals are never divided across train and test partitions.
2. **Dual Independent Prediction Pipelines**:
   - **Fault Category Classifier**: 8 multiclass categories (`voltage`, `load`, `power_quality`, `sensor`, `outage`, `frequency`, `context`, `compound`).
   - **Fault Severity Classifier**: 5 severity grades (`incipient`, `low`, `med`, `high`, `critical`).
3. **Model Diversity & Stacking Architecture**:
   - **5 Base Classifiers**: Random Forest, Extra Trees, Gradient Boosting, K-Nearest Neighbors, and Support Vector Machine.
   - **Ensemble Layer**: Soft Voting, Hard Voting, and Stacking Classifier using **Out-Of-Fold (OOF)** predictions with a **Logistic Regression Meta-Model**.
   - Model-specific preprocessing: Scaling (`StandardScaler`) strictly isolated to distance/kernel models (KNN, SVM), preserving unscaled dynamics for tree ensembles.
4. **Solar PV Telemetry Adapter**: Translates inverter electrical parameters (AC voltage, current, active/reactive power, power factor) into standardized features without fabricating missing sensor channels.
5. **Interactive Dashboard**: Streamlit interface supporting single-sample diagnosis, PV adapter mode, batch CSV timeline visualization, and model comparison.

---

## 🗂 Project Directory Structure
```
ML/
├── Database/
│   └── controlled-fault-benchmark-v1.0/   # Raw Zenodo benchmark CSVs
├── data/
│   ├── raw/
│   └── processed/                         # Leakage-free train.csv and test.csv
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis & visual profiling
│   ├── 02_preprocessing_and_splitting.ipynb# Zero-leakage group splitting & verification
│   └── 03_model_training_and_stacking.ipynb# Manual training & stacking evaluation
├── models/
│   ├── fault_category/                    # Serialized models for Category classification
│   ├── fault_severity/                    # Serialized models for Severity classification
│   └── preprocessors/                     # Trained StandardScaler artifact
├── src/
│   ├── config.py                          # Features, targets, label maps, paths
│   ├── feature_engineering.py             # Apparent/reactive power, THD, deviations
│   ├── pv_adapter.py                      # Solar Photovoltaic Inverter Adapter
│   ├── preprocessing.py                   # StratifiedGroupKFold data preparation
│   ├── train.py                           # 5 base models & OOF stacking meta-model
│   ├── evaluate.py                        # Macro-F1, ROC-AUC, confusion matrices
│   └── build_notebooks.py                 # Notebook generation script
├── app/
│   └── streamlit_app.py                   # Interactive web application
├── reports/
│   ├── dataset_profile.md                 # Full 29-column schema & class profile
│   └── metrics/                           # Comparative model benchmark logs
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Preprocessed Data & Leakage-Free Splitting
Run the preprocessing pipeline to generate the curated, leakage-free train and test datasets:
```bash
python -m src.preprocessing
```

### 2. Interactive Jupyter Notebooks
Open Jupyter and run the notebooks in sequence:
- `notebooks/01_eda.ipynb`: Explore signal distributions and sensor correlation heatmaps.
- `notebooks/02_preprocessing_and_splitting.ipynb`: Verify zero event overlap between train and test.
- `notebooks/03_model_training_and_stacking.ipynb`: Manually run base models and stacking training at your own pace.

### 3. Launch the Interactive Web Dashboard
Run the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```
