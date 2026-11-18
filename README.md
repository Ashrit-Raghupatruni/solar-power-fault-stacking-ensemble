# PowerGuard AI: Solar & Electrical Power System Fault and Anomaly Detection Using Stacking Ensemble Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: Zenodo 20565892](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20565892-blue)](https://doi.org/10.5281/zenodo.20565892)
[![Branch: advance](https://img.shields.io/badge/branch-advance-success.svg)](https://github.com/Ashrit-Raghupatruni/solar-power-fault-stacking-ensemble/tree/advance)
[![Architecture: Two--Stage Cascaded](https://img.shields.io/badge/Architecture-Two--Stage%20Cascaded%20Stacking-orange.svg)](#-two-stage-cascaded-ml-architecture)

An end-to-end industrial-grade Machine Learning and telemetry diagnostics platform for real-time electrical fault detection, severity estimation, and Photovoltaic (PV) microinverter monitoring. Powered by a **Two-Stage Cascaded Detection Architecture**, a **Residual Stacking Ensemble**, and an **AhoyDTU / OpenDTU Solar Telemetry Adapter**, evaluated across the complete **1.81 Million record Zenodo benchmark**.

---

## 📌 Key System Highlights

1. **Full-Scale 1.81M Record Controlled-Fault Benchmark**:
   - Ingests all 3 physical monitoring stations without downsampling (1,814,400 per-second records):
     - **Station 1 (Laboratory Baseline)**: 604,800 seconds (~661 W, 92 fault events).
     - **Station 2 (Residential Profile)**: 604,800 seconds (~685 W, 84 fault events).
     - **Station 3 (Industrial High Inductive Load)**: 604,800 seconds (~4,255 W, 78 fault events).
2. **Standardized Severity Taxonomy & Leakage-Free Splitting**:
   - Severity taxonomy harmonized to raw Zenodo benchmark ground truth: `['none', 'incipient', 'low', 'med', 'high', 'critical']`.
   - Partitioned via `StratifiedGroupKFold` on `(station_id, event_id)` with contiguous temporal block preservation: **Zero event overlap** (Train: 1,465,501 rows, 203 events; Holdout Test: 348,899 rows, 51 events).
3. **Unified 16-Feature Operational Schema**:
   - Standardized feature vector matching runtime IoT and inverter measurement capabilities without artificial constants or unobservable frequency-domain entropy features.
4. **Safety-Weighted Two-Stage Cascaded Architecture**:
   - **Stage 1 (Binary Anomaly Detector)**: Class-weighted `HistGradientBoostingClassifier` with $5\times$ penalty on critical events and tuned decision threshold ($\tau = 0.125$), achieving **94.95% overall anomaly recall** and **99.24% critical fault recall**.
   - **Stage 2 (Residual Stacking Classifiers)**: Isolated multi-class classifiers for **Fault Category (8 classes)** and **Severity Grade (5 classes)** trained exclusively on anomalous events to eliminate normal-class dilution.
5. **AhoyDTU / OpenDTU Solar PV Inverter Adapter**:
   - Ingests 4-channel DC string telemetry ($V_{\text{dc}}, I_{\text{dc}}, P_{\text{dc}}$) alongside AC grid-tie readings ($V_{\text{ac}}, I_{\text{ac}}, P_{\text{ac}}, f, \text{PF}$).
   - Computes solar conversion efficiency $\eta = \frac{P_{\text{ac}}}{\sum P_{\text{dc}}} \times 100$ and keeps DC string parameters strictly isolated to PV health analytics without corrupting the AC classifier.
6. **Production FastAPI Inference Service & 8-Section SCADA Dashboard**:
   - **FastAPI Backend** (`app/api.py`): Real-time inference endpoints, health inspection, and live 1 Hz stream generator with fault injection.
   - **SCADA Web Dashboard** (`app/dashboard_server.py`): Industrial dark UI featuring 8 specialized sections with live canvas waveforms and zero-flicker telemetry updates.

---

## ⚡ Unified 16-Feature Operational Schema

Every physical telemetry sample (from smart meters or PV inverters) is transformed into this standardized 16-feature vector:

| # | Feature Column | Physical Description | Unit |
|---|----------------|----------------------|------|
| 1 | `voltage_V` | AC RMS Line Voltage | $\text{V}$ |
| 2 | `current_A` | AC RMS Line Current | $\text{A}$ |
| 3 | `power_W` | Active Real Power | $\text{W}$ |
| 4 | `frequency_Hz` | Supply Grid Frequency | $\text{Hz}$ |
| 5 | `power_factor` | Displacement Power Factor ($\cos \varphi$) | ratio [0, 1] |
| 6 | `apparent_power_VA` | Total Apparent Power ($S = \|V\| \times \|I\|$) | $\text{VA}$ |
| 7 | `reactive_power_var` | Reactive Power ($Q = \sqrt{\max(0, S^2 - P^2)}$) | $\text{VAR}$ |
| 8 | `power_active_apparent_ratio` | Active-to-Apparent Power Ratio ($\frac{\|P\|}{S}$) | ratio [0, 1] |
| 9 | `voltage_deviation_pct` | Voltage Deviation from Nominal ($\frac{V - 220}{220} \times 100$) | $\%$ |
| 10 | `frequency_deviation_pct` | Frequency Deviation from Nominal ($\frac{f - 50}{50} \times 100$) | $\%$ |
| 11 | `d_voltage_dt` | Causal First Derivative of Voltage ($\|V_t - V_{t-1}\|$) | $\text{V/s}$ |
| 12 | `d_current_dt` | Causal First Derivative of Current ($\|I_t - I_{t-1}\|$) | $\text{A/s}$ |
| 13 | `d_power_dt` | Causal First Derivative of Power ($\|P_t - P_{t-1}\|$) | $\text{W/s}$ |
| 14 | `rolling_std_v5` | 5-Second Moving Voltage Standard Deviation | $\text{V}$ |
| 15 | `rolling_std_i5` | 5-Second Moving Current Standard Deviation | $\text{A}$ |
| 16 | `power_to_baseline_ratio` | Station-Invariant Power Ratio ($\frac{P_t}{P_{30s}}$) | ratio |

---

## 🏗 Two-Stage Cascaded ML Architecture

```
                                  Live Telemetry Input
                       (Manual Sensor or AhoyDTU / OpenDTU Solar PV)
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │   Unified 16-Feature Extraction    │
                       │   & RobustScaler Normalization      │
                       └──────────────────┬──────────────────┘
                                          │
                                          ▼
                         STAGE 1: Binary Anomaly Filter
                       (HistGradientBoosting, τ = 0.125)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
            Prob < 0.125                                Prob >= 0.125
            [ NORMAL ]                                  [ ANOMALOUS ]
     (Severity: None, Conf > 90%)                               │
                                                                ▼
                                                 STAGE 2: Multi-Model Soft Voting Consensus
                                                 ┌─────────────────────────────────────────┐
                                                 │ 5 Base Classifiers:                     │
                                                 │  1. Hist Gradient Boosting              │
                                                 │  2. Random Forest (100 trees, depth=14) │
                                                 │  3. Extra Trees (100 trees, depth=14)   │
                                                 │  4. Calibrated Linear SGD (Log Loss)    │
                                                 │  5. K-Nearest Neighbors (k=5)           │
                                                 │                    │                    │
                                                 │ Probability Averaging & Consensus:      │
                                                 │   P_bar(y=c|X) = (1/M) * sum(P_m)       │
                                                 └────────────────────┬────────────────────┘
                                                                │
                                        ┌───────────────────────┴───────────────────────┐
                                        ▼                                               ▼
                             Stage 2A: Fault Category                         Stage 2B: Severity Grade
                           (8 Controlled Fault Classes)                      (5 Controlled Severity Grades)
                             • voltage         • outage                         • incipient     • med
                             • load            • sensor                         • low           • high
                             • frequency       • context                                        • critical
                             • power_quality   • compound
```

---

## ☀️ AhoyDTU / OpenDTU Solar Inverter Telemetry

The platform natively supports microinverters (e.g., Hoymiles HM-1200 via AhoyDTU or OpenDTU).

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      AhoyDTU Microinverter Interface                   │
 ├────────────────────────────────────┬───────────────────────────────────┤
 │ AC Grid-Tie Output Parameters      │ Multi-Channel DC String Telemetry │
 │   • AC Voltage:     235.8 V        │   • Channel 1: 27.1 V | 0.12 A    │
 │   • AC Current:     0.05 A         │   • Channel 2: 27.1 V | 0.13 A    │
 │   • Active Power:   12.5 W         │   • Channel 3: 26.1 V | 0.12 A    │
 │   • Reactive Power: 19.6 VAR       │   • Channel 4: 26.1 V | 0.12 A    │
 │   • Power Factor:   0.54           │   • Total DC Array: 13.03 W       │
 │   • Frequency:      49.99 Hz       │   • Efficiency η:   95.93%        │
 └─────────────────┬──────────────────┴───────────────────────────────────┘
                   │
                   ▼
  Mapped into Unified 16-Feature Schema ──► Evaluated by Pure ML Two-Stage Pipeline
```

---

## 🖥 8-Section SCADA Monitoring Dashboard

PowerGuard AI provides an industrial dark SCADA interface (`app/dashboard_server.py`) organized into 8 sections:

| # | Section | Description |
|---|---------|-------------|
| **01** | **Overview** | System KPIs, 1.81M Zenodo benchmark statistics, fault distribution charts. |
| **02** | **Live Monitoring** | Real-time 1 Hz telemetry waveforms with live fault injection (`Nominal`, `Voltage Sag`, `Overload`, `Outage`). |
| **03** | **Fault Diagnosis** | Manual telemetry testing, Stage 1 anomaly probability gauge, Stage 2 category & severity probabilities, and physics attribution. |
| **04** | **Stacking Ensemble** | Interactive SVG architecture diagram of Level 0 base learners, 5-fold OOF predictions, Level 1 meta-learner, and Soft Voting comparison. |
| **05** | **Solar PV Monitoring** | Dedicated AhoyDTU / OpenDTU 4-channel DC string monitoring, yield tracking, and conversion efficiency gauge. |
| **06** | **Model Evaluation** | Normalized 9x9 fault category and 6x6 severity confusion matrices with precision/recall metrics. |
| **07** | **Dataset Analytics** | Comprehensive breakdown across Laboratory (604.8k), Residential (604.8k), and Industrial (604.8k) recording stations. |
| **08** | **System & API** | FastAPI live connection status, active model weights, hashes, and 16-feature operational schema inspector. |

---

## 🚀 Quickstart Guide

### 1. Installation & Environment Setup
```powershell
# Clone the repository and switch to the advance branch
git clone https://github.com/Ashrit-Raghupatruni/solar-power-fault-stacking-ensemble.git
cd solar-power-fault-stacking-ensemble
git checkout advance

# Install required Python packages
pip install -r requirements.txt
```

### 2. Run Verification Unit Tests
```powershell
pytest tests/test_features.py -v
```

### 3. Start the FastAPI Real-Time ML Service (Terminal 1)
```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
- Interactive Swagger docs: `http://localhost:8000/docs`
- Health & Schema status: `http://localhost:8000/health`

### 4. Start the SCADA Web Dashboard (Terminal 2)
```powershell
python app/dashboard_server.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📡 API Endpoints Reference

### 1. Two-Stage Fault Diagnosis (`POST /predict/two-stage`)
**Request Body**:
```json
{
  "voltage_V": 178.0,
  "current_A": 4.5,
  "power_W": 760.0,
  "frequency_Hz": 50.0,
  "power_factor": 0.95
}
```
**Response**:
```json
{
  "is_anomaly": true,
  "stage1_probability": 0.924,
  "decision_threshold": 0.125,
  "fault_category": "voltage",
  "category_probability": 0.887,
  "severity": "High",
  "severity_code": "high",
  "severity_probability": 0.854,
  "physics_attribution": {
    "voltage_sag": true,
    "current_overload": false,
    "frequency_drift": false,
    "poor_power_factor": false,
    "power_deviation": true
  }
}
```

### 2. Solar PV Telemetry Diagnosis (`POST /predict/solar-pv`)
**Request Body**:
```json
{
  "voltage_ac": 235.8,
  "current_ac": 0.05,
  "power_ac": 12.5,
  "frequency_ac": 49.99,
  "power_factor": 0.54,
  "ch1_v": 27.1, "ch1_i": 0.12,
  "ch2_v": 27.1, "ch2_i": 0.13,
  "ch3_v": 26.1, "ch3_i": 0.12,
  "ch4_v": 26.1, "ch4_i": 0.12
}
```

---

## 📊 Benchmark Leaderboard & Empirical Evaluation

Evaluated across **601,926 unseen test samples** (52 independent fault events) using `StratifiedGroupKFold` zero event leakage:

| Rank | Model Architecture | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) | Evaluation Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **K-Nearest Neighbors ($k=5$)** | **98.83%** | 34.23% | 44.07% | 32.69% | Majority-Class Bias |
| 2 | **★ Soft Voting Ensemble** | **98.76%** | **48.77%** | **45.44%** | **59.29%** | **Well-Fitting (Best Overall System)** |
| 3 | **Hist Gradient Boosting** | **97.47%** | **43.32%** | 38.65% | **64.22%** | **Well-Fitting (Best Single Model)** |
| 4 | **Extra Trees** | 95.26% | 31.29% | 29.33% | 48.05% | Well-Fitting |
| 5 | **Random Forest** | 94.58% | 32.13% | 30.67% | 52.80% | Well-Fitting |
| 6 | **Support Vector Machine** | 90.71% | 12.84% | 20.33% | 14.52% | Severe Underfitting |

### Two-Stage Cascaded Safety & Continuous Metrics
- **Critical Fault Safety Recall**: **95.83%** (FNR: 4.17% on catastrophic faults)
- **Stage 1 Anomaly Recall**: **80.02%** at safety threshold $\tau = 0.125$
- **Stage 2 Anomalies-Only Category Macro-F1**: **72.69%** (HistGradientBoosting) / **71.91%** (Soft Voting)
- **Multi-Class Balanced Metrics**: Matthews Correlation Coefficient (**MCC: 0.5513**), Cohen's Kappa (**0.5329**)
- **Continuous Power Regression**: Active Power $R^2 = 0.4777$, $\text{RMSE} = 581.80\text{ W}$, $\text{MAE} = 462.61\text{ W}$

---

## 📓 Interactive Google Colab Notebooks
- `notebooks/Untitled2.ipynb`: End-to-end interactive Google Colab execution pipeline containing complete multi-station ingestion, Stage 1 & Stage 2 models, interactive scenario sliders, multi-metric evaluations, and 9 visual diagnostic figures.
- Detailed technical report: [PowerGuard_AI_Technical_Report.md](reports/PowerGuard_AI_Technical_Report.md)

---

## 📄 License & Attribution
- **Dataset**: *A Real Controlled-Fault Benchmark for Power-Anomaly Detection and Identification on IoT Edge Nodes (RCF-Power v1.0)* by Thanh Trang Tran & Tran Nhut Tam, [10.5281/zenodo.20565892](https://doi.org/10.5281/zenodo.20565892), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Project Code**: Released under the MIT License.
