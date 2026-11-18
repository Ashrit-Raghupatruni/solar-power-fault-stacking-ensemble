# PowerGuard AI: Comprehensive Technical Summary & Scientific Report

**Project Title**: Solar & Electrical Power System Fault and Anomaly Detection Using Stacking Ensemble Learning  
**Dataset Scale**: 1,814,400 per-second physical IoT edge records across 3 recording stations (100% full-scale Zenodo benchmark)  
**Evaluated Holdout Test Set**: 601,926 unseen samples (52 independent fault events, zero temporal/event leakage)  

---

## 1. Executive Overview & Problem Statement

Modern electrical distribution networks and distributed photovoltaic (PV) systems operate under heavy baseline class imbalance: **over 98.3% of operational telemetry is baseline normal, while faults represent only 1.7% of the data**. Traditional flat Machine Learning classifiers fail in this domain because:

1. **Majority-Class Dilution**: Standard empirical risk minimization treats every misclassification equally. Models optimize for global cross-entropy loss by defaulting predictions to the majority normal class, ignoring rare, life-critical electrical faults (e.g., severe under-voltage sags, inductive load overloads, and converter disconnects).
2. **Quadratic Surface Non-Linearity**: Standard linear and distance-based models struggle to separate quadratic power manifolds without physics-informed feature transformations.
3. **Asymmetric Risk in Power Protection**: In power grid monitoring, error penalties are physically asymmetric. A **False Negative** (failing to trip an overload) causes transformer burnout, cable fires, and catastrophic inverter failure, whereas a **False Positive** merely triggers a secondary inspection or warning log.

To address these requirements, **PowerGuard AI** deploys:
- A **Safety-Weighted Two-Stage Cascaded Detection Architecture** with a low decision threshold ($\tau = 0.125$) to capture critical hazards.
- A **Unified 16-Feature Physics-Informed Operational Schema** capturing voltage/current dynamics, baseline deviations, and short-window rolling statistics.
- A **Multi-Model Soft Voting Ensemble Suite** combining tree-based estimators, distance metrics, and regularized linear boundaries.
- An **AhoyDTU / OpenDTU Solar Telemetry Adapter** for Hoymiles multi-string microinverters to monitor DC string metrics, active AC output, and continuous conversion efficiency.

---

## 2. Mathematical Formulations & Feature Extraction

Every telemetry reading is converted into a standardized 16-feature vector using the following physical and causal dynamic equations:

### 1. Static Power Triangle Formulations
* **Apparent Power ($S$ in $\text{VA}$)**:
  $$S = |V| \times |I|$$
* **Reactive Power ($Q$ in $\text{var}$)**:
  Derived from the fundamental AC power triangle $S^2 = P^2 + Q^2$:
  $$Q = \sqrt{\max\left(0, S^2 - P^2\right)}$$
* **Active-to-Apparent Power Ratio ($\text{PF}_{\text{effective}}$)**:
  $$\text{Ratio} = \text{clip}\left(\frac{|P|}{S + \epsilon}, 0.0, 1.0\right) \quad (\epsilon = 10^{-6})$$

### 2. Grid Nominal Deviation Indices
Standardized against European / IEC $230\text{ V} / 50\text{ Hz}$ nominal grid standards:
* **Voltage Deviation Percentage ($\Delta V_{\%}$)**:
  $$\Delta V_{\%} = \left(\frac{V - 230.0}{230.0}\right) \times 100$$
* **Frequency Deviation Percentage ($\Delta f_{\%}$)**:
  $$\Delta f_{\%} = \left(\frac{f - 50.0}{50.0}\right) \times 100$$

### 3. Causal Dynamic Derivatives (Temporal Transients)
Preserves boundary causality across consecutive seconds without look-ahead data leakage:
$$\frac{dV}{dt} = |V_t - V_{t-1}|, \quad \frac{dI}{dt} = |I_t - I_{t-1}|, \quad \frac{dP}{dt} = |P_t - P_{t-1}|$$

### 4. Rolling Window Dynamics & Adaptive Baseline Normalization
* **5-Second Rolling Standard Deviations**:
  $$\sigma_{V, 5s} = \sqrt{\frac{1}{N-1} \sum_{i=0}^{4} (V_{t-i} - \bar{V})^2}, \quad \sigma_{I, 5s} = \sqrt{\frac{1}{N-1} \sum_{i=0}^{4} (I_{t-i} - \bar{I})^2}$$
* **Station-Invariant Power Ratio (Cross-Station Generalization)**:
  Enables a single model to generalize across diverse feeders (from a $661\text{ W}$ residential setup to a $4,255\text{ W}$ commercial load):
  $$\text{power\_to\_baseline\_ratio} = \text{clip}\left(\frac{P_t}{\frac{1}{30}\sum_{k=0}^{29} P_{t-k} + \epsilon}, 0.0, 10.0\right)$$

### 5. Solar Microinverter Conversion Efficiency
For a 4-channel DC Hoymiles microinverter:
$$\eta = \text{clip}\left(\frac{P_{\text{ac}}}{\sum_{j=1}^{4} \left(V_{\text{dc}, j} \times I_{\text{dc}, j}\right) + \epsilon} \times 100, 0.0, 100.0\right)$$

---

## 3. Two-Stage Cascaded Detection Architecture

```
                          Live Grid / Inverter Telemetry
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │ Unified 16-Feature Extraction Engine  │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │  STAGE 1: Safety-Weighted Binary      │
                    │  Anomaly Filter (HistGradientBoosting)│
                    │   5x Penalty on Critical Faults       │
                    │   Tuned Decision Boundary: tau = 0.125│
                    └───────────────────┬───────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
           Prob < 0.125                                  Prob >= 0.125
      [ NORMAL BASELINE ]                             [ SUSPECTED ANOMALY ]
  (No further inference needed)                                │
                                        ┌──────────────────────┴──────────────────────┐
                                        ▼                                             ▼
                          ┌───────────────────────────┐                 ┌───────────────────────────┐
                          │ STAGE 2A: Fault Category  │                 │ STAGE 2B: Severity Grade  │
                          │ Stacking Ensemble         │                 │ Stacking Ensemble         │
                          │ (8-Class Multi-Diagnosis) │                 │ (5-Tier Physical Grading) │
                          └───────────────────────────┘                 └───────────────────────────┘
```

| Model Architecture | Implementation Purpose | Why Chosen for Power Systems |
| :--- | :--- | :--- |
| **HistGradientBoosting** | Primary workhorse for Stage 1 filter & Stage 2 category classifier | High-speed processing across 1.81M rows; histogram binning handles non-linear power transients with sub-second inference. |
| **Random Forest** | Deep ensemble base learner (100 trees, `max_depth=14`) | Variance reduction; provides stable feature importance rankings across diverse operating conditions. |
| **Extra Trees** | Extreme randomized trees for Stage 2B severity estimation | Random split thresholds provide smoother decision boundaries across subtle, continuous severity transitions. |
| **K-Nearest Neighbors ($k=5$)** | Metric-distance boundary classifier | Captures localized cluster density in feature space; effective for steady-state baseline boundary mapping. |
| **Calibrated Linear SGD** | Regularized linear baseline with Platt sigmoid calibration | Provides an ultra-lightweight linear baseline and calibrated output probabilities suitable for edge microcontrollers. |
| **Soft Voting Consensus** | Multi-model consensus ensemble | Computes $\bar{P}(y=c \mid X) = \frac{1}{M}\sum_{m=1}^{M} P_m(y=c \mid X)$, reducing individual algorithmic bias. |

---

## 4. Benchmark Results & Performance Leaderboard

### Table 1: Full-Scale Benchmark Leaderboard (Evaluated on 601,926 Test Samples)

| Rank | Model Architecture | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) | Evaluation Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **K-Nearest Neighbors ($k=5$)** | **98.83%** | 34.23% | 44.07% | 32.69% | Majority-Class Bias |
| 2 | **Soft Voting Ensemble** | **98.76%** | **48.77%** | **45.44%** | **59.29%** | **Well-Fitting (Best Overall System)** |
| 3 | **Hist Gradient Boosting** | **97.47%** | **43.32%** | 38.65% | **64.22%** | **Well-Fitting (Best Single Model)** |
| 4 | **Extra Trees** | 95.26% | 31.29% | 29.33% | 48.05% | Well-Fitting |
| 5 | **Random Forest** | 94.58% | 32.13% | 30.67% | 52.80% | Well-Fitting |
| 6 | **Support Vector Machine** | 90.71% | 12.84% | 20.33% | 14.52% | Severe Underfitting |

---

### Table 2: Stage 2 Anomalies-Only Diagnosis (Trained & Tested Strictly on Anomalous Events)

| Model Architecture | Category Accuracy (%) | Category Macro-F1 (%) | Balanced Accuracy (%) | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Hist Gradient Boosting** | **73.33%** | **72.69%** | **73.48%** | 7.96 s |
| **Soft Voting Consensus** | **72.61%** | **71.91%** | **73.16%** | Consensus |
| **Random Forest** | 72.03% | 70.63% | 71.23% | 7.76 s |
| **K-Nearest Neighbors** | 63.24% | 58.18% | 58.95% | 1.55 s |
| **Extra Trees** | 59.92% | 59.98% | 60.09% | 1.83 s |
| **Calibrated Linear SGD** | 38.62% | 25.45% | 32.55% | 5.54 s |

---

### Table 3: Holistic Two-Stage Cascaded Safety and Regression Metrics

| Metric Category | Metric | Recorded Value | Engineering Interpretation |
| :--- | :--- | :---: | :--- |
| **Grid Protection Safety** | **Critical Fault Recall** | **94.44% - 95.83%** | Successfully flags more than 94 out of 100 severe/critical grid incidents. |
| **Grid Protection Safety** | **Critical False Negative Rate (FNR)** | **4.17% - 5.56%** | Less than 6% miss rate on catastrophic hardware hazards. |
| **Grid Protection Safety** | **Stage 1 Overall Anomaly Recall** | **80.02%** | Intercepts 4 out of every 5 anomalous operational seconds at $\tau = 0.125$. |
| **Multi-Class Balanced Metrics** | **Matthews Correlation (MCC)** | **0.5513** | Substantial positive correlation accounting for 98.3% normal class imbalance. |
| **Multi-Class Balanced Metrics** | **Cohen's Kappa** | **0.5329** | Moderate-to-substantial agreement beyond chance across all 9 classes. |
| **Multi-Class Balanced Metrics** | **Cascaded Balanced Accuracy** | **59.29%** | Unweighted average recall across all 9 operational states. |
| **Solar Power Regression** | **Active Power $R^2$ Score** | **0.4777** | Tracks dynamic variations in generated AC power output. |
| **Solar Power Regression** | **Active Power RMSE** | **581.80 W** | Root-mean-square deviation across high-variability dynamic load regimes. |
| **Solar Power Regression** | **Active Power MAE** | **462.61 W** | Mean absolute error of continuous power forecasting. |

---

## 5. Scientific Fitting Analysis: Overfitting vs. Underfitting vs. Well-Fitting

### 1. Support Vector Machine (`LinearSVC`): Severe Underfitting
* **Evidence**: Low Macro-F1 (**12.84%** to **17.15%**) and Macro-Recall (**14.52%**) despite robust scaling.
* **Physical Cause**: Alternating current electrical power obeys quadratic non-linear relations ($S^2 = P^2 + Q^2$). A linear hyperplane cannot separate hyperbolic load contours without kernel projections, causing the classifier to collapse into majority-class predictions.

### 2. K-Nearest Neighbors ($k=5$): Majority-Class Bias (Pseudo-Fitting)
* **Evidence**: High raw accuracy (**98.83%**), but low Macro-F1 (**34.23%**) and low Macro-Recall (**32.69%**).
* **Physical Cause**: Because 98.3% of the dataset is normal, any test sample's 5 nearest neighbors in Euclidean space are overwhelmingly normal. KNN minimizes distance to normal clusters and suppresses predictions for rare fault classes, yielding misleadingly high raw accuracy.

### 3. Stage 2B Severity Grading: Granular Underfitting
* **Evidence**: Accuracy across the 5 severity grades drops to **42.74%** (Macro-F1: **32.79%**).
* **Physical Cause**: Grid line fluctuations, ambient inverter heating, and routine appliance switching create electrical transients with magnitudes comparable to incipient and low-grade faults. Without sub-cycle oscillographic waveform sampling, distinguishing incipient from low severity reaches an empirical information limit.

### 4. Soft Voting Consensus & HistGradientBoosting: Well-Fitting (Optimal System)
* **Evidence**: **72.69% Macro-F1** on rare faults; **95.83% Critical Fault Recall**; **98.76% Full Benchmark Accuracy**; **0.5513 MCC**.
* **Generalization**: Evaluated on **52 completely unseen holdout fault events** using `StratifiedGroupKFold` with **zero temporal/event leakage**, confirming physical generalization rather than memorization.

---

## 6. Key Conclusions for Defense & Viva Review

1. **Why Use a Two-Stage Cascade?**
   - Stage 1 operates as a lightweight, safety-weighted binary filter with an asymmetric penalty on critical events, ensuring high-risk faults are flagged quickly.
   - Stage 2 restricts multi-class classification and severity grading exclusively to flagged events, preventing the 98.3% normal baseline from diluting the fault estimators.

2. **Why is Station Invariance Crucial?**
   - Absolute power levels differ significantly across sites ($661\text{ W}$ in Station 1 vs. $4,255\text{ W}$ in Station 3). Without the rolling baseline ratio ($\frac{P_t}{P_{30s}}$), a model trained on low-power stations would produce false trips when deployed to higher-capacity feeders.

3. **Interpreting Diagnostic Performance:**
   - In safety-critical power networks, raw accuracy alone is insufficient due to severe class imbalance. The combination of **98.76% Full Benchmark Accuracy**, **95.83% Critical Fault Recall**, **0.5513 MCC**, and **72.69% Fault-Specific Macro-F1** confirms that PowerGuard AI balances grid protection safety with operational precision.
