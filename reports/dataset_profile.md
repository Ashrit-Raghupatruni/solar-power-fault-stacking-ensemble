# Phase 1: Dataset Investigation & Profile Report

**Dataset**: A Real Controlled-Fault Benchmark for Power-Anomaly Detection and Identification on IoT Edge Nodes (RCF-Power v1.0)  
**Authors**: Thanh Trang Tran (HUIT) & Tran Nhut Tam (HCMUT, VNU-HCM)  
**DOI**: [10.5281/zenodo.20565892](https://doi.org/10.5281/zenodo.20565892)  
**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Temporal Coverage**: 7 continuous 24-hour days per station @ 1 Hz  
- Start: `2026-02-09 00:00:00`  
- End: `2026-02-15 23:59:59` (Inclusive, 604,800 seconds per station)  
**Total Dataset Scale**: 1,814,400 per-second observations across 3 monitoring stations.

---

## 1. Station Composition & Sample Distribution

| Station ID | Installation & Environment Profile | Nominal Circuit Capacity | Total Seconds | Normal Samples | Anomaly Samples | Anomaly Prevalence (% of Station) | Distinct Fault Events |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Station 1** | Laboratory / Normal Benchmark | ~661 W | 604,800 | 593,970 | 10,830 | 1.790% | 92 |
| **Station 2** | Residential / Medium Load Profile | ~685 W | 604,800 | 594,357 | 10,443 | 1.727% | 84 |
| **Station 3** | Industrial / High Inductive Load | ~4,255 W | 604,800 | 595,931 | 8,869 | 1.466% | 78 |
| **Total** | Multi-Environment Benchmark | — | **1,814,400** | **1,784,258** | **30,142** | **1.661%** | **254** |

---

## 2. Definitive Target Taxonomies & Class Cardinalities

To ensure complete scientific rigor and eliminate ambiguity across documentation and models:

### A. Fault Category Taxonomy
- **8 Controlled Fault Categories (Excluding Normal)**: `voltage`, `load`, `frequency`, `power_quality`, `outage`, `sensor`, `context`, `compound`.
- **9 Total Classes for Direct Classification (Including Normal)**:
  `['normal', 'voltage', 'load', 'frequency', 'power_quality', 'outage', 'sensor', 'context', 'compound']`
- *Architectural Standard*: The pipeline trains on **9 classes** (Normal + 8 fault categories). When discussing fault identification exclusively on diagnosed anomalies, the 8 fault categories are reported.

### B. Fault Severity Taxonomy
- **5 Controlled Severity Grades (Excluding None)**: `incipient`, `low`, `med`, `high`, `critical`.
- **6 Total Classes for Full Severity Assignment (Including None)**:
  `['none', 'incipient', 'low', 'med', 'high', 'critical']`
- `none` is the designated ground-truth severity corresponding to `normal` baseline operation.

---

## 3. Verified Class Distribution & Denominators

The table below explicitly reports class prevalence against **both** denominators:
1. **% of Entire Dataset** ($N = 1,814,400$)
2. **% of Anomalous Population** ($N = 30,142$)

### Fault Category Distribution

| Category | Description / Physical Phenomena | Total Count | % of Entire Dataset ($N = 1,814,400$) | % of Anomalous Subset ($N = 30,142$) |
| :--- | :--- | :--- | :--- | :--- |
| **`normal`** | Baseline operation within statutory grid bounds | 1,784,258 | 98.339% | — |
| **`load`** | Overload, sudden inrush surge, creeping load drift | 6,267 | 0.345% | 20.792% |
| **`sensor`** | Sensor dropout, register freeze (stuck), loose wiring | 4,897 | 0.270% | 16.246% |
| **`power_quality`** | Harmonic distortion, low inductive power factor | 4,357 | 0.240% | 14.455% |
| **`context`** | Off-schedule / contextual load spikes | 4,282 | 0.236% | 14.206% |
| **`compound`** | Concurrent multi-source anomalies | 4,105 | 0.226% | 13.619% |
| **`voltage`** | Voltage sags, swells, and rapid flicker | 2,984 | 0.164% | 9.900% |
| **`outage`** | Total supply trip / opened circuit breaker | 2,174 | 0.120% | 7.213% |
| **`frequency`** | Grid frequency stepping away from 50.0 Hz | 1,076 | 0.059% | 3.570% |
| **Total** | 9-Class Taxonomy | **1,814,400** | **100.000%** | **100.000%** |

### Fault Severity Grade Distribution

| Severity Grade | Operational Interpretation | Total Count | % of Entire Dataset ($N = 1,814,400$) | % of Anomalous Subset ($N = 30,142$) |
| :--- | :--- | :--- | :--- | :--- |
| **`none`** | Nominal baseline / zero fault severity | 1,784,258 | 98.339% | — |
| **`high`** | Severe degradation, significant voltage/load stress | 8,407 | 0.463% | 27.891% |
| **`incipient`** | Subtle early-stage parameter drift / minor distortion | 6,959 | 0.384% | 23.087% |
| **`med`** | Noticeable excursion requiring monitoring | 5,969 | 0.329% | 19.803% |
| **`low`** | Minor deviation within equipment safety margin | 5,550 | 0.306% | 18.413% |
| **`critical`** | Complete outage, communication dropout, destructive overload | 3,257 | 0.179% | 10.806% |
| **Total** | 6-Grade Taxonomy | **1,814,400** | **100.000%** | **100.000%** |

---

## 4. Feature Taxonomy & Leakage Prevention Audit

To prevent any form of target leakage or contamination:

| Column Name | Type / Source | Audit Decision | Justification |
| :--- | :--- | :--- | :--- |
| `timestamp` | Datetime (1 Hz) | **Metadata / Index** | Used for temporal alignment; not an input feature. |
| `sample_id` | Integer (1..604800) | **Excluded** | Arbitrary sequential row identifier. |
| `station_id` | Integer (1, 2, 3) | **Grouping Variable** | Used for Leave-One-Station-Out (LOSO) evaluation. |
| `segment_id` | Integer (0..6) | **Grouping Variable** | Calendar-day index (enables Leave-One-Day-Out splits). |
| `event_id` | Integer (0..N) | **Grouping Variable** | Identifies contiguous fault episodes; **strictly excluded from inputs**. |
| `anomaly` | Binary (0, 1) | **Target** | Binary anomaly label; excluded from input predictors. |
| `anomaly_type`| Categorical (17 classes) | **Excluded** | Fine-grained label used only for verification. |
| `fault_category`| Categorical (9 classes) | **Target A** | Multi-class target variable. |
| `severity` | Categorical (6 classes) | **Target B** | Severity target variable. |
| `detector_flag`| Binary (0, 1) | **Excluded** | Output of author's baseline edge detector (potential target leakage). |
| `threshold` | Float | **Excluded** | Author baseline dynamic threshold; not an intrinsic sensor measurement. |
| `data_quality` | Categorical | **Excluded** | Target indicator (`real` vs `interpolated`). |
| `is_interpolated` | Binary | **Excluded** | Telemetry artifact indicator. |
| `gap_length_s` | Seconds | **Excluded** | Telemetry artifact indicator. |
| `wifi_rssi_dBm`| dBm | **Excluded** | Radio signal strength; specific to physical deployment node. |
| `voltage_V` | Volts (RMS) | **Input Feature** | Directly measured 1 Hz line voltage. |
| `current_A` | Amperes (RMS) | **Input Feature** | Directly measured 1 Hz load current. |
| `power_W` | Watts (Active) | **Input Feature** | Directly measured active electrical power. |
| `energy_kWh` | kWh (Register) | **Excluded** | Monotonically accumulating meter counter. |
| `frequency_Hz`| Hertz | **Input Feature** | Directly measured AC line frequency. |
| `power_factor`| Ratio [-1, 1] | **Input Feature** | Directly measured $\cos \phi$. |
| `PE_m3..m5` | Float | **Input Feature** | Permutation entropy (signal complexity indicator). |
| `WPE`, `MSPE` | Float | **Input Feature** | Weighted & Multi-Scale Permutation Entropy. |
| `CV` | Ratio | **Input Feature** | Coefficient of variation of power envelope. |
| `delta_PF` | Difference | **Input Feature** | Dynamic deviation in power factor. |
| `SII` | Index | **Input Feature** | Spectral irregularity index. |
| `apparent_power_VA` | VA ($V \cdot I$) | **Engineered Input** | Calculated AC apparent power: $\|V\| \cdot \|I\|$. |
| `reactive_power_var`| VAR ($\sqrt{S^2 - P^2}$) | **Engineered Input** | Calculated AC reactive power: $\sqrt{\max(0, S^2 - P^2)}$. |
| `power_active_apparent_ratio` | Ratio | **Engineered Input** | Efficiency / distortion proxy ($P / S$), clipped to $[0.0, 2.0]$. |
| `voltage_deviation_pct` | % from 220V | **Engineered Input** | Percentage deviation from nominal 220V grid standard. |
| `frequency_deviation_pct`| % from 50Hz | **Engineered Input** | Percentage deviation from nominal 50.0 Hz grid standard. |
| `current_per_watt` | A/W ($\|I\| / \|P\|$) | **Engineered Input** | Current-to-power intensity ratio. Zero-denominator handling: replaces $P=0$ with NaN, fills with 0.0, and clips to $[0.0, 1.0]$. |

### Explicit Feature Differentiation
- **Original Dataset Columns**: 29 raw columns (5 direct sensor readings, 8 edge signal-processing features, 5 labels/targets, 4 metadata/grouping columns, and 7 edge detector/telemetry flags).
- **Engineered Input Features**: Exactly 6 domain-specific electrical features calculated deterministically from the 5 physical sensor channels.

---

## 5. Group Splitting & Contiguous Normal Block Strategy

1. **Fault Contiguity**: Verified across all 3 station files. Each non-zero `event_id` represents a contiguous, unbroken temporal window of controlled fault induction.
2. **Contiguous Normal Data Grouping**: In raw telemetry, all normal rows share `event_id = 0`.
   - **True Contiguous Chunking**: Normal observations are grouped into unbroken, contiguous chronological temporal blocks of 300 seconds (5 minutes) within each calendar day (`segment_id`) before selection.
   - **Zero Cross-Split Temporal Mixing**: Entire contiguous 300-second blocks are sampled and assigned unique `group_id`s `(station_id * 100000 + 50000 + segment_id * 1000 + time_block)`. `StratifiedGroupKFold` splits strictly by these `group_id`s, ensuring temporally adjacent seconds are never fractured across train and test partitions.
3. **Canonical Label Mapping**:
   - The raw CSV value `'med'` is formally mapped to the canonical label `'medium'` in both `src/config.py` and `src/preprocessing.py` using `RAW_SEVERITY_MAP` to ensure alignment with the official benchmark datasheet.
3. **Leave-One-Station-Out (LOSO) Validation**:
   - Station 1: ~661 W baseline (Laboratory)
   - Station 2: ~685 W baseline (Residential)
   - Station 3: ~4,255 W baseline (Industrial)
   - Testing on Station 3 while training on Stations 1 & 2 provides the ultimate test of cross-environment generalization.

---

## 6. Preprocessing & Imputation Protocols

- **No Cross-Event Forward Filling**: Missing values (~0.17% in physical sensor channels) are imputed strictly within each continuous operating window. Imputation never bridges across event boundaries or during actual outage events.
- **Fitting Restrictions**: Preprocessor scalers (`StandardScaler`) are fit **only on training partitions** and then applied to validation/test sets to guarantee zero look-ahead bias.
