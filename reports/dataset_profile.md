# Phase 1: Dataset Investigation & Profile Report

**Dataset**: A Real Controlled-Fault Benchmark for Power-Anomaly Detection and Identification on IoT Edge Nodes (RCF-Power v1.0)  
**DOI**: [10.5281/zenodo.20565892](https://doi.org/10.5281/zenodo.20565892)  
**License**: CC BY 4.0  
**Sampling Rate**: 1 Hz (1 sample per second over 7 continuous days: 2026-02-09 to 2026-02-15)  
**Total Records**: 1,814,400 across 3 distinct stations.

---

## 1. Station Composition & Sample Counts

| Station ID | Installation Context | Mean Power | Total Samples | Normal Samples | Anomaly Samples | Prevalence (%) | Distinct Fault Events |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Station 1** | Laboratory / Normal Baseline | ~661 W | 604,800 | 593,970 | 10,830 | 1.79% | 92 |
| **Station 2** | Residential / Medium Load | ~685 W | 604,800 | 594,357 | 10,443 | 1.73% | 84 |
| **Station 3** | Industrial / High Load | ~4,255 W | 604,800 | 595,931 | 8,869 | 1.47% | 78 |
| **Total** | Multi-Environment Benchmark | — | **1,814,400** | **1,784,258** | **30,142** | **1.66%** | **254** |

---

## 2. Comprehensive 29-Column Schema & Data Types

The dataset contains exactly 29 columns per station CSV:

| Index | Column Name | Physical Unit / Type | Description / Role in ML Pipeline | Null % (Station 1) |
|:---|:---|:---|:---|:---|
| 01 | `timestamp` | Datetime (YYYY-MM-DD HH:MM:SS) | 1 Hz recording timestamp | 0.00% |
| 02 | `sample_id` | Integer (1 ... 604800) | Sequential row counter per station | 0.00% |
| 03 | `station_id` | Integer (1, 2, 3) | Physical monitoring node identifier | 0.00% |
| 04 | `segment_id` | Integer (0 ... 6) | Calendar day index (enables leave-one-day-out CV) | 0.00% |
| 05 | `voltage_V` | Volts (V) | RMS AC line voltage | 0.17% |
| 06 | `current_A` | Amperes (A) | RMS AC load current | 0.17% |
| 07 | `power_W` | Watts (W) | Active electrical real power | 0.00% |
| 08 | `energy_kWh` | Kilowatt-hours (kWh) | Cumulative real energy meter register | 0.00% |
| 09 | `frequency_Hz` | Hertz (Hz) | AC line electrical frequency (nominally 50.0 Hz) | 0.17% |
| 10 | `power_factor` | Ratio [-1.0, 1.0] | Power factor (cos $\phi$) | 0.17% |
| 11 | `wifi_rssi_dBm`| dBm | Wi-Fi signal strength of IoT monitoring node | 0.00% |
| 12 | `is_interpolated`| Binary (0, 1) | Quality flag: 1 if bridged by linear interpolation | 0.00% |
| 13 | `gap_length_s` | Seconds (s) | Duration of interpolated data communication loss | 0.00% |
| 14 | `data_quality` | Categorical | Quality status (`real`, `interpolated`) | 0.00% |
| 15 | `PE_m3` | Float | Permutation Entropy (embedding order m=3) | 0.02% |
| 16 | `PE_m4` | Float | Permutation Entropy (embedding order m=4) | 0.02% |
| 17 | `PE_m5` | Float | Permutation Entropy (embedding order m=5) | 0.02% |
| 18 | `WPE` | Float | Weighted Permutation Entropy | 0.02% |
| 19 | `MSPE` | Float | Multi-Scale Permutation Entropy | 0.02% |
| 20 | `CV` | Ratio | Coefficient of Variation of power/current | 0.08% |
| 21 | `delta_PF` | Difference | Dynamic deviation in power factor | 0.02% |
| 22 | `SII` | Index | Spectral Irregularity Index | 0.08% |
| 23 | `threshold` | Float | Baseline adaptive anomaly threshold | 0.00% |
| 24 | `detector_flag` | Binary (0, 1) | Output flag of IoT baseline edge detector | 0.00% |
| 25 | `anomaly` | Binary (0, 1) | **Target: Binary Anomaly Label** | 0.00% |
| 26 | `anomaly_type` | Categorical (17 classes) | Specific controlled fault type (16 faults + normal) | 0.00% |
| 27 | `fault_category`| Categorical (8 classes) | **Target A: Multiclass Fault Category** | 0.00% |
| 28 | `severity` | Categorical (5 classes) | **Target B: Severity Grade** | 0.00% |
| 29 | `event_id` | Integer (0 = normal, 1..N) | Controlled fault event ID for group-aware splitting | 0.00% |

---

## 3. Ground-Truth Label Mapping Table

Through scanning the 1.81M samples, the dataset implements a clean deterministic hierarchical taxonomy:
- **16 Fine-grained Anomaly Types**
- **8 Broad Fault Categories** (excluding normal)
- **5 Severity Grades** (excluding none)

### Anomaly Type $\rightarrow$ Fault Category Mapping

| # | Anomaly Type (`anomaly_type`) | Fault Category (`fault_category`) | Physical Description / Manifestation |
|:--|:---|:---|:---|
| 0 | `normal` | `normal` | Standard operating conditions within nominal bands |
| 1 | `voltage_sag` | `voltage` | Supply drop from 220 V to 180–190 V |
| 2 | `voltage_swell` | `voltage` | Supply rise above nominal operating tolerance |
| 3 | `flicker` | `voltage` | Repetitive short-interval voltage fluctuations |
| 4 | `overload` | `load` | Excessive current drawing exceeding rated circuit capacity |
| 5 | `inrush_surge` | `load` | Abrupt high-current startup spike from energizing equipment |
| 6 | `gradual_drift` | `load` | Creeping load consumption drift indicating degradation |
| 7 | `frequency_deviation` | `frequency` | Grid frequency stepping away from 50.0 Hz (to 49 or 51 Hz) |
| 8 | `harmonic_distortion` | `power_quality` | Non-linear load injection, waveform distortion, elevated THD |
| 9 | `low_power_factor` | `power_quality` | High reactive power draw from lightly loaded inductive motors |
| 10 | `outage` | `outage` | Complete supply disconnection / opened MCCB circuit breaker |
| 11 | `sensor_dropout` | `sensor` | Missing / zeroed sensor transmission packets |
| 12 | `sensor_stuck` | `sensor` | Frozen measurement register holding identical values |
| 13 | `intermittent` | `sensor` | Loose electrical wiring, intermittent contact resistance |
| 14 | `spike_transient` | `sensor` | High-frequency impulse sensor noise |
| 15 | `contextual_load` | `context` | Unusual energy consumption given the time/context profile |
| 16 | `compound` | `compound` | Simultaneous concurrent multi-source faults |

---

## 4. Class Distribution Across Fault Categories & Severities

### Fault Category Counts (Anomalous Samples across all 3 stations)
1. **`normal`**: 1,784,258 (98.34%)
2. **`load`**: 6,267 (20.8% of anomalies)
3. **`power_quality`**: 4,357 (14.5% of anomalies)
4. **`sensor`**: 4,897 (16.2% of anomalies)
5. **`context`**: 4,282 (14.2% of anomalies)
6. **`compound`**: 4,105 (13.6% of anomalies)
7. **`voltage`**: 2,984 (9.9% of anomalies)
8. **`outage`**: 2,174 (7.2% of anomalies)
9. **`frequency`**: 1,076 (3.6% of anomalies)

### Severity Grade Distribution
1. **`none`**: 1,784,258 (Normal baseline)
2. **`incipient`**: 6,959 (Early subtle anomaly signatures)
3. **`low`**: 5,550
4. **`med`**: 5,969
5. **`high`**: 8,407
6. **`critical`**: 3,257 (Severe dropouts, outages, critical overloads)

---

## 5. Group-Aware Splitting & Validation Verification

- **Direct Event IDs exist**: The dataset explicitly includes `event_id` (0 for normal, 1 to 92 for Station 1, 1 to 84 for Station 2, 1 to 78 for Station 3).
- **Contiguity**: Contiguous fault samples are strictly linked to their `event_id`.
- **Validation strategy confirmed**:
  1. **Event-Aware Group Stratified K-Fold**: Splitting by `(station_id, event_id)` ensures whole fault events are assigned exclusively to train OR test, eliminating data leakage.
  2. **Leave-One-Station-Out (LOSO) Generalization**: Training on Stations 1 & 2 and testing on Station 3 (and vice versa) tests the ensemble's ability to adapt to unseen industrial vs residential baseline electrical loads.
- **Missing Value Handling**: Only raw physical sensor columns (`voltage_V`, `current_A`, `frequency_Hz`, `power_factor`) have minor dropouts (~0.17%), which can be cleanly forward-filled or median-imputed.
