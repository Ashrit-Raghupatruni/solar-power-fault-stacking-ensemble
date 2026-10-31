# RCF-Power — Real Controlled-Fault Power-Monitoring Benchmark (v1.0)

A benchmark of **real controlled-fault** power recordings from low-cost IoT metering hardware (ESP32 + PZEM-004T), for reproducible power-fault detection and identification at the edge.

## Contents
- `station1_normal_features.csv`, `station2_medium_features.csv`, `station3_high_features.csv` — 3 stations × 604,800 rows (7 days @ 1 Hz; 1,814,400 samples total).
- `DATASHEET.md` — full datasheet (Gebru et al. format): motivation, composition, collection, uses, limitations.
- `LICENSE` — CC-BY-4.0.
- `code/` — feature-extraction + cross-validation scripts to reproduce the paper's tables/figures.

## Quick facts
- ≈99.8% directly measured (≤0.3% interpolated, flagged by `is_interpolated`/`data_quality`).
- Labels: binary `anomaly`; 16 `anomaly_type`; 8 `fault_category`; 5 `severity`; `event_id`.
- `segment_id` = one calendar day → use **leave-one-day-out** CV (leakage-safe; every category spans ≥2 days).
- Anomaly prevalence 1.5–1.8% per station.

## Important scope note (read before use)
At 1 Hz, labels denote the **1 Hz envelope/RMS signature** of each phenomenon. "flicker"/"harmonic" here are NOT sub-cycle waveform quantities (an IEC flicker meter or harmonic analyzer measures different things). Faults are **induced under controlled protocols**, not naturally occurring. Not a general-purpose anomaly benchmark (out-of-domain AUROC ≈0.50 on NAB).

## Cite
> Tran, T. T., & Tran, N. T. (2026). RCF-Power: A Real Controlled-Fault Power-Monitoring Benchmark for IoT Edge (v1.0) [Data set]. https://doi.org/[to-be-assigned]

## How to reproduce
```bash
pip install pandas numpy scikit-learn scipy pywavelets lightgbm
python code/run_improve.py        # detection: baseline vs +spectral, 3 stations, 7-fold
# further analyses: analyze_multiclass.py, analyze_rpd_ablation.py, analyze_feature_cost.py, analyze_loso.py
```
Figures (MATLAB): `plot_improve.m`, `plot_reviewer.m`.
