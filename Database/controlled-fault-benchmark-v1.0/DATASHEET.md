# Datasheet — Real Controlled-Fault Power-Monitoring Benchmark (RCF-Power)

> Following Gebru et al., "Datasheets for Datasets" (CACM 2021). Prepared for public release alongside the manuscript.

## Motivation
- **Why created.** Rigorous on-device evaluation of power-fault detection is hampered by the scarcity of *real*, per-sample-labelled fault data; most prior work uses synthetic injection. This benchmark provides real controlled-fault recordings at 1 Hz from commodity metering hardware, with sixteen fault types and per-event labels, to enable reproducible, leakage-aware evaluation.
- **Created by.** Thanh Trang Tran (HUIT, corresponding author) and Tran Nhut Tam (HCMUT, VNU-HCM).
- **Funding.** The authors received no specific funding for this work.

## Composition
- **Instances.** 1,814,400 per-second samples = 3 stations × 604,800 (7 days @ 1 Hz, 2026-02-09 → 02-15).
- **Stations.** Station 1 (laboratory/normal, ~661 W), Station 2 (residential/medium, ~685 W), Station 3 (industrial/high, ~4,255 W).
- **Fields (29 columns).** timestamp, sample_id, station_id, segment_id (= 1 calendar day), voltage_V, current_A, power_W, energy_kWh, frequency_Hz, power_factor, wifi_rssi_dBm, is_interpolated, gap_length_s, data_quality, precomputed features (PE_m3/m4/m5, WPE, MSPE, CV, delta_PF, SII), threshold, detector_flag, and labels (anomaly, anomaly_type, fault_category, severity, event_id).
- **Labels.** Binary `anomaly`; 16 `anomaly_type`; 8 `fault_category` (load, voltage, frequency, power_quality, sensor, outage, context, compound); 5 `severity` (incipient/low/medium/high/critical); `event_id` groups contiguous fault samples.
- **Class balance.** Anomaly prevalence 1.5–1.8% per station (10,830 / 10,443 / 8,869 anomalous samples).
- **Real vs interpolated.** ≈99.8% directly measured (St1 99.83%, St2 99.78%, St3 99.72%); ≤0.3% linearly interpolated to bridge short gaps (flagged by `is_interpolated` / `data_quality`).
- **Fault×day coverage.** Every 8-category fault spans ≥2 distinct days on all stations (enables leakage-free leave-one-day-out CV); the only 16-type exception is `overload` on Station 3 (one day).

## Collection process
- **Hardware.** ESP32-WROOM-32 + PZEM-004T, reporting V/I/P/energy/frequency/PF at 1 Hz over Wi-Fi.
- **Fault induction (author-confirmed, 2026-06-05).** Faults were physically induced under controlled protocols by acting on the installation or its supply, never by software injection into recorded traces. *Load-side:* overload by switching on many appliances at once or mechanically loading a motor; load surge/inrush by abruptly energizing high-power appliances (electric kettles, hotplates, hair dryers, space heaters) and motors/transformers; low power factor via lightly loaded induction motors; harmonic distortion by connecting non-linear loads (variable-frequency drives, diode rectifiers, high-power switch-mode supplies, UPS units), cross-checked with a THD measurement from a power-quality analyzer at protocol setup; irregular/time-varying load events by repeatedly toggling or stepping motor, pump, and air-compressor loads. *Grid-side (emulated at the supply side, since genuine grid-wide excursions cannot be induced safely from a consumer installation):* frequency deviation by feeding the monitored circuit from a programmable AC source or adjustable-frequency inverter stepped from 50 Hz toward ≈49/51 Hz; voltage sag by lowering the supply from 220 V to 180–190 V with a variac, programmable AC source, or grid simulator; voltage swell by symmetrically raising the supply above nominal with the same apparatus; outage by opening the MCCB/contactor feeding the station for several seconds and re-closing (energy register holds its last value). *Sensor/telemetry:* frozen sensor, Wi-Fi loss, electrical noise, loose connector.
- **Labeling.** Events were labelled at acquisition time as part of the controlled fault protocols (onset/offset timestamps, fault type, and severity recorded per event; `event_id` groups contiguous samples).
- **Timeframe.** 7 consecutive days per station.

## Preprocessing / cleaning
- 1 Hz uniform grid; short gaps linearly interpolated (≤0.3%, flagged).
- Precomputed features included for reproducibility; raw V/I/P/f/PF retained so users can recompute.
- **Scope of labels at 1 Hz:** labels denote the 1 Hz envelope/RMS signature of each phenomenon; "flicker"/"harmonic" are NOT waveform-level (sub-cycle) quantities (see manuscript Sect. 4.2).

## Uses
- **Suitable for.** Benchmarking 1 Hz power-fault detection/identification, edge-ML feature studies, leakage-aware CV protocols.
- **Not suitable for.** Sub-cycle PQ analysis (flicker/harmonic at their defining frequencies); general-purpose time-series anomaly detection (out-of-domain AUROC ≈0.50 on NAB); claims about *naturally-occurring* faults (these are induced).

## Distribution & maintenance
- **Release.** Zenodo, CC-BY-4.0, versioned DOI: https://doi.org/10.5281/zenodo.20565892 (3 station CSVs, this datasheet, the feature/CV code, and a `LICENSE`).
- **Format.** CSV (UTF-8), one row per second.
- **Maintainer / contact.** trangtt@huit.edu.vn.
- **Versioning.** v1.0 = this release; record any corrections in a CHANGELOG.

## Ethical / legal
- No personal data; aggregate electrical measurements only.

---
### Suggested citation
> Tran, T. T., & Tran, N. T. (2026). A Real Controlled-Fault Benchmark for Power-Anomaly Detection and Identification on IoT Edge Nodes (three-station 1 Hz dataset) (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20565892
