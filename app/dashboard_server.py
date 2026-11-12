"""
PowerGuard AI - Electrical Intelligence & Solar PV Monitoring Dashboard Server.
Pure zero-dependency, ultra-responsive HTML5/CSS3/JavaScript industrial monitoring platform.
Features 8 dedicated sections:
  01. Overview - KPIs & fault distribution charts
  02. Live Monitoring - Real-time 1 Hz telemetry waveforms with fault injection
  03. Fault Diagnosis - Two-stage cascaded detection with probability gauges
  04. Stacking Ensemble - Architecture visualizer & meta-learner comparison
  05. Solar PV Monitoring - AhoyDTU / OpenDTU multi-string & efficiency monitoring
  06. Model Evaluation - Normalized confusion matrices & classification metrics
  07. Dataset Analytics - 1.81M multi-station distribution and class imbalance
  08. System & API - FastAPI health, loaded models & operational schema
"""
import http.server
import socketserver
import os
import json
from pathlib import Path

PORT = 8501
BASE_DIR = Path(__file__).resolve().parent.parent

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PowerGuard AI — Electrical Intelligence & Solar PV Platform</title>
    <style>
        :root {
            --bg-base: #0a0e17;
            --bg-card: #111827;
            --bg-card-hover: #172133;
            --border: #1f293d;
            --border-highlight: #38bdf8;
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.2);
            --accent: #f59e0b;
            --danger: #ef4444;
            --success: #10b981;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --sidebar-width: 260px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-base); color: var(--text-main); display: flex; min-height: 100vh; overflow-x: hidden; }

        /* Left Navigation Sidebar */
        aside {
            width: var(--sidebar-width);
            background: var(--bg-card);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            position: fixed;
            height: 100vh;
            z-index: 100;
        }

        .brand-box {
            padding: 24px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, #38bdf8, #2563eb);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .brand-title { font-size: 16px; font-weight: 700; letter-spacing: 0.5px; }
        .brand-sub { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }

        .nav-list { list-style: none; padding: 15px 10px; flex: 1; overflow-y: auto; }
        .nav-item {
            padding: 12px 14px;
            margin-bottom: 4px;
            border-radius: 8px;
            font-size: 13.5px;
            font-weight: 500;
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 12px;
            transition: all 0.2s ease;
        }

        .nav-item:hover { color: var(--text-main); background: rgba(56, 189, 248, 0.05); }
        .nav-item.active {
            color: var(--primary);
            background: rgba(56, 189, 248, 0.12);
            border-left: 3px solid var(--primary);
            font-weight: 600;
        }

        .nav-num { font-size: 11px; opacity: 0.6; font-family: monospace; }

        /* Backend Status Pill */
        .status-box {
            padding: 16px;
            border-top: 1px solid var(--border);
            font-size: 12px;
        }
        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            background: rgba(16, 185, 129, 0.1);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.2);
            font-weight: 600;
        }
        .status-pill.offline {
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger);
            border-color: rgba(239, 68, 68, 0.2);
        }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

        /* Main Container */
        main {
            margin-left: var(--sidebar-width);
            flex: 1;
            padding: 30px;
            max-width: 1400px;
        }

        /* Top Bar */
        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border);
        }

        .tab-title { font-size: 22px; font-weight: 700; color: #fff; }
        .tab-subtitle { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

        /* Offline Disconnect Banner */
        .alert-banner {
            display: none;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid var(--danger);
            color: #fca5a5;
            padding: 12px 18px;
            border-radius: 8px;
            margin-bottom: 25px;
            font-size: 13px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Grid Cards */
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
        .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 24px; }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            position: relative;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .card-title { font-size: 14px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        .card-val { font-size: 28px; font-weight: 700; color: #fff; }
        .card-sub { font-size: 12px; color: var(--text-muted); margin-top: 6px; }

        /* Form Inputs */
        .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 20px; }
        .form-group { display: flex; flex-direction: column; gap: 6px; }
        .form-group label { font-size: 12px; font-weight: 600; color: var(--text-muted); }
        .form-group input {
            background: #090d16;
            border: 1px solid var(--border);
            padding: 10px 14px;
            border-radius: 6px;
            color: #fff;
            font-size: 14px;
            font-family: monospace;
            outline: none;
            transition: border 0.2s;
        }
        .form-group input:focus { border-color: var(--primary); }

        button.btn-primary {
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: #fff;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: transform 0.1s, opacity 0.2s;
        }
        button.btn-primary:hover { opacity: 0.95; }
        button.btn-secondary {
            background: #1f2937;
            color: #e5e7eb;
            border: 1px solid #374151;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            font-weight: 500;
            margin-right: 8px;
        }
        button.btn-secondary:hover { background: #374151; }

        /* Tab Sections */
        .tab-section { display: none; }
        .tab-section.active { display: block; animation: fadeIn 0.25s ease-out; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

        /* Diagnosis Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .badge-normal { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; }
        .badge-warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #d97706; }
        .badge-danger { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #dc2626; }

        /* Waveform Canvas */
        canvas.stream-canvas {
            width: 100%;
            height: 140px;
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 8px;
            display: block;
        }

        /* SVG Architecture Box */
        .arch-container {
            background: #090d16;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            overflow-x: auto;
        }

        /* Confusion Matrix Image Container */
        .img-container {
            border-radius: 8px;
            border: 1px solid var(--border);
            overflow: hidden;
            background: #090d16;
            text-align: center;
            padding: 10px;
        }
        .img-container img { max-width: 100%; height: auto; border-radius: 4px; }
    </style>
</head>
<body>

    <!-- Sidebar Navigation -->
    <aside>
        <div class="brand-box">
            <div class="brand-icon">⚡</div>
            <div>
                <div class="brand-title">PowerGuard AI</div>
                <div class="brand-sub">Electrical & Solar</div>
            </div>
        </div>

        <ul class="nav-list">
            <li class="nav-item active" onclick="switchNav(0)">
                <span class="nav-num">01.</span> Overview
            </li>
            <li class="nav-item" onclick="switchNav(1)">
                <span class="nav-num">02.</span> Live Monitoring
            </li>
            <li class="nav-item" onclick="switchNav(2)">
                <span class="nav-num">03.</span> Fault Diagnosis
            </li>
            <li class="nav-item" onclick="switchNav(3)">
                <span class="nav-num">04.</span> Stacking Ensemble
            </li>
            <li class="nav-item" onclick="switchNav(4)">
                <span class="nav-num">05.</span> Solar PV Monitoring
            </li>
            <li class="nav-item" onclick="switchNav(5)">
                <span class="nav-num">06.</span> Model Evaluation
            </li>
            <li class="nav-item" onclick="switchNav(6)">
                <span class="nav-num">07.</span> Dataset Analytics
            </li>
            <li class="nav-item" onclick="switchNav(7)">
                <span class="nav-num">08.</span> System & API
            </li>
        </ul>

        <div class="status-box">
            <div class="status-pill" id="api-status-pill">
                <div class="status-dot"></div>
                <span id="api-status-text">Backend Connected</span>
            </div>
        </div>
    </aside>

    <!-- Main Viewport -->
    <main>
        <!-- Offline Fail-Safe Banner -->
        <div class="alert-banner" id="disconnect-banner">
            <div>
                <strong>⚠️ Backend ML Engine Disconnected (Port 8000)</strong><br>
                Predictions suspended. To restore pure ML inference, run: <code>uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload</code>
            </div>
            <button class="btn-secondary" onclick="checkApiHealth()">Retry</button>
        </div>

        <!-- 01. OVERVIEW -->
        <section class="tab-section active" id="sec-0">
            <div class="top-bar">
                <div>
                    <div class="tab-title">Platform Overview & System KPIs</div>
                    <div class="tab-subtitle">Real-time status of 1.81 Million dataset benchmark and active electrical monitoring nodes.</div>
                </div>
            </div>

            <div class="grid-4">
                <div class="card">
                    <div class="card-title">Dataset Scale</div>
                    <div class="card-val" style="color: var(--primary);">1,814,400</div>
                    <div class="card-sub">Total 1 Hz observations across 3 stations</div>
                </div>
                <div class="card">
                    <div class="card-title">Normal Baseline</div>
                    <div class="card-val" style="color: var(--success);">98.34%</div>
                    <div class="card-sub">1,784,258 nominal records</div>
                </div>
                <div class="card">
                    <div class="card-title">Controlled Anomalies</div>
                    <div class="card-val" style="color: var(--accent);">30,142</div>
                    <div class="card-sub">254 distinct physical fault events</div>
                </div>
                <div class="card">
                    <div class="card-title">Stage 1 Anomaly Recall</div>
                    <div class="card-val" style="color: #38bdf8;">94.95%</div>
                    <div class="card-sub">Tuned safety threshold τ = 0.125</div>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Fault Category Distribution (30,142 Anomalies)</div>
                    </div>
                    <p style="font-size: 13px; color: var(--text-muted); line-height: 1.6;">
                        • <strong>Load Faults</strong>: 6,267 (20.8%)<br>
                        • <strong>Sensor Artifacts</strong>: 4,897 (16.2%)<br>
                        • <strong>Power Quality (Low PF)</strong>: 4,357 (14.5%)<br>
                        • <strong>Operating Context</strong>: 4,282 (14.2%)<br>
                        • <strong>Compound Faults</strong>: 4,105 (13.6%)<br>
                        • <strong>Voltage Sags / Swells</strong>: 2,984 (9.9%)<br>
                        • <strong>Supply Outages</strong>: 2,174 (7.2%)<br>
                        • <strong>Frequency Excursions</strong>: 1,076 (3.6%)
                    </p>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Controlled Severity Distribution</div>
                    </div>
                    <p style="font-size: 13px; color: var(--text-muted); line-height: 1.6;">
                        • <strong>High Severity</strong>: 8,407 records<br>
                        • <strong>Incipient Faults</strong>: 6,959 records<br>
                        • <strong>Medium Severity (med)</strong>: 5,969 records<br>
                        • <strong>Low Severity</strong>: 5,550 records<br>
                        • <strong>Critical Faults</strong>: 3,257 records (Safety-weighted 5x)
                    </p>
                </div>
            </div>
        </section>

        <!-- 02. LIVE MONITORING -->
        <section class="tab-section" id="sec-1">
            <div class="top-bar">
                <div>
                    <div class="tab-title">Real-Time 1 Hz Telemetry Stream Simulator</div>
                    <div class="tab-subtitle">Live IoT continuous circular stream. Ingests simulated telemetry directly into FastAPI inference.</div>
                </div>
                <div>
                    <button class="btn-primary" id="btn-stream-toggle" onclick="toggleStream()">▶ Start 1 Hz Stream</button>
                    <button class="btn-secondary" onclick="resetStream()">Reset</button>
                </div>
            </div>

            <div style="margin-bottom: 20px;">
                <span style="font-size: 13px; color: var(--text-muted); margin-right: 12px;">Inject Controlled Faults:</span>
                <button class="btn-secondary" onclick="injectFault('normal')">✓ Nominal Grid</button>
                <button class="btn-secondary" onclick="injectFault('sag')">⚡ Voltage Sag (175V)</button>
                <button class="btn-secondary" onclick="injectFault('overload')">⚡ Inverter Overcurrent (24.5A)</button>
                <button class="btn-secondary" onclick="injectFault('outage')">⚡ Grid Outage (0V)</button>
            </div>

            <div class="grid-4">
                <div class="card">
                    <div class="card-title">Live AC Voltage</div>
                    <div class="card-val" id="live-v">220.0 V</div>
                    <div class="card-sub">Nominal 220.0 ±10%</div>
                </div>
                <div class="card">
                    <div class="card-title">Live AC Current</div>
                    <div class="card-val" id="live-i">5.0 A</div>
                    <div class="card-sub">Rated branch limit: 20 A</div>
                </div>
                <div class="card">
                    <div class="card-title">Live Active Power</div>
                    <div class="card-val" id="live-p">1,078 W</div>
                    <div class="card-sub">Dynamic P = V * I * PF</div>
                </div>
                <div class="card">
                    <div class="card-title">Live ML Status</div>
                    <div class="card-val" id="live-status" style="color: var(--success); font-size: 20px;">NOMINAL</div>
                    <div class="card-sub" id="live-prob">Stage 1 Anom Prob: 1.2%</div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 20px;">
                <div class="card-title" style="margin-bottom: 12px;">Live Line Voltage Waveform (1 Hz Telemetry Buffer)</div>
                <canvas id="waveformCanvas" class="stream-canvas" width="1200" height="140"></canvas>
            </div>
        </section>

        <!-- 03. FAULT DIAGNOSIS -->
        <section class="tab-section" id="sec-2">
            <div class="top-bar">
                <div>
                    <div class="tab-title">Two-Stage Hierarchical Fault Diagnosis</div>
                    <div class="tab-subtitle">Stage 1 Binary Anomaly Filter (τ = 0.125) + Stage 2 Residual Stacking Classifiers.</div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 24px;">
                <div class="card-title" style="margin-bottom: 16px;">Manual Sensor Telemetry Input</div>
                <div class="form-grid">
                    <div class="form-group">
                        <label>AC Voltage (V) [Mandatory]</label>
                        <input type="number" id="diag_v" value="220.0" step="0.5">
                    </div>
                    <div class="form-group">
                        <label>AC Current (A) [Mandatory]</label>
                        <input type="number" id="diag_i" value="5.2" step="0.1">
                    </div>
                    <div class="form-group">
                        <label>Active Power (W) [Optional]</label>
                        <input type="number" id="diag_p" value="1144.0" step="10">
                    </div>
                    <div class="form-group">
                        <label>Frequency (Hz) [Optional]</label>
                        <input type="number" id="diag_f" value="50.0" step="0.05">
                    </div>
                    <div class="form-group">
                        <label>Power Factor (cos φ) [Optional]</label>
                        <input type="number" id="diag_pf" value="0.98" step="0.01">
                    </div>
                    <div class="form-group" style="justify-content: flex-end;">
                        <button class="btn-primary" onclick="runManualDiagnosis()">⚡ Execute ML Diagnosis</button>
                    </div>
                </div>

                <div style="font-size: 12px; color: var(--text-muted);">
                    Preset Test Scenarios: 
                    <a href="javascript:void(0)" onclick="setPreset(220, 5.2, 1144, 50, 0.98)" style="color: var(--primary); margin-left: 6px;">Nominal Baseline</a> |
                    <a href="javascript:void(0)" onclick="setPreset(178, 4.5, 760, 50, 0.95)" style="color: var(--primary); margin-left: 6px;">Voltage Sag (178V)</a> |
                    <a href="javascript:void(0)" onclick="setPreset(225, 26.0, 5700, 50, 0.96)" style="color: var(--primary); margin-left: 6px;">Current Overload (26A)</a> |
                    <a href="javascript:void(0)" onclick="setPreset(0, 0, 0, 50, 0)" style="color: var(--primary); margin-left: 6px;">Grid Outage (0V)</a>
                </div>
            </div>

            <div class="card" id="diag-results-box" style="display: none;">
                <div class="card-header">
                    <div class="card-title">Real-Time Inference Results (FastAPI ML Service)</div>
                    <span id="diag-badge" class="badge badge-normal">Normal Baseline</span>
                </div>

                <div class="grid-3">
                    <div class="card" style="background: #090d16;">
                        <div class="card-title">Stage 1: Binary Filter</div>
                        <div class="card-val" id="diag-s1-val" style="font-size: 20px;">Nominal (3.4%)</div>
                        <div class="card-sub">Decision threshold τ = 0.125</div>
                    </div>
                    <div class="card" style="background: #090d16;">
                        <div class="card-title">Stage 2A: Fault Category</div>
                        <div class="card-val" id="diag-cat-val" style="font-size: 20px; color: var(--primary);">Normal</div>
                        <div class="card-sub" id="diag-cat-conf">Confidence: 96.6%</div>
                    </div>
                    <div class="card" style="background: #090d16;">
                        <div class="card-title">Stage 2B: Severity Grade</div>
                        <div class="card-val" id="diag-sev-val" style="font-size: 20px; color: var(--accent);">None</div>
                        <div class="card-sub" id="diag-sev-conf">Confidence: 96.6%</div>
                    </div>
                </div>

                <div style="background: #090d16; border: 1px solid var(--border); border-radius: 8px; padding: 16px;">
                    <div style="font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 8px;">💡 Physics-Guided Diagnostic Indicators:</div>
                    <div id="diag-triggers" style="font-size: 13px; color: var(--text-muted); line-height: 1.6;"></div>
                </div>
            </div>
        </section>

        <!-- 04. STACKING ENSEMBLE -->
        <section class="tab-section" id="sec-3">
            <div class="top-bar">
                <div>
                    <div class="tab-title">Stacking Ensemble & Model Intelligence</div>
                    <div class="tab-subtitle">Visual architecture of Level 0 Base Learners, 5-Fold OOF Predictions, and Level 1 Meta-Learners.</div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 24px;">
                <div class="card-title" style="margin-bottom: 16px;">Residual Stacking Ensemble Flowchart</div>
                <div class="arch-container">
                    <svg width="1000" height="260" viewBox="0 0 1000 260">
                        <!-- Input Block -->
                        <rect x="20" y="90" width="160" height="80" rx="8" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
                        <text x="100" y="125" fill="#fff" font-weight="700" font-size="13" text-anchor="middle">16 Operational</text>
                        <text x="100" y="145" fill="#9ca3af" font-size="11" text-anchor="middle">Feature Vector</text>

                        <!-- Lines to Level 0 -->
                        <line x1="180" y1="130" x2="260" y2="40" stroke="#475569" stroke-width="2"/>
                        <line x1="180" y1="130" x2="260" y2="85" stroke="#475569" stroke-width="2"/>
                        <line x1="180" y1="130" x2="260" y2="130" stroke="#475569" stroke-width="2"/>
                        <line x1="180" y1="130" x2="260" y2="175" stroke="#475569" stroke-width="2"/>
                        <line x1="180" y1="130" x2="260" y2="220" stroke="#475569" stroke-width="2"/>

                        <!-- Level 0 Base Learners -->
                        <rect x="260" y="20" width="180" height="35" rx="6" fill="#111827" stroke="#475569"/>
                        <text x="350" y="42" fill="#fff" font-size="12" text-anchor="middle">Hist Gradient Boosting</text>

                        <rect x="260" y="65" width="180" height="35" rx="6" fill="#111827" stroke="#475569"/>
                        <text x="350" y="87" fill="#fff" font-size="12" text-anchor="middle">Random Forest (120 Trees)</text>

                        <rect x="260" y="110" width="180" height="35" rx="6" fill="#111827" stroke="#475569"/>
                        <text x="350" y="132" fill="#fff" font-size="12" text-anchor="middle">Extra Trees (120 Trees)</text>

                        <rect x="260" y="155" width="180" height="35" rx="6" fill="#111827" stroke="#475569"/>
                        <text x="350" y="177" fill="#fff" font-size="12" text-anchor="middle">Linear SGD (Log Loss)</text>

                        <rect x="260" y="200" width="180" height="35" rx="6" fill="#111827" stroke="#475569"/>
                        <text x="350" y="222" fill="#fff" font-size="12" text-anchor="middle">K-Nearest Neighbors</text>

                        <!-- Meta Feature Block -->
                        <rect x="520" y="80" width="170" height="100" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
                        <text x="605" y="115" fill="#fff" font-weight="700" font-size="12" text-anchor="middle">5-Fold OOF Predictions</text>
                        <text x="605" y="135" fill="#f59e0b" font-weight="600" font-size="11" text-anchor="middle">+ Soft-Voting Consensus</text>
                        <text x="605" y="155" fill="#9ca3af" font-size="10" text-anchor="middle">(Residual Skip Connection)</text>

                        <!-- Lines to Meta Feature -->
                        <line x1="440" y1="37" x2="520" y2="130" stroke="#475569"/>
                        <line x1="440" y1="82" x2="520" y2="130" stroke="#475569"/>
                        <line x1="440" y1="127" x2="520" y2="130" stroke="#475569"/>
                        <line x1="440" y1="172" x2="520" y2="130" stroke="#475569"/>
                        <line x1="440" y1="217" x2="520" y2="130" stroke="#475569"/>

                        <!-- Level 1 Meta Learner -->
                        <line x1="690" y1="130" x2="760" y2="130" stroke="#f59e0b" stroke-width="2"/>
                        <rect x="760" y="90" width="190" height="80" rx="8" fill="#111827" stroke="#10b981" stroke-width="2"/>
                        <text x="855" y="125" fill="#10b981" font-weight="700" font-size="13" text-anchor="middle">Level 1: Meta-Learner</text>
                        <text x="855" y="145" fill="#fff" font-size="12" text-anchor="middle">Logistic Regression (L2)</text>
                    </svg>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <div class="card-title">Ensemble Architecture Comparison</div>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 13px;">
                        <thead>
                            <tr style="border-bottom: 1px solid var(--border); text-align: left; color: var(--text-muted);">
                                <th style="padding: 8px;">Architecture</th>
                                <th style="padding: 8px;">Mechanism</th>
                                <th style="padding: 8px;">Isolated Stage 2 F1</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid var(--border);">
                                <td style="padding: 10px 8px; font-weight: 600; color: var(--primary);">Soft Voting</td>
                                <td style="padding: 10px 8px; color: var(--text-muted);">Uniform Probability Average</td>
                                <td style="padding: 10px 8px; font-weight: 700; color: #fff;">70.35%</td>
                            </tr>
                            <tr>
                                <td style="padding: 10px 8px; font-weight: 600; color: var(--success);">Residual Stacking</td>
                                <td style="padding: 10px 8px; color: var(--text-muted);">Learned Meta-Model + Consensus</td>
                                <td style="padding: 10px 8px; font-weight: 700; color: #fff;">64.02%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="card">
                    <div class="card-title">Why Stacking Differs From Soft Voting</div>
                    <p style="font-size: 13px; color: var(--text-muted); line-height: 1.6; margin-top: 10px;">
                        • <strong>Soft Voting</strong> averages probability vectors across base models without learning class-dependent weights.<br>
                        • <strong>Residual Stacking</strong> fits a Logistic Regression meta-learner on 5-fold Out-Of-Fold predictions, learning which base classifier to trust for each fault type while incorporating consensus probabilities.
                    </p>
                </div>
            </div>
        </section>

        <!-- 05. SOLAR PV MONITORING -->
        <section class="tab-section" id="sec-4">
            <div class="top-bar">
                <div>
                    <div class="tab-title">Solar PV Inverter Monitoring (AhoyDTU / OpenDTU)</div>
                    <div class="tab-subtitle">Direct microinverter ingestion. Maps AC parameters to ML fault detection and isolates DC string yield analytics.</div>
                </div>
            </div>

            <div class="grid-4">
                <div class="card">
                    <div class="card-title">Inverter Efficiency (η)</div>
                    <div class="card-val" id="pv-eff" style="color: var(--success);">95.9%</div>
                    <div class="card-sub">η = (P_ac / Σ P_dc) * 100</div>
                </div>
                <div class="card">
                    <div class="card-title">Total DC Array Power</div>
                    <div class="card-val" id="pv-total-dc">13.0 W</div>
                    <div class="card-sub">Channels 1–4 aggregated</div>
                </div>
                <div class="card">
                    <div class="card-title">AC Output Power</div>
                    <div class="card-val" id="pv-ac-power">12.5 W</div>
                    <div class="card-sub">Grid-tie active power</div>
                </div>
                <div class="card">
                    <div class="card-title">AC Reactive Power (Q)</div>
                    <div class="card-val" id="pv-ac-q">19.6 VAR</div>
                    <div class="card-sub">Q = sqrt(S² - P²)</div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 24px;">
                <div class="card-title" style="margin-bottom: 14px;">Hoymiles / AhoyDTU Multi-Channel DC String Telemetry</div>
                <div class="grid-4">
                    <div class="card" style="background: #090d16;">
                        <div style="font-size: 12px; font-weight: 600; color: var(--primary);">Channel 1</div>
                        <div style="font-size: 18px; font-weight: 700; margin-top: 4px;">27.1 V | 0.12 A</div>
                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">Power: 3.25 W</div>
                    </div>
                    <div class="card" style="background: #090d16;">
                        <div style="font-size: 12px; font-weight: 600; color: var(--primary);">Channel 2</div>
                        <div style="font-size: 18px; font-weight: 700; margin-top: 4px;">27.1 V | 0.13 A</div>
                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">Power: 3.52 W</div>
                    </div>
                    <div class="card" style="background: #090d16;">
                        <div style="font-size: 12px; font-weight: 600; color: var(--primary);">Channel 3</div>
                        <div style="font-size: 18px; font-weight: 700; margin-top: 4px;">26.1 V | 0.12 A</div>
                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">Power: 3.13 W</div>
                    </div>
                    <div class="card" style="background: #090d16;">
                        <div style="font-size: 12px; font-weight: 600; color: var(--primary);">Channel 4</div>
                        <div style="font-size: 18px; font-weight: 700; margin-top: 4px;">26.1 V | 0.12 A</div>
                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">Power: 3.13 W</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- 06. MODEL EVALUATION -->
        <section class="tab-section" id="sec-5">
            <div class="top-bar">
                <div>
                    <div class="tab-title">Model Evaluation & Cascaded Confusion Matrices</div>
                    <div class="tab-subtitle">Empirical performance evaluated across 348,899 test observations without data leakage.</div>
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <div class="card-title" style="margin-bottom: 12px;">Cascaded Fault Category Confusion Matrix</div>
                    <div class="img-container">
                        <img src="/reports/figures/cascaded_category_confusion_matrix.png" alt="Category Confusion Matrix" onerror="this.alt='Matrix will render upon test completion';">
                    </div>
                </div>
                <div class="card">
                    <div class="card-title" style="margin-bottom: 12px;">Cascaded Fault Severity Confusion Matrix</div>
                    <div class="img-container">
                        <img src="/reports/figures/cascaded_severity_confusion_matrix.png" alt="Severity Confusion Matrix" onerror="this.alt='Matrix will render upon test completion';">
                    </div>
                </div>
            </div>
        </section>

        <!-- 07. DATASET ANALYTICS -->
        <section class="tab-section" id="sec-6">
            <div class="top-bar">
                <div>
                    <div class="tab-title">Zenodo Controlled-Fault Benchmark Analytics</div>
                    <div class="tab-subtitle">1,814,400 per-second observations across 3 physical monitoring stations.</div>
                </div>
            </div>

            <div class="grid-3">
                <div class="card">
                    <div class="card-title">Station 1: Laboratory</div>
                    <div class="card-val">604,800</div>
                    <div class="card-sub">Baseline load ~661 W | 92 Fault Events</div>
                </div>
                <div class="card">
                    <div class="card-title">Station 2: Residential</div>
                    <div class="card-val">604,800</div>
                    <div class="card-sub">Medium load ~685 W | 84 Fault Events</div>
                </div>
                <div class="card">
                    <div class="card-title">Station 3: Industrial</div>
                    <div class="card-val">604,800</div>
                    <div class="card-sub">Inductive load ~4,255 W | 78 Fault Events</div>
                </div>
            </div>
        </section>

        <!-- 08. SYSTEM & API -->
        <section class="tab-section" id="sec-7">
            <div class="top-bar">
                <div>
                    <div class="tab-title">System & API Status</div>
                    <div class="tab-subtitle">FastAPI REST interface, active model hashes, and operational feature schema.</div>
                </div>
            </div>

            <div class="card">
                <div class="card-title" style="margin-bottom: 14px;">Unified 16-Feature Operational Schema Inspector</div>
                <pre style="background: #090d16; padding: 16px; border-radius: 6px; font-size: 13px; color: #38bdf8; overflow-x: auto;">
1.  voltage_V                   (AC Line Voltage in Volts)
2.  current_A                   (AC Line Current in Amperes)
3.  power_W                     (Active Real Power in Watts)
4.  frequency_Hz                (Grid Supply Frequency in Hertz)
5.  power_factor                (Displacement Power Factor cos φ)
6.  apparent_power_VA           (S = |V| * |I|)
7.  reactive_power_var          (Q = sqrt(max(0, S² - P²)))
8.  power_active_apparent_ratio (Active to Apparent Ratio |P| / S)
9.  voltage_deviation_pct       ((V - 220) / 220 * 100)
10. frequency_deviation_pct     ((f - 50) / 50 * 100)
11. d_voltage_dt                (Causal Voltage Derivative |V_t - V_{t-1}|)
12. d_current_dt                (Causal Current Derivative |I_t - I_{t-1}|)
13. d_power_dt                  (Causal Power Derivative |P_t - P_{t-1}|)
14. rolling_std_v5              (5-Second Moving Voltage Volatility)
15. rolling_std_i5              (5-Second Moving Current Volatility)
16. power_to_baseline_ratio     (Station-Invariant Ratio P_t / P_{30s})
                </pre>
            </div>
        </section>
    </main>

    <script>
        // Navigation Switcher
        function switchNav(idx) {
            document.querySelectorAll('.nav-item').forEach((item, i) => {
                item.classList.toggle('active', i === idx);
            });
            document.querySelectorAll('.tab-section').forEach((sec, i) => {
                sec.classList.toggle('active', i === idx);
            });
        }

        // Check FastAPI Backend Health
        let isBackendOnline = false;
        async function checkApiHealth() {
            try {
                const res = await fetch("http://localhost:8000/health", { cache: "no-store" });
                if (res.ok) {
                    isBackendOnline = true;
                    document.getElementById('api-status-pill').className = "status-pill";
                    document.getElementById('api-status-text').innerText = "Backend Connected";
                    document.getElementById('disconnect-banner').style.display = "none";
                } else {
                    throw new Error();
                }
            } catch (e) {
                isBackendOnline = false;
                document.getElementById('api-status-pill').className = "status-pill offline";
                document.getElementById('api-status-text').innerText = "Backend Disconnected";
                document.getElementById('disconnect-banner').style.display = "flex";
            }
        }
        setInterval(checkApiHealth, 3000);
        checkApiHealth();

        // Preset Scenario Setter
        function setPreset(v, i, p, f, pf) {
            document.getElementById('diag_v').value = v;
            document.getElementById('diag_i').value = i;
            document.getElementById('diag_p').value = p;
            document.getElementById('diag_f').value = f;
            document.getElementById('diag_pf').value = pf;
            runManualDiagnosis();
        }

        // Manual Diagnosis Execution
        async function runManualDiagnosis() {
            if (!isBackendOnline) {
                alert("Cannot run diagnosis: FastAPI ML Engine is offline on port 8000.");
                return;
            }

            const payload = {
                voltage_V: parseFloat(document.getElementById('diag_v').value),
                current_A: parseFloat(document.getElementById('diag_i').value),
                power_W: parseFloat(document.getElementById('diag_p').value),
                frequency_Hz: parseFloat(document.getElementById('diag_f').value),
                power_factor: parseFloat(document.getElementById('diag_pf').value)
            };

            try {
                const res = await fetch("http://localhost:8000/predict/two-stage", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                document.getElementById('diag-results-box').style.display = "block";
                
                // Stage 1
                document.getElementById('diag-s1-val').innerText = (data.is_anomaly ? "Anomaly Flagged (" : "Nominal (") + (data.stage1_probability * 100).toFixed(1) + "%)";
                document.getElementById('diag-s1-val').style.color = data.is_anomaly ? "var(--danger)" : "var(--success)";
                
                // Stage 2
                document.getElementById('diag-cat-val').innerText = data.fault_category;
                document.getElementById('diag-cat-conf').innerText = "Confidence: " + (data.category_confidence * 100).toFixed(1) + "%";
                document.getElementById('diag-sev-val').innerText = data.fault_severity;
                document.getElementById('diag-sev-conf').innerText = "Confidence: " + (data.severity_confidence * 100).toFixed(1) + "%";

                const isCrit = data.fault_severity.toLowerCase() === "critical";
                document.getElementById('diag-badge').className = "badge " + (data.is_anomaly ? (isCrit ? "badge-danger" : "badge-warning") : "badge-normal");
                document.getElementById('diag-badge').innerText = data.is_anomaly ? (isCrit ? "Critical Hazard" : "Anomaly Detected") : "Normal Baseline";

                const triggers = data.physics_attributions || [];
                document.getElementById('diag-triggers').innerHTML = triggers.length ? triggers.map(t => `<div>• ${t}</div>`).join('') : "• All measurements conform to statutory grid bounds (220V ±10%, 50Hz ±1%).";
            } catch (err) {
                alert("Error during model inference: " + err.message);
            }
        }

        // Live 1 Hz Telemetry Stream Simulator
        let streamInterval = null;
        let waveBuffer = Array(60).fill(220.0);

        function drawWaveform() {
            const canvas = document.getElementById('waveformCanvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw grid guides
            ctx.strokeStyle = "#1e293b";
            ctx.lineWidth = 1;
            for (let y = 20; y < canvas.height; y += 30) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }

            // Draw live voltage waveform
            ctx.strokeStyle = "#38bdf8";
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            const step = canvas.width / (waveBuffer.length - 1);
            waveBuffer.forEach((val, i) => {
                const y = canvas.height - ((val - 140) / 120) * canvas.height;
                if (i === 0) ctx.moveTo(0, y);
                else ctx.lineTo(i * step, y);
            });
            ctx.stroke();
        }

        async function tickStream() {
            if (!isBackendOnline) return;
            try {
                const res = await fetch("http://localhost:8000/stream/step");
                const data = await res.json();
                const tel = data.telemetry;
                const diag = data.diagnosis;

                document.getElementById('live-v').innerText = tel.voltage_V.toFixed(1) + " V";
                document.getElementById('live-i').innerText = tel.current_A.toFixed(1) + " A";
                document.getElementById('live-p').innerText = tel.power_W.toLocaleString() + " W";
                
                document.getElementById('live-status').innerText = diag.is_anomaly ? diag.fault_category.toUpperCase() : "NOMINAL";
                document.getElementById('live-status').style.color = diag.is_anomaly ? "var(--danger)" : "var(--success)";
                document.getElementById('live-prob').innerText = "Stage 1 Anom Prob: " + (diag.stage1_probability * 100).toFixed(1) + "%";

                waveBuffer.shift();
                waveBuffer.push(tel.voltage_V);
                drawWaveform();
            } catch (e) {}
        }

        function toggleStream() {
            if (streamInterval) {
                clearInterval(streamInterval);
                streamInterval = null;
                document.getElementById('btn-stream-toggle').innerText = "▶ Start 1 Hz Stream";
            } else {
                streamInterval = setInterval(tickStream, 1000);
                document.getElementById('btn-stream-toggle').innerText = "⏸ Pause Stream";
            }
        }

        async function injectFault(type) {
            if (!isBackendOnline) return;
            await fetch("http://localhost:8000/stream/inject", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ fault_type: type })
            });
        }

        function resetStream() {
            waveBuffer = Array(60).fill(220.0);
            drawWaveform();
            injectFault('normal');
        }

        drawWaveform();
    </script>
</body>
</html>
"""

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve reports figures directly
        if self.path.startswith("/reports/figures/"):
            file_path = BASE_DIR / self.path.lstrip("/")
            if file_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # Serve Dashboard HTML
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

def run_server():
    server_address = ("", PORT)
    with socketserver.TCPServer(server_address, DashboardHandler) as httpd:
        print(f"⚡ PowerGuard AI Live at: http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
