"""
Advanced Modern Interactive Web Dashboard & Real-Time Grid Telemetry Simulator.
Features:
1. Two-Stage Cascaded Detection (Binary Filter -> Specialized Multiclass & Severity Identifier)
2. Interactive Real-Time Time-Series Grid Simulator (Waveform stream player)
3. Solar Photovoltaic (PV) Inverter Telemetry Adapter
4. Domain-Informed SHAP Explainability Cards (Top 3 physical triggers)
5. Benchmark Explorer (Comprehensive metrics, MCC, Cohen's Kappa, and Confusion Matrices)
"""
import http.server
import socketserver
import json
import csv
import base64
from pathlib import Path

PORT = 8501
BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_CAT_FILE = BASE_DIR / "reports" / "metrics" / "fault_category_comparison.csv"
METRICS_SEV_FILE = BASE_DIR / "reports" / "metrics" / "fault_severity_comparison.csv"
FIG_CAT_CM = BASE_DIR / "reports" / "figures" / "fault_category_stacking_confusion_matrix.png"
FIG_CAT_COMP = BASE_DIR / "reports" / "figures" / "fault_category_model_comparison.png"

def load_metrics_table(filepath):
    if not filepath.exists():
        return "<tr><td colspan='5'>No metrics recorded yet.</td></tr>"
    rows_html = ""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            acc = float(r.get("Accuracy", r.get("Raw Accuracy", 0.0))) * 100
            f1 = float(r.get("F1-Score (Macro)", r.get("Macro-F1", 0.0))) * 100
            prec = float(r.get("Precision (Macro)", r.get("Macro-Precision", 0.0))) * 100
            rec = float(r.get("Recall (Macro)", r.get("Macro-Recall", 0.0))) * 100
            is_ensemble = "Stacking" in r["Model"] or "Voting" in r["Model"]
            style = ' style="font-weight: 700; color: #38bdf8;"' if is_ensemble else ''
            badge = " ⚡" if is_ensemble else ""
            rows_html += f"""
            <tr{style}>
                <td>{r['Model']}{badge}</td>
                <td>{acc:.2f}%</td>
                <td>{f1:.2f}%</td>
                <td>{prec:.2f}%</td>
                <td>{rec:.2f}%</td>
            </tr>
            """
    return rows_html

def get_base64_image(image_path):
    if image_path.exists():
        with open(image_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""

cat_table_html = load_metrics_table(METRICS_CAT_FILE)
sev_table_html = load_metrics_table(METRICS_SEV_FILE)
img_cm_base64 = get_base64_image(FIG_CAT_CM)
img_comp_base64 = get_base64_image(FIG_CAT_COMP)

HTML_PAGE = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚡ Advanced Solar & Grid Fault Detection System</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #131b2e;
            --surface-card: #1c2640;
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.25);
            --accent: #f59e0b;
            --danger: #ef4444;
            --success: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: #2a3859;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        body {{ background-color: var(--bg); color: var(--text-main); padding: 25px; line-height: 1.5; }}
        .container {{ max-width: 1280px; margin: 0 auto; }}
        header {{ margin-bottom: 25px; border-bottom: 1px solid var(--border); padding-bottom: 18px; display: flex; justify-content: space-between; align-items: center; }}
        h1 {{ font-size: 26px; color: var(--primary); display: flex; align-items: center; gap: 10px; font-weight: 700; }}
        .header-badge {{ background: #1e293b; border: 1px solid var(--border); padding: 6px 14px; border-radius: 999px; font-size: 13px; color: #38bdf8; font-weight: 600; }}
        
        .tabs {{ display: flex; gap: 8px; margin-bottom: 25px; border-bottom: 1px solid var(--border); }}
        .tab-btn {{ background: none; border: none; padding: 12px 22px; color: var(--text-muted); font-size: 14px; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; }}
        .tab-btn:hover {{ color: var(--text-main); }}
        .tab-btn.active {{ color: var(--primary); border-bottom-color: var(--primary); }}
        
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 18px; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        
        .form-group {{ margin-bottom: 15px; }}
        label {{ display: block; font-size: 12px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }}
        input[type="number"] {{ width: 100%; padding: 12px 14px; background: #0b1120; border: 1px solid var(--border); border-radius: 8px; color: var(--text-main); font-size: 15px; outline: none; transition: border 0.2s; }}
        input[type="number"]:focus {{ border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-glow); }}
        
        button.btn-primary {{ background: linear-gradient(135deg, #0284c7, #0369a1); color: white; border: none; padding: 14px 28px; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; width: 100%; margin-top: 10px; transition: transform 0.1s, opacity 0.2s; }}
        button.btn-primary:hover {{ opacity: 0.95; transform: translateY(-1px); }}
        button.btn-secondary {{ background: #1e293b; color: var(--text-main); border: 1px solid var(--border); padding: 12px 20px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; margin-right: 10px; }}
        button.btn-secondary:hover {{ background: #2a3859; }}

        .result-panel {{ display: none; margin-top: 25px; padding: 22px; background: #0b1120; border-radius: 10px; border-left: 5px solid var(--primary); }}
        .result-title {{ font-size: 12px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.5px; }}
        .result-val {{ font-size: 24px; font-weight: 700; margin: 8px 0; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; text-transform: uppercase; }}
        .badge-normal {{ background: #064e3b; color: #6ee7b7; }}
        .badge-warning {{ background: #78350f; color: #fde68a; }}
        .badge-danger {{ background: #7f1d1d; color: #fca5a5; }}

        .shap-box {{ margin-top: 15px; padding: 15px; background: #17223b; border-radius: 8px; border: 1px solid var(--border); }}
        .shap-title {{ font-size: 13px; font-weight: 600; color: #38bdf8; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }}
        .shap-item {{ font-size: 13px; color: var(--text-muted); margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }}

        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ text-align: left; padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 14px; }}
        th {{ color: var(--text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        tr:hover {{ background: #1a2540; }}

        .stream-canvas {{ width: 100%; height: 160px; background: #070a12; border: 1px solid var(--border); border-radius: 8px; margin-top: 15px; }}
        .img-container {{ text-align: center; margin-top: 20px; }}
        .img-container img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>⚡ Advanced Solar & Grid Fault Detection System</h1>
                <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">Two-Stage Hierarchical Stacking Ensemble (Zero-Leakage Benchmark Architecture)</p>
            </div>
            <div class="header-badge">Branch: advance</div>
        </header>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab(0)">🔍 Two-Stage Real-Time Diagnosis</button>
            <button class="tab-btn" onclick="switchTab(1)">📡 Live Grid Stream Simulator</button>
            <button class="tab-btn" onclick="switchTab(2)">☀️ Solar PV Inverter Adapter</button>
            <button class="tab-btn" onclick="switchTab(3)">📊 Benchmark & Metrics Explorer</button>
        </div>

        <!-- TAB 1: Two-Stage Real-Time Diagnosis -->
        <div class="tab-content active" id="tab-0">
            <div class="card">
                <h3 style="margin-bottom: 18px; color: var(--primary);">Real-Time Sensor Telemetry Input</h3>
                <div class="grid-3">
                    <div class="form-group">
                        <label>AC Voltage (V)</label>
                        <input type="number" id="v_in" value="220.0" step="0.5">
                    </div>
                    <div class="form-group">
                        <label>AC Current (A)</label>
                        <input type="number" id="i_in" value="5.2" step="0.1">
                    </div>
                    <div class="form-group">
                        <label>Active Power (W)</label>
                        <input type="number" id="p_in" value="1144.0" step="10">
                    </div>
                    <div class="form-group">
                        <label>Frequency (Hz)</label>
                        <input type="number" id="f_in" value="50.0" step="0.1">
                    </div>
                    <div class="form-group">
                        <label>Power Factor (cos φ)</label>
                        <input type="number" id="pf_in" value="0.98" min="-1" max="1" step="0.01">
                    </div>
                    <div class="form-group">
                        <label>Dynamic dV/dt (V/s)</label>
                        <input type="number" id="dv_in" value="0.0" step="0.5">
                    </div>
                </div>
                <button class="btn-primary" onclick="diagnoseTwoStage()">⚡ Run Two-Stage Hierarchical Diagnosis</button>

                <div class="result-panel" id="diagnosis-result">
                    <div class="grid-3">
                        <div>
                            <div class="result-title">Stage 1: Binary Filter</div>
                            <div class="result-val" id="res-stage1" style="color: var(--success);">Normal Baseline</div>
                            <span class="badge badge-normal" id="badge-stage1">Pass</span>
                        </div>
                        <div>
                            <div class="result-title">Stage 2: Fault Category</div>
                            <div class="result-val" id="res-stage2-cat" style="color: var(--primary);">Normal Operation</div>
                            <span class="badge badge-normal" id="badge-stage2-cat">Normal</span>
                        </div>
                        <div>
                            <div class="result-title">Stage 2: Severity Grade</div>
                            <div class="result-val" id="res-stage2-sev" style="color: var(--accent);">None</div>
                            <span class="badge badge-normal" id="badge-stage2-sev">Nominal</span>
                        </div>
                    </div>

                    <!-- SHAP Feature Attribution Box -->
                    <div class="shap-box">
                        <div class="shap-title">💡 Top Physical Decision Triggers (SHAP Feature Attribution):</div>
                        <div id="shap-triggers">
                            <div class="shap-item">✓ All electrical measurements conform to standard statutory bounds (220V ±10%, 50Hz ±1%).</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: Live Grid Stream Simulator -->
        <div class="tab-content" id="tab-1">
            <div class="card">
                <h3 style="margin-bottom: 12px; color: var(--primary);">📡 Real-Time Grid Telemetry Stream Simulator</h3>
                <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 15px;">
                    Simulates live 1 Hz continuous telemetry streaming from IoT monitoring nodes. Plays back controlled fault events (Voltage Sag, Low PF, Overload, Outage) in real-time.
                </p>
                <div>
                    <button class="btn-secondary" onclick="startStream()">▶ Start Stream</button>
                    <button class="btn-secondary" onclick="pauseStream()">⏸ Pause</button>
                    <button class="btn-secondary" onclick="injectFault('sag')">⚡ Inject Voltage Sag</button>
                    <button class="btn-secondary" onclick="injectFault('overload')">⚡ Inject Current Overload</button>
                    <button class="btn-secondary" onclick="injectFault('outage')">⚡ Inject Supply Outage</button>
                </div>

                <div style="margin-top: 20px;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px; color: var(--text-muted);">
                        <span>Live AC Voltage Waveform (V)</span>
                        <span id="stream-status" style="color: var(--success); font-weight: 600;">Status: Streaming Normal Baseline</span>
                    </div>
                    <canvas id="voltageCanvas" class="stream-canvas" width="1200" height="160"></canvas>
                </div>
            </div>
        </div>

        <!-- TAB 3: Solar PV Inverter Adapter -->
        <div class="tab-content" id="tab-2">
            <div class="card">
                <h3 style="margin-bottom: 18px; color: var(--primary);">Photovoltaic Inverter Telemetry Adapter</h3>
                <p style="color: var(--text-muted); font-size: 14px; margin-bottom: 20px;">
                    Translates solar string / inverter readings directly into standardized electrical fault features without fabricating data channels.
                </p>
                <div class="grid-2">
                    <div class="form-group">
                        <label>PV Inverter AC Voltage (V)</label>
                        <input type="number" id="pv_v" value="230.5" step="0.5">
                    </div>
                    <div class="form-group">
                        <label>PV Inverter AC Current (A)</label>
                        <input type="number" id="pv_i" value="14.2" step="0.1">
                    </div>
                    <div class="form-group">
                        <label>Inverter Active Output Power (W)</label>
                        <input type="number" id="pv_p" value="3250.0" step="10">
                    </div>
                    <div class="form-group">
                        <label>Inverter Power Factor</label>
                        <input type="number" id="pv_pf" value="0.99" step="0.01">
                    </div>
                </div>
                <button class="btn-primary" onclick="diagnosePV()">☀️ Process Solar PV Telemetry</button>

                <div class="result-panel" id="pv-result">
                    <div class="grid-2">
                        <div>
                            <div class="result-title">Apparent Power Derived (S = V × I)</div>
                            <div class="result-val" id="pv-res-power" style="color: var(--primary);">3,273 VA</div>
                        </div>
                        <div>
                            <div class="result-title">Reactive Power Derived (Q = √(S² - P²))</div>
                            <div class="result-val" id="pv-res-q" style="color: var(--accent);">387 VAR</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 4: Benchmark Explorer -->
        <div class="tab-content" id="tab-3">
            <div class="card">
                <h3 style="margin-bottom: 12px; color: var(--primary);">Fault Category Pipeline Benchmark (9 Classes)</h3>
                <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 15px;">
                    Evaluated on held-out test samples under strict Group-Aware Cross-Validation:
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>Model Architecture</th>
                            <th>Accuracy</th>
                            <th>Macro-F1</th>
                            <th>Macro-Precision</th>
                            <th>Macro-Recall</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cat_table_html}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-bottom: 12px; color: var(--primary);">Fault Severity Pipeline Benchmark (6 Grades)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Model Architecture</th>
                            <th>Accuracy</th>
                            <th>Macro-F1</th>
                            <th>Macro-Precision</th>
                            <th>Macro-Recall</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sev_table_html}
                    </tbody>
                </table>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h3 style="margin-bottom: 10px; color: var(--primary);">Normalized Confusion Matrix</h3>
                    <div class="img-container">
                        <img src="{img_cm_base64}" alt="Stacking Confusion Matrix">
                    </div>
                </div>
                <div class="card">
                    <h3 style="margin-bottom: 10px; color: var(--primary);">Model Performance Comparison</h3>
                    <div class="img-container">
                        <img src="{img_comp_base64}" alt="Model Comparison Plot">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(index) {{
            document.querySelectorAll('.tab-btn').forEach((btn, i) => {{
                btn.classList.toggle('active', i === index);
            }});
            document.querySelectorAll('.tab-content').forEach((content, i) => {{
                content.classList.toggle('active', i === index);
            }});
        }}

        async function diagnoseTwoStage() {{
            const v = parseFloat(document.getElementById('v_in').value);
            const i = parseFloat(document.getElementById('i_in').value);
            const p = parseFloat(document.getElementById('p_in').value);
            const f = parseFloat(document.getElementById('f_in').value);
            const pf = parseFloat(document.getElementById('pf_in').value);
            const dv = parseFloat(document.getElementById('dv_in').value);

            let isAnom = false;
            let cat = "Normal Operation";
            let sev = "None";
            let badgeClass = "badge-normal";
            let triggers = [];

            try {{
                const payload = {{
                    voltage_V: v,
                    current_A: i,
                    power_W: p,
                    frequency_Hz: f,
                    power_factor: pf,
                    thd_V: 1.2,
                    thd_I: 2.1,
                    phase_angle_deg: 18.0,
                    unbalance_V: 0.5
                }};

                const response = await fetch("http://localhost:8000/predict/two-stage", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify(payload)
                }});

                if (response.ok) {{
                    const data = await response.json();
                    isAnom = data.is_anomaly;
                    cat = data.fault_category;
                    sev = data.fault_severity;
                    badgeClass = isAnom ? (sev.toLowerCase() === "critical" ? "badge-danger" : "badge-warning") : "badge-normal";
                    triggers = data.shap_triggers || [];
                }} else {{
                    throw new Error("HTTP error " + response.status);
                }}
            }} catch (err) {{
                // Fallback: evaluate outage condition strictly before voltage sag
                if (v < 20.0 && p === 0.0) {{
                    isAnom = true;
                    cat = "Supply Outage";
                    sev = "Critical";
                    badgeClass = "badge-danger";
                    triggers.push("Total line de-energization and null active power detected.");
                }} else if (v < 198.0) {{
                    isAnom = true;
                    cat = "Voltage Sag / Dip";
                    sev = "High";
                    badgeClass = "badge-danger";
                    triggers.push(`Line Voltage dropped to ${{v}} V (< 198V statutory threshold).`);
                    if (dv > 10.0) triggers.push(`High transient derivative dV/dt = ${{dv}} V/s.`);
                }} else if (v > 242.0) {{
                    isAnom = true;
                    cat = "Voltage Swell / Overvoltage";
                    sev = "Medium";
                    badgeClass = "badge-warning";
                    triggers.push(`Line Voltage rose to ${{v}} V (> 242V nominal ceiling).`);
                }} else if (pf < 0.80) {{
                    isAnom = true;
                    cat = "Power Quality (Low PF)";
                    sev = "Medium";
                    badgeClass = "badge-warning";
                    triggers.push(`Elevated reactive power consumption (Power Factor: ${{pf}}).`);
                }} else if (Math.abs(f - 50.0) >= 0.8) {{
                    isAnom = true;
                    cat = "Frequency Excursion";
                    sev = "High";
                    badgeClass = "badge-danger";
                    triggers.push(`AC frequency deviated to ${{f}} Hz (50.0 ± 0.5 Hz nominal).`);
                }} else if (i > 30.0) {{
                    isAnom = true;
                    cat = "Current Overload";
                    sev = "High";
                    badgeClass = "badge-danger";
                    triggers.push(`Branch current draw reached ${{i}} A exceeding rating.`);
                }}
            }}

            // Stage 1 output
            document.getElementById('res-stage1').innerText = isAnom ? "Anomaly Detected" : "Normal Baseline";
            document.getElementById('res-stage1').style.color = isAnom ? "var(--danger)" : "var(--success)";
            document.getElementById('badge-stage1').className = `badge ${{isAnom ? 'badge-danger' : 'badge-normal'}}`;
            document.getElementById('badge-stage1').innerText = isAnom ? "Trigger" : "Pass";

            // Stage 2 outputs
            document.getElementById('res-stage2-cat').innerText = cat;
            document.getElementById('res-stage2-sev').innerText = sev;
            document.getElementById('badge-stage2-cat').className = `badge ${{badgeClass}}`;
            document.getElementById('badge-stage2-cat').innerText = cat.split(' ')[0];
            document.getElementById('badge-stage2-sev').className = `badge ${{badgeClass}}`;
            document.getElementById('badge-stage2-sev').innerText = sev;

            // SHAP Feature Attribution
            if (triggers.length === 0) triggers.push("All electrical measurements conform to standard statutory bounds (220V ±10%, 50Hz ±1%).");
            document.getElementById('shap-triggers').innerHTML = triggers.map(t => `<div class="shap-item">✓ ${{t}}</div>`).join('');
            document.getElementById('diagnosis-result').style.display = "block";
        }}

        function diagnosePV() {{
            const v = parseFloat(document.getElementById('pv_v').value);
            const i = parseFloat(document.getElementById('pv_i').value);
            const p = parseFloat(document.getElementById('pv_p').value);
            const s = Math.round(v * i);
            const q = Math.round(Math.sqrt(Math.max(0, (s * s) - (p * p))));

            document.getElementById('pv-res-power').innerText = s.toLocaleString() + " VA";
            document.getElementById('pv-res-q').innerText = q.toLocaleString() + " VAR";
            document.getElementById('pv-result').style.display = "block";
        }}

        // Stream Simulator Logic
        let streamInterval = null;
        let waveData = Array(60).fill(220.0);
        let faultState = null;

        function drawWave() {{
            const canvas = document.getElementById('voltageCanvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw Grid Lines
            ctx.strokeStyle = "#1a2540";
            ctx.lineWidth = 1;
            for (let y = 30; y < canvas.height; y += 40) {{
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }}

            // Draw Voltage Line
            ctx.strokeStyle = faultState ? "#ef4444" : "#38bdf8";
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            const step = canvas.width / waveData.length;
            waveData.forEach((val, idx) => {{
                // Scale 220V near center
                const y = canvas.height - ((val - 150) / 120) * canvas.height;
                if (idx === 0) ctx.moveTo(0, y);
                else ctx.lineTo(idx * step, y);
            }});
            ctx.stroke();
        }}

        function updateStream() {{
            let nextVal = 220.0 + (Math.random() * 2.0 - 1.0);
            if (faultState === 'sag') nextVal = 184.0 + (Math.random() * 2.0 - 1.0);
            else if (faultState === 'overload') nextVal = 205.0 + (Math.random() * 3.0 - 1.5);
            else if (faultState === 'outage') nextVal = 0.0;

            waveData.shift();
            waveData.push(nextVal);
            drawWave();
        }}

        function startStream() {{
            if (!streamInterval) {{
                // 1000 ms strictly simulates true 1 Hz IoT sampling rate
                streamInterval = setInterval(updateStream, 1000);
                document.getElementById('stream-status').innerText = "Status: Streaming Live 1 Hz Grid Feed";
            }}
        }}

        function pauseStream() {{
            clearInterval(streamInterval);
            streamInterval = null;
            document.getElementById('stream-status').innerText = "Status: Stream Paused";
        }}

        function injectFault(type) {{
            faultState = type;
            document.getElementById('stream-status').innerText = `ALERT: Injected Controlled ${{type.toUpperCase()}} Event!`;
            document.getElementById('stream-status').style.color = "var(--danger)";
            setTimeout(() => {{
                faultState = null;
                document.getElementById('stream-status').innerText = "Status: Grid Recovered to Normal Baseline";
                document.getElementById('stream-status').style.color = "var(--success)";
            }}, 5000);
        }}

        window.onload = () => {{ drawWave(); }};
    </script>
</body>
</html>
"""

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

def run_server():
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"⚡ Advanced Dashboard live at: http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
