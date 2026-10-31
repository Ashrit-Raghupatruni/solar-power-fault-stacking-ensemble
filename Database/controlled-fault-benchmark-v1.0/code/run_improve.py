"""
run_improve.py — Cai thien tren DU LIEU SU CO THAT (3 tram x 604,800 dong).

Giu NGUYEN methodology cua improve_pipeline.py:
  - GB: GradientBoostingClassifier(300, depth5, lr0.08, subsample0.8, rs42)
  - XGB: 400 trees depth6, scale_pos_weight, GPU(cuda)
  - LGB: 500 trees, num_leaves63, class_weight balanced, GPU
  - 7-fold leave-one-segment(day)-out CV, event post-processing (merge30/minlen15),
    best Event-F1 sweep (37 thresholds), AUROC, AUPRC.
  - BASE 7-feat (PE+RPD) vs EXTENDED (+spectral/FFT/wavelet/multi-scale).

THEM (khong doi so):
  - Chay SONG SONG cac fold GB tren nhieu nhan CPU (cac fold doc lap, rs co dinh).
  - Per-type AUROC (cau chuyen frequency_deviation/flicker/intermittent).
  - Luu KET QUA TANG DAN: ket_qua_cai_thien.csv + ket_qua_per_type.csv.

Output:
  ket_qua_cai_thien.csv   -> Station,FeatureSet,Model,EvF1,EvDR,EvPrec,AUROC,AUPRC
  ket_qua_per_type.csv    -> Station,FeatureSet,Model,anomaly_type,n_pos,AUROC
"""
import os, sys, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed

try: import pywt; HAS_PYWT=True
except Exception: HAS_PYWT=False
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False
try: import lightgbm as lgb; HAS_LGB=True
except Exception: HAS_LGB=False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "02_processed_features")
STATIONS = [("station1_normal_features.csv", "Station 1 (Lab)"),
            ("station2_medium_features.csv", "Station 2 (Resid.)"),
            ("station3_high_features.csv", "Station 3 (Indust.)")]
RPD_W = 32
N_JOBS_GB = min(7, os.cpu_count() or 4)   # auto: 7 fold, gioi han boi so nhan may
OUT_MAIN = os.path.join(HERE, "ket_qua_cai_thien.csv")
OUT_PT   = os.path.join(HERE, "ket_qua_per_type.csv")

# ---------------- event scoring (identical to paper) ----------------
def find_events(a):
    ev, i = [], 0
    while i < len(a):
        if a[i] == 1:
            j = i
            while j < len(a) and a[j] == 1: j += 1
            ev.append((i, j-1)); i = j
        else: i += 1
    return ev

def post(a, merge=30, ml=15):
    r = a.copy(); ev = find_events(r)
    for k in range(len(ev)-1):
        _, e1 = ev[k]; s2, _ = ev[k+1]
        if 0 < s2-e1-1 <= merge: r[e1+1:s2] = 1
    for s, e in find_events(r):
        if e-s+1 < ml: r[s:e+1] = 0
    return r

def evf1(d, g):
    ge, de = find_events(g), find_events(d)
    if not ge or not de: return 0.0, 0.0, 0.0
    det = sum(1 for s, e in ge if np.any(d[s:e+1] == 1)); dr = det/len(ge)*100
    fe = sum(1 for s, e in de if not np.any(g[s:e+1] == 1)); pr = (len(de)-fe)/len(de)*100
    return (2*pr*dr/(pr+dr) if pr+dr else 0.0), dr, pr

def best_event_f1(score, y, mask):
    best = (0, 0, 0, 0.5)
    for thr in np.linspace(0.05, 0.95, 37):
        d = post(((score >= thr) & mask).astype(int))
        f, dr, pr = evf1(d, y)
        if f > best[0]: best = (f, dr, pr, thr)
    return best

# ---------------- (A) extended spectral / multi-scale features ----------------
def add_features(df):
    P = df["power_W"].values.astype(float)
    F = df["frequency_Hz"].values.astype(float) if "frequency_Hz" in df else np.full(len(P), 50.0)
    s = pd.Series(P)
    med = s.rolling(RPD_W, min_periods=1).median().values
    df["RPD"] = np.abs(P - med) / (np.abs(med) + 1e-9)
    for w in (16, 64, 256):
        df[f"CV_{w}"] = (s.rolling(w, min_periods=1).std() /
                         (s.rolling(w, min_periods=1).mean().abs() + 1e-9)).values
    df["dP"] = np.abs(np.diff(P, prepend=P[0]))
    df["d2P"] = np.abs(np.diff(P, n=2, prepend=[P[0], P[0]]))
    WIN = 64
    bandhi = np.zeros(len(P)); flick = np.zeros(len(P))
    for i in range(WIN, len(P)):
        w = P[i-WIN:i]; w = w - w.mean()
        sp = np.abs(np.fft.rfft(w * np.hanning(WIN)))**2
        tot = sp.sum() + 1e-9
        bandhi[i] = sp[len(sp)//2:].sum() / tot
        flick[i] = sp[1:6].sum() / tot
    df["FFT_highband"] = bandhi
    df["FFT_flicker"] = flick
    df["freq_dev"] = np.abs(F - 50.0)
    if HAS_PYWT:
        we = np.zeros(len(P))
        for i in range(WIN, len(P)):
            c = pywt.wavedec(P[i-WIN:i], "db4", level=3)
            we[i] = np.sum(c[1]**2) / (np.sum(P[i-WIN:i]**2) + 1e-9)
        df["wavelet_E"] = we
    return df

BASE_FEATS = ["PE_m4", "MSPE", "CV", "delta_PF", "RPD", "PE_m3", "PE_m5"]
EXT_FEATS = BASE_FEATS + ["CV_16", "CV_64", "CV_256", "dP", "d2P",
                          "FFT_highband", "FFT_flicker", "freq_dev"] + \
            (["wavelet_E"] if HAS_PYWT else [])

# ---------------- per-fold workers (independent -> parallelizable) ----------------
def _fold_gb(X, y, seg, ts):
    te = seg == ts; tr = ~te
    if y[tr].sum() < 3 or te.sum() < 10: return None
    sc = StandardScaler().fit(X[tr])
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                     subsample=0.8, random_state=42)
    clf.fit(sc.transform(X[tr]), y[tr])
    return np.where(te)[0], clf.predict_proba(sc.transform(X[te]))[:, 1]

def _fold_other(X, y, seg, ts, model):
    te = seg == ts; tr = ~te
    if y[tr].sum() < 3 or te.sum() < 10: return None
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    spw = max(1.0, (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    if model == "XGB":
        try:
            clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                    tree_method="hist", device="cuda", eval_metric="aucpr",
                                    random_state=42)
            clf.fit(Xtr, y[tr])
        except Exception as e:
            print(f"      [XGB GPU fail -> CPU] {repr(e)[:120]}")
            clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                    tree_method="hist", eval_metric="aucpr", random_state=42)
            clf.fit(Xtr, y[tr])
    else:  # LGB
        try:
            clf = lgb.LGBMClassifier(n_estimators=500, max_depth=-1, num_leaves=63,
                                     learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                     class_weight="balanced", device_type="gpu",
                                     random_state=42, verbose=-1)
            clf.fit(Xtr, y[tr])
        except Exception as e:
            print(f"      [LGB GPU fail -> CPU] {repr(e)[:120]}")
            clf = lgb.LGBMClassifier(n_estimators=500, max_depth=-1, num_leaves=63,
                                     learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                     class_weight="balanced", random_state=42, verbose=-1)
            clf.fit(Xtr, y[tr])
    return np.where(te)[0], clf.predict_proba(Xte)[:, 1]

def cv_oof(X, y, seg, model):
    oof = np.full(len(y), np.nan)
    segs = np.unique(seg)
    if model == "GB":
        res = Parallel(n_jobs=N_JOBS_GB, backend="loky")(
            delayed(_fold_gb)(X, y, seg, ts) for ts in segs)
    else:
        res = [ _fold_other(X, y, seg, ts, model) for ts in segs ]  # GPU -> serial
    for r in res:
        if r is not None:
            idx, p = r; oof[idx] = p
    return oof

# ---------------- per-type AUROC ----------------
def per_type_auroc(oof, mask, atype):
    rows = []
    is_normal = (atype == "normal")
    types = sorted(set(atype[~is_normal & mask]))
    for t in types:
        sel = mask & (is_normal | (atype == t))
        yt = (atype[sel] == t).astype(int)
        if yt.sum() < 5 or (yt == 0).sum() < 5: continue
        try: a = roc_auc_score(yt, oof[sel])
        except Exception: a = np.nan
        rows.append((t, int(yt.sum()), round(float(a), 3)))
    return rows

# ---------------- incremental save ----------------
def append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as fh:
        if new: fh.write(",".join(header) + "\n")
        fh.write(",".join(str(x) for x in row) + "\n")

# ---------------- main ----------------
def run():
    # reset output files
    for p in (OUT_MAIN, OUT_PT):
        if os.path.exists(p): os.remove(p)
    main_rows = []
    for fn, label in STATIONS:
        t0 = time.time()
        df = pd.read_csv(os.path.join(DATA_DIR, fn))
        df = add_features(df)
        y = (df["anomaly"].fillna(0).astype(float) > 0).astype(int).values
        seg = df["segment_id"].values
        atype = df["anomaly_type"].fillna("normal").astype(str).values
        print(f"\n##### {label} | {df.shape[0]} rows | {y.sum()} anom ({y.mean()*100:.2f}%) | features ready [{time.time()-t0:.0f}s]")
        for feats, fname in [(BASE_FEATS, "baseline 7-feat"), (EXT_FEATS, "extended +spectral")]:
            X = df[feats].fillna(0).values
            models = ["GB"] + (["XGB"] if HAS_XGB else []) + (["LGB"] if HAS_LGB else [])
            for model in models:
                tm = time.time()
                oof = cv_oof(X, y, seg, model)
                m = ~np.isnan(oof)
                auc = roc_auc_score(y[m], oof[m]); ap = average_precision_score(y[m], oof[m])
                f, dr, pr, thr = best_event_f1(oof, y, m)
                row = [label, fname, model, round(f,1), round(dr,1), round(pr,1), round(auc,3), round(ap,3)]
                main_rows.append(row)
                append_csv(OUT_MAIN, ["Station","FeatureSet","Model","EvF1","EvDR","EvPrec","AUROC","AUPRC"], row)
                # per-type
                for (t, npos, a) in per_type_auroc(oof, m, atype):
                    append_csv(OUT_PT, ["Station","FeatureSet","Model","anomaly_type","n_pos","AUROC"],
                               [label, fname, model, t, npos, a])
                print(f"  {fname:20s} | {model:3s} | EvF1={f:5.1f} AUROC={auc:.3f} AUPRC={ap:.3f}  [{time.time()-tm:.0f}s, tot {time.time()-t0:.0f}s]")
    print("\n================ TOM TAT ================")
    out = pd.DataFrame(main_rows, columns=["Station","FeatureSet","Model","EvF1","EvDR","EvPrec","AUROC","AUPRC"])
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT_MAIN}\nSaved: {OUT_PT}")

if __name__ == "__main__":
    print(f"pywt={HAS_PYWT} xgb={HAS_XGB} lgb={HAS_LGB} | N_JOBS_GB={N_JOBS_GB}")
    print(f"EXT_FEATS ({len(EXT_FEATS)}): {EXT_FEATS}")
    run()
