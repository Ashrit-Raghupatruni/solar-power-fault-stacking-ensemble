"""
analyze_pq_baseline.py — Baseline NGANH DIEN kinh dien (dap R2 MAJOR-5).
PQ disturbance classification co dien: wavelet multi-resolution analysis (MRA) -> nang luong tung band -> classifier.
Day la front-end PQ chuan (Khokhar et al. [25], wavelet/S-transform school).
Tinh tren cua so 64 mau: db4 wavelet 4 muc -> nang luong + std + entropy moi band -> feature vector.
So sanh detection (binary) voi proposed extended duoi CUNG protocol (7-fold day, LightGBM, event-F1).
Output: rev_pq_baseline.csv (Station, Method[wavelet-MRA vs proposed-extended], EvF1, AUROC, AUPRC)
"""
import os,sys,time,warnings,numpy as np,pandas as pd
sys.stdout.reconfigure(line_buffering=True); warnings.filterwarnings("ignore")
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
from run_improve import add_features,best_event_f1,DATA_DIR,STATIONS
import lightgbm as lgb
HERE=os.path.dirname(os.path.abspath(__file__))
EXT16=["PE_m4","MSPE","CV","delta_PF","RPD","PE_m3","PE_m5","CV_16","CV_64","CV_256","dP","d2P","FFT_highband","FFT_flicker","freq_dev","wavelet_E"]
WIN=64; LEVELS=4

def wavelet_mra_features(P):
    """Feature PQ kinh dien: db4 MRA 4 muc tren cua so 64 -> [energy, std, log-energy-entropy] moi band."""
    n=len(P); nb=LEVELS+1; feats=np.zeros((n,nb*3))
    for i in range(WIN,n):
        w=P[i-WIN:i]
        c=pywt.wavedec(w,"db4",level=LEVELS)  # [cA4, cD4, cD3, cD2, cD1]
        for bi,coef in enumerate(c):
            e=np.sum(coef**2); feats[i,bi*3]=e
            feats[i,bi*3+1]=np.std(coef)
            p=coef**2; p=p/(p.sum()+1e-12); feats[i,bi*3+2]=-np.sum(p*np.log2(p+1e-12))
    return feats

def lgb_oof(X,y,seg):
    oof=np.full(len(y),np.nan)
    for ts in np.unique(seg):
        te=seg==ts;tr=~te
        if y[tr].sum()<3 or te.sum()<10:continue
        sc=StandardScaler().fit(X[tr])
        kw=dict(n_estimators=500,num_leaves=63,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,class_weight="balanced",random_state=42,verbose=-1)
        try:clf=lgb.LGBMClassifier(device_type="gpu",**kw).fit(sc.transform(X[tr]),y[tr])
        except Exception:clf=lgb.LGBMClassifier(**kw).fit(sc.transform(X[tr]),y[tr])
        oof[te]=clf.predict_proba(sc.transform(X[te]))[:,1]
    return oof

def met(oof,y):
    m=~np.isnan(oof);f,dr,pr,thr=best_event_f1(oof,y,m)
    return round(f,1),round(roc_auc_score(y[m],oof[m]),3),round(average_precision_score(y[m],oof[m]),3)

OUT=os.path.join(HERE,"rev_pq_baseline.csv")
if os.path.exists(OUT):os.remove(OUT)
for fn,label in STATIONS:
    t0=time.time()
    df=pd.read_csv(os.path.join(DATA_DIR,fn)); df=add_features(df)
    y=(df["anomaly"].fillna(0).astype(float)>0).astype(int).values; seg=df["segment_id"].values
    # PQ wavelet-MRA baseline
    Xw=wavelet_mra_features(df["power_W"].values.astype(float))
    oofw=lgb_oof(Xw,y,seg); fw,auw,apw=met(oofw,y)
    # proposed extended (de doi chieu)
    oofe=lgb_oof(df[EXT16].fillna(0).values,y,seg); fe,aue,ape=met(oofe,y)
    for nm,(f,au,ap) in [("wavelet-MRA (PQ classic)",(fw,auw,apw)),("proposed-extended",(fe,aue,ape))]:
        new=not os.path.exists(OUT)
        with open(OUT,"a") as fh:
            if new: fh.write("Station,Method,EvF1,AUROC,AUPRC\n")
            fh.write(f"{label},{nm},{f},{au},{ap}\n")
        print(f"{label} {nm:26s}: EvF1={f} AUROC={au} AUPRC={ap} [{time.time()-t0:.0f}s]")
print("Saved rev_pq_baseline.csv")
