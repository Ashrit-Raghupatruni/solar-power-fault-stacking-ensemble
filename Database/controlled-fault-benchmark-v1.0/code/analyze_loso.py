"""
analyze_loso.py — Leave-One-Station-Out (dap R1/R4 P2 #7: cross-station generalization).
Train tren 2 tram, test tren tram thu 3 (3 hoan vi). Detection binary, LightGBM extended.
So sanh voi within-station (da co) de thay tut bao nhieu khi chuyen mien tram.
Output: rev_loso.csv (TestStation, AUROC, AUPRC, EvF1)
"""
import os,sys,time,warnings,numpy as np,pandas as pd
sys.stdout.reconfigure(line_buffering=True); warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
from run_improve import add_features,best_event_f1,DATA_DIR,STATIONS
import lightgbm as lgb
HERE=os.path.dirname(os.path.abspath(__file__))
EXT16=["PE_m4","MSPE","CV","delta_PF","RPD","PE_m3","PE_m5","CV_16","CV_64","CV_256","dP","d2P","FFT_highband","FFT_flicker","freq_dev","wavelet_E"]

# nap truoc 3 tram (add_features 1 lan)
print("Loading + features..."); t0=time.time()
DATA={}
for fn,label in STATIONS:
    df=pd.read_csv(os.path.join(DATA_DIR,fn)); df=add_features(df)
    y=(df["anomaly"].fillna(0).astype(float)>0).astype(int).values
    DATA[label]=(df[EXT16].fillna(0).values, y)
    print(f"  {label} ready [{time.time()-t0:.0f}s]")

OUT=os.path.join(HERE,"rev_loso.csv")
if os.path.exists(OUT):os.remove(OUT)
labels=[l for _,l in STATIONS]
for test in labels:
    trains=[l for l in labels if l!=test]
    Xtr=np.vstack([DATA[l][0] for l in trains]); ytr=np.concatenate([DATA[l][1] for l in trains])
    Xte,yte=DATA[test]
    sc=StandardScaler().fit(Xtr)
    kw=dict(n_estimators=500,num_leaves=63,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,class_weight="balanced",random_state=42,verbose=-1)
    try:clf=lgb.LGBMClassifier(device_type="gpu",**kw).fit(sc.transform(Xtr),ytr)
    except Exception:clf=lgb.LGBMClassifier(**kw).fit(sc.transform(Xtr),ytr)
    p=clf.predict_proba(sc.transform(Xte))[:,1]
    m=np.ones(len(yte),bool)
    au=roc_auc_score(yte,p); ap=average_precision_score(yte,p); f,dr,pr,thr=best_event_f1(p,yte,m)
    with open(OUT,"a") as fh:
        if test==labels[0]: fh.write("TestStation,TrainStations,AUROC,AUPRC,EvF1\n")
        fh.write(f"{test},\"{'+'.join(trains)}\",{au:.3f},{ap:.3f},{f:.1f}\n")
    print(f"LOSO test={test:20s} (train {'+'.join(trains)}): AUROC={au:.3f} AUPRC={ap:.3f} EvF1={f:.1f} [{time.time()-t0:.0f}s]")
print("Saved rev_loso.csv")
