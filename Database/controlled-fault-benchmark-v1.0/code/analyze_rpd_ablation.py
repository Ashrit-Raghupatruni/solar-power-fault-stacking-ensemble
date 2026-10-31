"""
analyze_rpd_ablation.py — KIEM CHUNG phat hien "RPD gan nhu vo dung tren du lieu THAT".
MI(kNN) cho RPD~0.0001 co the la THAT, hoac ARTIFACT do RPD duoi nang (outlier khi outage median~0).
Hai kiem tra doc lap:
  (1) ABLATION: detection LightGBM CO vs KHONG RPD (baseline & extended) -> AUROC/EvF1 co doi khong?
      Neu khong doi -> RPD that su khong dong gop.
  (2) ASSOCIATION ben vung: point-biserial corr + MI tren RANK(RPD) (chong artifact duoi nang) + clip.
Chay GPU (LightGBM). Output: ket_qua_rpd_ablation.csv + in association stats.
"""
import os, sys, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True); warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pointbiserialr, rankdata
from run_improve import add_features, best_event_f1, DATA_DIR, STATIONS
import lightgbm as lgb

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"ket_qua_rpd_ablation.csv")
BASE7=["PE_m4","MSPE","CV","delta_PF","RPD","PE_m3","PE_m5"]
BASE6=[f for f in BASE7 if f!="RPD"]
EXT16=BASE7+["CV_16","CV_64","CV_256","dP","d2P","FFT_highband","FFT_flicker","freq_dev","wavelet_E"]
EXT15=[f for f in EXT16 if f!="RPD"]
SETS=[("baseline+RPD",BASE7),("baseline-RPD",BASE6),("extended+RPD",EXT16),("extended-RPD",EXT15)]

def lgb_oof(X,y,seg):
    oof=np.full(len(y),np.nan)
    for ts in np.unique(seg):
        te=seg==ts; tr=~te
        if y[tr].sum()<3 or te.sum()<10: continue
        sc=StandardScaler().fit(X[tr])
        kw=dict(n_estimators=500,num_leaves=63,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                class_weight="balanced",random_state=42,verbose=-1)
        try: clf=lgb.LGBMClassifier(device_type="gpu",**kw).fit(sc.transform(X[tr]),y[tr])
        except Exception: clf=lgb.LGBMClassifier(**kw).fit(sc.transform(X[tr]),y[tr])
        oof[te]=clf.predict_proba(sc.transform(X[te]))[:,1]
    return oof

def run():
    if os.path.exists(OUT): os.remove(OUT)
    rows=[]
    for fn,label in STATIONS:
        df=pd.read_csv(os.path.join(DATA_DIR,fn)); df=add_features(df)
        y=(df["anomaly"].fillna(0).astype(float)>0).astype(int).values
        seg=df["segment_id"].values; rpd=df["RPD"].fillna(0).values.astype(float)
        # (2) association ben vung
        rc=np.clip(rpd,np.percentile(rpd,0.5),np.percentile(rpd,99.5))
        try: pb=pointbiserialr(y,rc)[0]
        except Exception: pb=float("nan")
        mi_raw=mutual_info_classif(rpd.reshape(-1,1),y,random_state=42)[0]
        mi_rank=mutual_info_classif(rankdata(rpd).reshape(-1,1),y,random_state=42)[0]
        print(f"\n##### {label}: RPD assoc -> point-biserial(clip)={pb:.4f} | MI_raw={mi_raw:.4f} | MI_rank={mi_rank:.4f}")
        print(f"   RPD stats: mean={rpd.mean():.3f} std={rpd.std():.3f} max={rpd.max():.1f} %>0={np.mean(rpd>1e-6)*100:.1f}")
        # (1) ablation
        for name,feats in SETS:
            oof=lgb_oof(df[feats].fillna(0).values,y,seg); m=~np.isnan(oof)
            auc=roc_auc_score(y[m],oof[m]); ap=average_precision_score(y[m],oof[m])
            f,dr,pr,thr=best_event_f1(oof,y,m)
            rows.append([label,name,round(f,1),round(auc,3),round(ap,3)])
            print(f"   {name:14s}: EvF1={f:5.1f} AUROC={auc:.3f} AUPRC={ap:.3f}")
    out=pd.DataFrame(rows,columns=["Station","FeatureSet","EvF1","AUROC","AUPRC"])
    out.to_csv(OUT,index=False)
    print("\n=== KET LUAN: so sanh +RPD vs -RPD (neu ~bang nhau -> RPD khong dong gop) ===")
    print(out.to_string(index=False))
    print(f"Saved {OUT}")

if __name__=="__main__":
    run()
