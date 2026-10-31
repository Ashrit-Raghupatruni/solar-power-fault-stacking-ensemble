"""
analyze_multiclass.py — NHAN DANG su co dien da lop (8 fault_category) tren DU LIEU THAT.
Bai toan: cho mot mau BAT THUONG, phan loai no thuoc nhom su co nao (load/voltage/frequency/
power_quality/sensor/outage/context/compound). => nang tu "phat hien nhi phan" -> "nhan dang".
Khong dung 'severity progression' (du lieu khong co tien trien -> tranh over-claim).

Thiet ke: chi train/test tren mau bat thuong (~10k/tram) -> nho -> nhanh.
7-fold segment(day) CV. baseline 7-feat vs extended +spectral. GB (faithful) + XGB + LGB (GPU).
Faithful: dung lai add_features cua run_improve.

Output: ket_qua_multiclass.csv          (Station,FeatureSet,Model,macroF1,weightedF1,accuracy,n)
        ket_qua_multiclass_perclass.csv (Station,FeatureSet,Model,category,f1,support)
CHAY SAU pipeline chinh (xep trong queue).
"""
import os, sys, time, warnings, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True); warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from run_improve import add_features, BASE_FEATS, EXT_FEATS, DATA_DIR, STATIONS
try: import xgboost as xgb; HAS_XGB=True
except Exception: HAS_XGB=False
try: import lightgbm as lgb; HAS_LGB=True
except Exception: HAS_LGB=False

HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,"ket_qua_multiclass.csv"); OUTP=os.path.join(HERE,"ket_qua_multiclass_perclass.csv")
OUTCM=os.path.join(HERE,"ket_qua_multiclass_confusion.csv")  # confusion matrix (extended featureset) cho figure Q1

def cv_predict(X,y,seg,nclass,model):
    oof=np.full(len(y),-1)
    for ts in np.unique(seg):
        te=seg==ts; tr=~te
        if tr.sum()<50 or te.sum()<5 or len(np.unique(y[tr]))<2: continue
        sc=StandardScaler().fit(X[tr]); Xtr,Xte=sc.transform(X[tr]),sc.transform(X[te])
        if model=="GB":
            clf=GradientBoostingClassifier(n_estimators=300,max_depth=5,learning_rate=0.08,
                                           subsample=0.8,random_state=42).fit(Xtr,y[tr])
        elif model=="XGB" and HAS_XGB:
            try:
                clf=xgb.XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.08,subsample=0.8,
                    colsample_bytree=0.8,objective="multi:softprob",num_class=nclass,
                    tree_method="hist",device="cuda",random_state=42).fit(Xtr,y[tr])
            except Exception:
                clf=xgb.XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.08,subsample=0.8,
                    colsample_bytree=0.8,objective="multi:softprob",num_class=nclass,
                    tree_method="hist",random_state=42).fit(Xtr,y[tr])
        elif model=="LGB" and HAS_LGB:
            try:
                clf=lgb.LGBMClassifier(n_estimators=500,num_leaves=63,learning_rate=0.05,subsample=0.8,
                    colsample_bytree=0.8,objective="multiclass",num_class=nclass,class_weight="balanced",
                    device_type="gpu",random_state=42,verbose=-1).fit(Xtr,y[tr])
            except Exception:
                clf=lgb.LGBMClassifier(n_estimators=500,num_leaves=63,learning_rate=0.05,subsample=0.8,
                    colsample_bytree=0.8,objective="multiclass",num_class=nclass,class_weight="balanced",
                    random_state=42,verbose=-1).fit(Xtr,y[tr])
        else:
            clf=GradientBoostingClassifier(random_state=42).fit(Xtr,y[tr])
        oof[te]=clf.predict(Xte)
    return oof

def append(path,header,row):
    new=not os.path.exists(path)
    with open(path,"a",encoding="utf-8") as f:
        if new: f.write(",".join(header)+"\n")
        f.write(",".join(str(x) for x in row)+"\n")

def run():
    for p in (OUT,OUTP,OUTCM):
        if os.path.exists(p): os.remove(p)
    for fn,label in STATIONS:
        t0=time.time()
        df=pd.read_csv(os.path.join(DATA_DIR,fn)); df=add_features(df)
        m=(df["anomaly_type"].astype(str)!="normal").values
        sub=df[m]
        le=LabelEncoder(); y=le.fit_transform(sub["fault_category"].astype(str).values)
        seg=sub["segment_id"].values; nclass=len(le.classes_)
        print(f"\n##### {label} | {m.sum()} anomaly samples | {nclass} fault categories: {list(le.classes_)} [{time.time()-t0:.0f}s]")
        for feats,fname in [(BASE_FEATS,"baseline 7-feat"),(EXT_FEATS,"extended +spectral")]:
            X=sub[feats].fillna(0).values
            # Bo sklearn-GB khoi multiclass (8x cay -> qua cham); giu XGB+LGB (GPU, native multiclass). LightGBM la model chinh.
            _models = (["XGB"] if HAS_XGB else [])+(["LGB"] if HAS_LGB else [])
            if not _models: _models = ["GB"]
            for model in _models:
                tm=time.time(); oof=cv_predict(X,y,seg,nclass,model); ok=oof>=0
                mf1=f1_score(y[ok],oof[ok],average="macro"); wf1=f1_score(y[ok],oof[ok],average="weighted")
                acc=accuracy_score(y[ok],oof[ok])
                append(OUT,["Station","FeatureSet","Model","macroF1","weightedF1","accuracy","n"],
                       [label,fname,model,round(mf1,3),round(wf1,3),round(acc,3),int(ok.sum())])
                perc=f1_score(y[ok],oof[ok],average=None,labels=range(nclass))
                sup=np.bincount(y[ok],minlength=nclass)
                for ci,cat in enumerate(le.classes_):
                    append(OUTP,["Station","FeatureSet","Model","category","f1","support"],
                           [label,fname,model,cat,round(float(perc[ci]),3),int(sup[ci])])
                # confusion matrix (chi luu cho extended featureset -> figure Q1)
                if fname=="extended +spectral":
                    cm=confusion_matrix(y[ok],oof[ok],labels=range(nclass))
                    for ti in range(nclass):
                        for pi in range(nclass):
                            append(OUTCM,["Station","Model","true","pred","count"],
                                   [label,model,le.classes_[ti],le.classes_[pi],int(cm[ti,pi])])
                print(f"  {fname:20s}|{model:3s}| macroF1={mf1:.3f} wF1={wf1:.3f} acc={acc:.3f} [{time.time()-tm:.0f}s]")
    print("\n=== MULTICLASS DONE ===")
    print(pd.read_csv(OUT).to_string(index=False))
    print(f"Saved {OUT}, {OUTP}")

if __name__=="__main__":
    print(f"xgb={HAS_XGB} lgb={HAS_LGB}")
    run()
