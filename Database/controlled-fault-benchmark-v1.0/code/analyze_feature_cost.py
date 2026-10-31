"""
analyze_feature_cost.py — Do CHI PHI TINH FEATURE (dap review R3 MAJOR-7/MINOR-8).
Do thoi gian tinh tung nhom feature tren 1 cua so 64 mau (host), roi uoc luong ESP32.
Khong train gi — chi do thoi gian extraction. Nhanh.
Output: rev_feature_cost.csv
"""
import os,sys,time,numpy as np,pandas as pd
sys.stdout.reconfigure(line_buffering=True)
import pywt
HERE=os.path.dirname(os.path.abspath(__file__))

# Lay 1 doan power that de do
df=pd.read_csv(os.path.join(HERE,"..","02_processed_features","station1_normal_features.csv"),
               usecols=["power_W","frequency_Hz"], nrows=5000)
P=df["power_W"].values.astype(float); F=df["frequency_Hz"].values.astype(float)
WIN=64; N=2000  # do tren 2000 cua so

def t_it(fn,reps=N):
    t0=time.perf_counter()
    for i in range(reps): fn(i)
    return (time.perf_counter()-t0)/reps*1e6  # microseconds/window

# tung thanh phan
def f_fft(i):
    w=P[i:i+WIN]-P[i:i+WIN].mean(); sp=np.abs(np.fft.rfft(w*np.hanning(WIN)))**2
    return sp[len(sp)//2:].sum()/(sp.sum()+1e-9)
def f_wavelet(i):
    c=pywt.wavedec(P[i:i+WIN],"db4",level=3); return np.sum(c[1]**2)/(np.sum(P[i:i+WIN]**2)+1e-9)
def f_cv(i):
    w=P[i:i+WIN]; return w.std()/(abs(w.mean())+1e-9)
def f_freqdev(i):
    return abs(F[i+WIN-1]-50.0)
def f_pe(i):
    # permutation entropy m=4 (xap xi chi phi)
    w=P[i:i+WIN]; m=4
    from itertools import permutations
    patt={}
    for j in range(len(w)-m+1):
        key=tuple(np.argsort(w[j:j+m])); patt[key]=patt.get(key,0)+1
    c=np.array(list(patt.values()),float); c/=c.sum()
    return -np.sum(c*np.log2(c))

rows=[]
HOST_GHZ=2.7; ESP_MHZ=240  # i7-4800MQ vs ESP32
factor=HOST_GHZ*1000/ESP_MHZ  # ~11.25x cham hon (uoc luong tho, chua tinh kien truc)
for nm,fn in [("FFT_highband",f_fft),("wavelet_db4",f_wavelet),("CV",f_cv),
              ("freq_dev",f_freqdev),("PE_m4",f_pe)]:
    us=t_it(fn)
    rows.append([nm,round(us,2),round(us*factor,1)])
    print(f"{nm:14s}: {us:7.2f} us/window (host) -> ~{us*factor:8.1f} us/window (ESP32 est)")

out=pd.DataFrame(rows,columns=["feature","host_us_per_window","esp32_est_us_per_window"])
tot_host=out["host_us_per_window"].sum(); tot_esp=out["esp32_est_us_per_window"].sum()
print(f"\nTONG full feature set: {tot_host:.1f} us/sample (host) -> ~{tot_esp:.1f} us (~{tot_esp/1000:.2f} ms) /sample (ESP32 est)")
print(f"Budget @1Hz = 1,000,000 us/sample -> dung {tot_esp/1e6*100:.3f}% budget. Real-time @1Hz: {'OK' if tot_esp<1e6 else 'NO'}")
out.to_csv(os.path.join(HERE,"rev_feature_cost.csv"),index=False)
# them dong tong
with open(os.path.join(HERE,"rev_feature_cost.csv"),"a") as fh:
    fh.write(f"TOTAL,{tot_host:.1f},{tot_esp:.1f}\n")
print("Saved rev_feature_cost.csv")
