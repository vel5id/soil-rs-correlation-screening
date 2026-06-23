"""
farm_lofo_honest_audit.py
=========================
Honest predictability audit on the PAPER dataset (master_dataset_old.csv:
20 farms, 81 fields, 1085 samples, 2022-2023) — the same file both manuscripts
screen. Quantifies how each soil property's apparent |rho| collapses as leakage
is removed, ending at the honest Farm-LOFO (leave-one-farm-out) metric.

Columns produced per target:
  screen_raw      : whole-dataset univariate |rho|max over the FULL 512 pool (draft's number)
  screen_clean    : whole-dataset univariate |rho|max after dropping summer/late_summer/
                    autumn + GLCM texture + ts_/composite (temporal-leakage-clean)
  field_lofo_rho  : RF, Field-LOFO (81 folds), leakage-clean features, out-of-fold Spearman
  farm_lofo_rho   : RF, Farm-LOFO (20 folds), leakage-clean features, out-of-farm Spearman  <-- HONEST
  farm_lofo_r2    : RF, Farm-LOFO R^2
  farm_lofo_rho_fullfeat : RF, Farm-LOFO with the FULL feature set (incl. leaky seasonal)
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
CSV  = ROOT / "data" / "features" / "master_dataset_old.csv"
TARGETS = ["ph", "soc", "no3", "p", "k", "s"]
SEED = 42
RF = dict(n_estimators=400, max_depth=None, min_samples_leaf=2,
          max_features=0.5, random_state=SEED, n_jobs=-1)

EXCL = {"id","year","farm","field_name","grid_id","centroid_lon","centroid_lat",
        "geometry_wkt","protocol_number","analysis_date","sampling_date","hu",
        "cu","fe","mg","mn","mo","zn"}  # also drop leaked soil micronutrients

df = pd.read_csv(CSV, low_memory=False)
print(f"{CSV.name}: {len(df)} samples, {df['farm'].nunique()} farms, {df['field_name'].nunique()} fields")

num = [c for c in df.columns if df[c].dtype in ("float64","int64")
       and c not in EXCL and c not in TARGETS]

SEAS = ("summer","late_summer","autumn")
VEG  = ("ndvi","savi","gndvi","evi","ndre","msavi","rendvi","psri")
def is_clean(c):
    cl=c.lower()
    if any(s in cl for s in SEAS): return False
    if "glcm" in cl: return False
    if cl.startswith(("ts_","range_","delta_","amp_","cv_","cs_")) or "spectral_" in cl: return False
    return True

full_feats  = num
clean_feats = [c for c in num if is_clean(c)]
print(f"full pool={len(full_feats)}  leakage-clean pool={len(clean_feats)}")

def screen_uni(target, pool):
    best=(0.0,None)
    for c in pool:
        m=df[[target,c]].notna().all(axis=1)
        if m.sum()<10: continue
        r,_=spearmanr(df.loc[m,target], df.loc[m,c])
        if pd.notna(r) and abs(r)>best[0]: best=(abs(r),c)
    return best

def lofo_rho(target, group_col, feats):
    sub=df.dropna(subset=[target]).copy()
    X=sub[feats]; y=sub[target].values; g=sub[group_col].values
    preds=np.full(len(sub), np.nan)
    for grp in np.unique(g):
        tr=g!=grp; te=g==grp
        Xtr=X[tr].copy(); Xte=X[te].copy()
        med=Xtr.median(); Xtr=Xtr.fillna(med); Xte=Xte.fillna(med)
        m=RandomForestRegressor(**RF); m.fit(Xtr, y[tr])
        preds[te]=m.predict(Xte)
    v=np.isfinite(preds)
    rho,_=spearmanr(y[v], preds[v]); r2=r2_score(y[v], preds[v])
    return rho, r2

rows=[]
for t in TARGETS:
    sr,sf  = screen_uni(t, full_feats)
    cr,cf  = screen_uni(t, clean_feats)
    fl,_   = lofo_rho(t, "field_name", clean_feats)
    fa,fa2 = lofo_rho(t, "farm",       clean_feats)
    faF,_  = lofo_rho(t, "farm",       full_feats)
    rows.append(dict(target=t,
        screen_raw=round(sr,3), screen_raw_feat=sf,
        screen_clean=round(cr,3), screen_clean_feat=cf,
        field_lofo_rho=round(fl,3),
        farm_lofo_rho=round(fa,3), farm_lofo_r2=round(fa2,3),
        farm_lofo_rho_fullfeat=round(faF,3)))
    print(f"[{t}] raw={sr:.3f}({sf})  clean={cr:.3f}  fieldLOFO={fl:.3f}  "
          f"FARM-LOFO={fa:.3f} (R2={fa2:.3f})  farmLOFO-fullfeat={faF:.3f}")

out=pd.DataFrame(rows)
(ROOT/"ML"/"results").mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT/"ML"/"results"/"farm_lofo_honest_audit.csv", index=False)
print("\n", out[["target","screen_raw","screen_clean","field_lofo_rho",
                 "farm_lofo_rho","farm_lofo_r2","farm_lofo_rho_fullfeat"]].to_string(index=False))
