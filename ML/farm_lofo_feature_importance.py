"""
farm_lofo_feature_importance.py
===============================
Multivariate, Farm-LOFO-based feature-QUALITY analysis on the paper dataset
(master_dataset_old.csv: 20 farms, 81 fields, 1085 samples).

For each target a Random Forest is trained Farm-LOFO (leave-one-farm-out, 20 folds)
on the temporal-leakage-clean feature pool (spring spectral + topo + climate, 47 feats).
Feature quality = OUT-OF-FARM permutation importance: in every fold the held-out farm's
rows are used to measure how much each feature, when shuffled, degrades prediction
(increase in MSE) under a model that never saw that farm. This rewards only features
that transfer ACROSS farms, not within-farm spatial autocorrelation.

Reported per target (ML/results/farm_lofo_feature_importance_<target>.csv):
  imp_mean   : fold-size-weighted mean permutation importance (relative, sums to 1)
  imp_std    : between-fold dispersion of importance (lower = more stable)
  top5_freq  : fraction of the 20 folds where the feature ranks in that fold's top-5
               (a stability / robustness score, 0..1)
Plus an overall summary (ML/results/farm_lofo_feature_importance_summary.csv) with the
honest Farm-LOFO rho / R2 and the single best transferable feature per target.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
CSV  = ROOT / "data" / "features" / "master_dataset_old.csv"
RES  = ROOT / "ML" / "results"; RES.mkdir(parents=True, exist_ok=True)
TARGETS = ["ph", "soc", "no3", "p", "k", "s"]
SEED = 42
N_REPEATS = 15
RF = dict(n_estimators=300, max_depth=None, min_samples_leaf=2,
          max_features=0.5, random_state=SEED, n_jobs=-1)

EXCL = {"id","year","farm","field_name","grid_id","centroid_lon","centroid_lat",
        "geometry_wkt","protocol_number","analysis_date","sampling_date","hu",
        "cu","fe","mg","mn","mo","zn"}
SEAS = ("summer","late_summer","autumn")
def is_clean(c):
    cl=c.lower()
    if any(s in cl for s in SEAS): return False
    if "glcm" in cl: return False
    if cl.startswith(("ts_","range_","delta_","amp_","cv_","cs_")) or "spectral_" in cl: return False
    return True

df = pd.read_csv(CSV, low_memory=False)
num = [c for c in df.columns if df[c].dtype in ("float64","int64") and c not in EXCL and c not in TARGETS]
feats = [c for c in num if is_clean(c)]
print(f"{CSV.name}: {len(df)} samples, {df['farm'].nunique()} farms; clean pool={len(feats)} features")

summary=[]
for t in TARGETS:
    sub = df.dropna(subset=[t]).copy()
    X = sub[feats]; y = sub[t].values; farms = sub["farm"].values
    uniq = np.unique(farms)
    preds = np.full(len(sub), np.nan)
    imp_acc = np.zeros(len(feats)); w_acc = 0.0
    imp_sq  = np.zeros(len(feats))
    top5_count = np.zeros(len(feats))
    for fm in uniq:
        tr = farms!=fm; te = farms==fm
        Xtr=X[tr].copy(); Xte=X[te].copy()
        med=Xtr.median(); Xtr=Xtr.fillna(med); Xte=Xte.fillna(med)
        m=RandomForestRegressor(**RF); m.fit(Xtr, y[tr])
        preds[te]=m.predict(Xte)
        if te.sum()>=3:
            pi=permutation_importance(m, Xte, y[te], n_repeats=N_REPEATS,
                                      random_state=SEED, scoring="neg_mean_squared_error", n_jobs=-1)
            imp=np.clip(pi.importances_mean, 0, None)
            if imp.sum()>0: impn=imp/imp.sum()
            else: impn=imp
            w=te.sum()
            imp_acc += impn*w; imp_sq += (impn**2)*w; w_acc += w
            top5=np.argsort(impn)[::-1][:5]
            top5_count[top5]+=1
    mean_imp = imp_acc/w_acc
    var_imp  = np.clip(imp_sq/w_acc - mean_imp**2, 0, None)
    std_imp  = np.sqrt(var_imp)
    top5_freq= top5_count/len(uniq)
    v=np.isfinite(preds)
    rho,_=spearmanr(y[v],preds[v]); r2=r2_score(y[v],preds[v])

    fi=pd.DataFrame({"feature":feats,"imp_mean":mean_imp.round(4),
                     "imp_std":std_imp.round(4),"top5_freq":top5_freq.round(2)}
                    ).sort_values("imp_mean",ascending=False).reset_index(drop=True)
    fi.to_csv(RES/f"farm_lofo_feature_importance_{t}.csv",index=False)
    best=fi.iloc[0]
    summary.append(dict(target=t, farm_lofo_rho=round(rho,3), farm_lofo_r2=round(r2,3),
                        n=int(v.sum()), n_farms=len(uniq),
                        best_feature=best["feature"], best_imp=best["imp_mean"],
                        best_top5_freq=best["top5_freq"]))
    print(f"\n### {t.upper()}  Farm-LOFO rho={rho:.3f} R2={r2:.3f}  (n={v.sum()}, {len(uniq)} farms)")
    print(fi.head(8).to_string(index=False))

pd.DataFrame(summary).to_csv(RES/"farm_lofo_feature_importance_summary.csv",index=False)
print("\n==== SUMMARY ====")
print(pd.DataFrame(summary).to_string(index=False))
