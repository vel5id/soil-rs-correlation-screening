import pandas as pd, numpy as np
tag = pd.read_csv("ML/results/feature_quality_tagged.csv")
leak = pd.read_csv("math_statistics/output/leakage_controlled_screening.csv").set_index("target")
LAB = {"ph":"pH","soc":"SOC","no3":"NO3","p":"P2O5","k":"K2O","s":"S"}
ORDER = ["ph","k","p","no3","soc","s"]

def best_robust(lab):
    r = tag[(tag.target==lab) & (tag.robust==True)].copy()
    if not len(r): return "-"
    r["a"]=r.block_within.abs()
    b=r.sort_values("a",ascending=False).iloc[0]
    return f"{b.feature} ({b.rho_full:+.2f})"

rows=[]
for t in ORDER:
    lab=LAB[t]
    sub=tag[tag.target==lab]
    g=int((sub.feature_class=="generalizable").sum())
    rob=int(sub.robust.sum())
    z=int((sub.feature_class=="zonal_only").sum())
    u=int((sub.feature_class=="unstable").sum())
    w=int((sub.feature_class=="weak").sum())
    rho=leak.loc[t,"rho_max_aligned"]; flofo=leak.loc[t,"farm_lofo_rho"]
    # verdict
    if rob>=5 and flofo>=0.30: v="mappable (regional + local)"
    elif rob>=3: v="weakly mappable"
    elif rob>0 or g>0: v="indirect / weak out-of-farm" if flofo>=0.15 else "regional gradient only"
    else: v="unpredictable"
    rows.append({"Property":lab,"|rho|max(LC)":round(rho,3),"Farm-LOFO rho":round(flofo,3),
                 "general.(robust)":f"{g} ({rob})","zonal_only":z,"unstable":u,"weak":w,
                 "best transferable feature":best_robust(lab),"verdict":v})
df=pd.DataFrame(rows)
cols=["Property","|rho|max(LC)","Farm-LOFO rho","general.(robust)","zonal_only","unstable","weak","best transferable feature","verdict"]
print(df[cols].to_string(index=False))
df[cols].to_csv("ML/results/key_table_taxonomy.csv",index=False)
print("\noverall:", dict(tag.feature_class.value_counts()), "| robust total:", int(tag.robust.sum()))
print("saved ML/results/key_table_taxonomy.csv")
