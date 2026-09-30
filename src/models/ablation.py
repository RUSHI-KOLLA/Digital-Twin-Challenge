"""Phase 2c: ablation A->D, identical settings, GroupKFold by patient.

A: CGM only (glucose/slope/rolling + tod/dow calendar)
B: + meals (carbs_60/120, time_since_meal; mono +1 carbs)
C: + wearable (steps_30/60, hr_30, activity_60; mono -1 steps)
D: + static EHR (age/bmi/a1c/fasting_glu)
Metrics: RMSE@60, AUROC, AUPRC (OOF). Saves results/ablation.csv + .png
"""
import os

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")

from lightgbm_v1 import add_features, add_static  # noqa: E402

VARIANTS = {
    "A_cgm": ["glucose", "slope_15", "slope_30", "roll_mean_30",
              "roll_std_30", "roll_mean_60", "roll_std_60",
              "tod_sin", "tod_cos", "dow"],
    "B_meals": None,  # A + meals
    "C_wearable": None,  # B + wearable
    "D_ehr": None,  # C + static
}
VARIANTS["B_meals"] = VARIANTS["A_cgm"] + ["carbs_60", "carbs_120",
                                          "time_since_meal"]
VARIANTS["C_wearable"] = VARIANTS["B_meals"] + ["steps_30", "steps_60",
                                               "hr_30", "activity_60"]
VARIANTS["D_ehr"] = VARIANTS["C_wearable"] + ["age", "bmi", "a1c",
                                             "fasting_glu"]


def mono_for(feats):
    return [1 if f.startswith("carbs_") else
            (-1 if f.startswith("steps_") else 0) for f in feats]


def run_variant(d, feats, groups):
    X = d[feats].to_numpy()
    yt = d["target_glucose_60"].to_numpy()
    ye = d["event_60"].to_numpy()
    mono = mono_for(feats)
    gkf = GroupKFold(n_splits=5)
    pr = np.full(len(d), np.nan)
    pc = np.full(len(d), np.nan)
    for tr, te in gkf.split(X, groups=groups):
        r = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, min_child_samples=50,
                              monotone_constraints=mono, verbose=-1)
        r.fit(X[tr], yt[tr])
        pr[te] = r.predict(X[te])
        c = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=31, min_child_samples=100,
                               monotone_constraints=mono, verbose=-1)
        c.fit(X[tr], ye[tr])
        pc[te] = c.predict_proba(X[te])[:, 1]
    return (float(np.sqrt(np.mean((yt - pr) ** 2))),
            float(roc_auc_score(ye, pc)),
            float(average_precision_score(ye, pc)))


def main():
    u = pd.read_parquet("data/processed/unified.parquet")
    u["timestamp"] = pd.to_datetime(u["timestamp"])
    u = add_features(u)
    u = add_static(u)
    m = u["target_glucose_120"].notna() & u["event_60"].notna()
    d = u.loc[m].reset_index(drop=True)
    groups = d["patient_id"].to_numpy()
    rows = []
    for name, feats in VARIANTS.items():
        rmse, auroc, auprc = run_variant(d, feats, groups)
        rows.append({"variant": name, "n_features": len(feats),
                     "rmse_60": rmse, "auroc": auroc, "auprc": auprc})
        print(rows[-1])
    ab = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    ab.to_csv("results/ablation.csv", index=False)
    # table image
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    ax.table(cellText=np.round(ab[["rmse_60", "auroc", "auprc"]].to_numpy(), 3),
             rowLabels=ab["variant"].tolist(),
             colLabels=["RMSE@60", "AUROC", "AUPRC"],
             loc="center")
    plt.title("Ablation (GroupKFold OOF, n=45 patients)")
    plt.tight_layout()
    plt.savefig("results/ablation.png", dpi=150)
    plt.close()
    print("saved results/ablation.csv + results/ablation.png")


if __name__ == "__main__":
    main()
