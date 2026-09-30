"""Phase 2b: LightGBM v1, GroupKFold by patient, monotone carbs+/steps-.

Features: glucose, slope_15/30, roll mean/std 30/60, carbs 60/120,
time-since-meal, steps 30/60, hr mean 30, tod sin/cos, dow,
+ static age/BMI/A1c/fasting (bio.csv; no meds in CGMacros).
Targets: regress 30/60/120 + event_60 classifier.
Saves: results/lgbm_metrics.json, models, SHAP png.
"""
import json
import os

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")

IN = "data/processed/unified.parquet"
BIO = "data/raw/cgmacros/unzipped/CGMacros/bio.csv"
H = {30: 6, 60: 12, 120: 24}

FEATS = [
    "glucose", "slope_15", "slope_30",
    "roll_mean_30", "roll_std_30", "roll_mean_60", "roll_std_60",
    "carbs_60", "carbs_120", "time_since_meal",
    "steps_30", "steps_60", "hr_30", "activity_60",
    "tod_sin", "tod_cos", "dow",
    "age", "bmi", "a1c", "fasting_glu",
]
# monotone: +1 carbs features, -1 steps features
MONO = [1 if f.startswith("carbs_") else (-1 if f.startswith("steps_") else 0)
        for f in FEATS]


def add_features(u: pd.DataFrame) -> pd.DataFrame:
    u = u.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    g = u.groupby("patient_id", group_keys=False)
    u["slope_15"] = g["glucose"].diff(3) / 15.0
    u["slope_30"] = g["glucose"].diff(6) / 30.0
    u["roll_mean_30"] = g["glucose"].transform(
        lambda s: s.rolling(6, min_periods=1).mean())
    u["roll_std_30"] = g["glucose"].transform(
        lambda s: s.rolling(6, min_periods=1).std()).fillna(0.0)
    u["roll_mean_60"] = g["glucose"].transform(
        lambda s: s.rolling(12, min_periods=1).mean())
    u["roll_std_60"] = g["glucose"].transform(
        lambda s: s.rolling(12, min_periods=1).std()).fillna(0.0)
    u["carbs_60"] = g["meal_carbs"].transform(
        lambda s: s.rolling(12, min_periods=1).sum())
    u["carbs_120"] = g["meal_carbs"].transform(
        lambda s: s.rolling(24, min_periods=1).sum())
    # time since meal (min), cap 1440
    def tsm(s):
        idx = s.index[s > 0.5]
        out = pd.Series(1440.0, index=s.index)
        if len(idx):
            last = None
            vals = {}
            for i in s.index:
                if i in set(idx):
                    last = i
                vals[i] = 0.0 if last is None else (i - last) * 5.0
            out = pd.Series(vals).clip(0, 1440)
        return out
    u["time_since_meal"] = g["meal_carbs"].transform(tsm)
    u["steps_30"] = g["steps"].transform(lambda s: s.rolling(6, min_periods=1).sum())
    u["steps_60"] = g["steps"].transform(lambda s: s.rolling(12, min_periods=1).sum())
    u["hr_30"] = g["hr"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    u["activity_60"] = g["activity_kcal"].transform(
        lambda s: s.rolling(12, min_periods=1).sum())
    hod = u["timestamp"].dt.hour + u["timestamp"].dt.minute / 60.0
    u["tod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    u["tod_cos"] = np.cos(2 * np.pi * hod / 24.0)
    u["dow"] = u["timestamp"].dt.dayofweek
    for h, steps in H.items():
        u[f"target_glucose_{h}"] = g["glucose"].shift(-steps)
    fut = [g["glucose"].shift(-i) for i in range(1, 13)]
    u["event_60"] = (pd.concat(fut, axis=1).max(axis=1) > 180).astype(float)
    return u


def add_static(u: pd.DataFrame) -> pd.DataFrame:
    b = pd.read_csv(BIO)
    b.columns = [c.strip() for c in b.columns]
    b = b.rename(columns={"subject": "patient_id", "Age": "age",
                          "BMI": "bmi", "A1c PDL (Lab)": "a1c",
                          "Fasting GLU - PDL (Lab)": "fasting_glu"})
    b["patient_id"] = b["patient_id"].astype(float)
    u["patient_id"] = u["patient_id"].astype(float)
    u = u.merge(b[["patient_id", "age", "bmi", "a1c", "fasting_glu"]],
                on="patient_id", how="left")
    return u


def main():
    u = pd.read_parquet(IN)
    u["timestamp"] = pd.to_datetime(u["timestamp"])
    u = add_features(u)
    u = add_static(u)
    # keep rows with full 120-min future for comparable horizons + event label
    m = u["target_glucose_120"].notna() & u["event_60"].notna()
    d = u.loc[m].reset_index(drop=True)
    X = d[FEATS].to_numpy()
    groups = d["patient_id"].to_numpy()
    print(f"rows={len(d)} patients={d['patient_id'].nunique()} "
          f"event={d['event_60'].mean() * 100:.2f}%")

    gkf = GroupKFold(n_splits=5)
    oof_reg = {h: np.full(len(d), np.nan) for h in H}
    oof_clf = np.full(len(d), np.nan)
    for tr, te in gkf.split(X, groups=groups):
        for h in H:
            r = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, min_child_samples=50,
                                  monotone_constraints=MONO, verbose=-1)
            r.fit(X[tr], d[f"target_glucose_{h}"].to_numpy()[tr])
            oof_reg[h][te] = r.predict(X[te])
        c = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               num_leaves=31, min_child_samples=100,
                               monotone_constraints=MONO, verbose=-1)
        c.fit(X[tr], d["event_60"].to_numpy()[tr])
        oof_clf[te] = c.predict_proba(X[te])[:, 1]

    with open("results/metrics.json") as f:
        base = json.load(f)
    res = {"n_rows": int(len(d)), "features": FEATS,
           "monotone_constraints": MONO}
    for h in H:
        yt = d[f"target_glucose_{h}"].to_numpy()
        yp = oof_reg[h]
        res[f"rmse_{h}"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        res[f"mae_{h}"] = float(np.mean(np.abs(yt - yp)))
        res[f"rmse_{h}_persistence"] = base["persistence"][f"rmse_{h}"]
    res["event_auroc"] = float(roc_auc_score(d["event_60"], oof_clf))
    res["event_auprc"] = float(average_precision_score(d["event_60"], oof_clf))
    # false alerts per patient-day: alert=prob>=0.5, false=alert & no event
    alert = (oof_clf >= 0.5).astype(int)
    false = ((alert == 1) & (d["event_60"].to_numpy() == 0)).sum()
    patient_days = len(d) / 288.0
    res["false_alerts_per_patient_day"] = float(false / patient_days)
    print(json.dumps(res, indent=2))

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open("results/lgbm_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    # refit on all data for demo/freeze; save 60-min reg + event clf
    r60 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                            num_leaves=31, min_child_samples=50,
                            monotone_constraints=MONO, verbose=-1)
    r60.fit(X, d["target_glucose_60"].to_numpy())
    r60.booster_.save_model("models/lgbm_reg60.txt")
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             num_leaves=31, min_child_samples=100,
                             monotone_constraints=MONO, verbose=-1)
    clf.fit(X, d["event_60"].to_numpy())
    clf.booster_.save_model("models/lgbm_event.txt")
    # SHAP summary (reg60, sample 2000)
    import shap
    sample = X[np.random.RandomState(0).choice(len(X), min(2000, len(X)),
                                               replace=False)]
    ex = shap.TreeExplainer(r60)
    sv = ex.shap_values(sample)
    import matplotlib.pyplot as plt
    plt.figure()
    shap.summary_plot(sv, sample, feature_names=FEATS, show=False)
    plt.tight_layout()
    plt.savefig("results/shap_summary.png", dpi=120)
    plt.close()
    print("saved models/lgbm_reg60.txt models/lgbm_event.txt "
          "results/shap_summary.png")


if __name__ == "__main__":
    main()
