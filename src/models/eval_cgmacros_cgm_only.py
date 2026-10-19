"""Evaluate frozen v2 on CGMacros with meal and HR/activity features zeroed.

This simulates the ShanghaiT2DM condition (CGM-only) on the training cohort
to separate missing-data effect from population shift.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

IN = "data/processed/unified.parquet"
BIO = "data/raw/cgmacros/unzipped/CGMacros/bio.csv"
MODEL_REG = "models/lgbm_reg60.txt"
MODEL_CLF = "models/lgbm_event.txt"
OUT = "results/cgmacros_cgm_only.json"

# Frozen-v2 feature set (15)
V2_FEATS = ["glucose", "slope_15", "slope_30",
            "roll_mean_30", "roll_std_30", "roll_mean_60", "roll_std_60",
            "carbs_60", "carbs_120", "time_since_meal",
            "hr_30", "activity_60", "tod_sin", "tod_cos", "dow"]

H = {30: 6, 60: 12, 120: 24}


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
    u["time_since_meal"] = 1440.0
    for _, grp in u.groupby("patient_id"):
        ts = grp["timestamp"].to_numpy(dtype="datetime64[m]").astype("int64")
        mm = ts[grp["meal_carbs"].to_numpy() > 0.5]
        if len(mm):
            j = np.searchsorted(mm, ts, side="right") - 1
            ok = j >= 0
            vals = np.full(len(grp), 1440.0)
            vals[ok] = (ts[ok] - mm[j[ok]]).astype(float)
            u.loc[grp.index, "time_since_meal"] = np.clip(vals, 0, 1440)
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


def boot(pids, df, fn, n_boot=500):
    rng = np.random.RandomState(0)
    groups = {p: g for p, g in df.groupby("patient_id")}
    vals = [fn(pd.concat([groups[p] for p in
                          rng.choice(pids, size=len(pids), replace=True)]))
            for _ in range(n_boot)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main():
    u = pd.read_parquet(IN)
    u["timestamp"] = pd.to_datetime(u["timestamp"])
    u = add_features(u)
    u = add_static(u)
    m = u["target_glucose_120"].notna() & u["event_60"].notna()
    d = u.loc[m].reset_index(drop=True)
    d["t2d"] = d["a1c"] >= 6.5
    
    # ZERO OUT meal and HR/activity features (simulate CGM-only)
    zero_feats = ["carbs_60", "carbs_120", "hr_30", "activity_60"]
    for f in zero_feats:
        d[f] = 0.0
    
    X = d[V2_FEATS].to_numpy(dtype=float)
    reg = lgb.Booster(model_file=MODEL_REG)
    clf = lgb.Booster(model_file=MODEL_CLF)
    d["p60"] = reg.predict(X)
    d["escore"] = clf.predict(X)
    
    pids = np.array(sorted(d["patient_id"].unique()))
    rmse = lambda s: float(np.sqrt(np.mean((s["target_glucose_60"] - s["p60"]) ** 2)))
    
    res = {
        "n_patients": int(len(pids)),
        "n_rows": int(len(d)),
        "event_60_rate": float(d["event_60"].mean()),
        "rmse_60": rmse(d),
        "rmse_60_ci": boot(pids, d, rmse),
        "event_auroc": float(roc_auc_score(d["event_60"], d["escore"])),
        "event_auprc": float(average_precision_score(d["event_60"], d["escore"])),
        "method": "frozen v2 on CGMacros with meal/HR/activity features ZEROED "
                  "(simulates CGM-only cohort); patient-wise bootstrap CIs",
    }
    res["event_auroc_ci"] = boot(pids, d, lambda s: float(
        roc_auc_score(s["event_60"], s["escore"])))
    
    # Also compute persistence baseline on CGMacros for reference
    d["pers_60"] = d["glucose"]
    res["persistence_rmse_60"] = float(np.sqrt(np.mean((d["target_glucose_60"] - d["pers_60"]) ** 2)))
    res["persistence_rmse_60_ci"] = boot(pids, d, lambda s: float(
        np.sqrt(np.mean((s["target_glucose_60"] - s["pers_60"]) ** 2))))
    
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in res.items()}, indent=1))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
