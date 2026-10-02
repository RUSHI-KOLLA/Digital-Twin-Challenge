"""Phase 2b (refreeze v2): LightGBM, GroupKFold by patient, monotone carbs+1.

REFREEZE NOTE: ships B+wearable-lite (CGM + meals + HR/activity, 15 feats).
Ablation B (+meals) scored RMSE 25.46; D (full EHR) 25.74. Dropped from D:
steps_* (0 for 44/45 patients — dead feature, zero gain) and static EHR
(no measurable gain at this n, CIs overlap). HR/activity kept so the walk
what-if moves through learned dense signals. The walk vs carb-swap
what-if relative size varies by state; the UI shows both and ranks neither.
Targets: regress 30/60/120 + event_60 classifier.
Saves: results/lgbm_metrics.json, results/oof_frozen.parquet, models, SHAP png.

Label note: 0.5% of adjacent 5-min pairs span gaps (sensor dropouts); labels
and rolling features use positional bins (comparable across all tables).
time_since_meal uses true timestamps (gap-safe).
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
    "hr_30", "activity_60",
    "tod_sin", "tod_cos", "dow",
    # v3 Tier-1 additions (all past-only; MONO 0 except carbs_+1)
    "fat_60", "fat_120", "protein_60", "protein_120",
    "glucose_accel", "glucose_z", "hr_vs_baseline", "pm_rise_baseline",
]
# monotone: +1 carbs features (what-if curves move the right way)
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
    # time since meal (min, true timestamps so gaps don't undercount), cap 1440
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
    # --- v3 Tier-1: meal composition (rolling sums, past-only like carbs) ---
    u["fat_60"] = g["meal_fat"].transform(
        lambda s: s.rolling(12, min_periods=1).sum())
    u["fat_120"] = g["meal_fat"].transform(
        lambda s: s.rolling(24, min_periods=1).sum())
    u["protein_60"] = g["meal_protein"].transform(
        lambda s: s.rolling(12, min_periods=1).sum())
    u["protein_120"] = g["meal_protein"].transform(
        lambda s: s.rolling(24, min_periods=1).sum())
    # --- v3 Tier-1: glucose acceleration (2nd derivative, within-group shift) ---
    # NaN on warmup; LightGBM handles NaN natively (no imputation = no leakage)
    u["glucose_accel"] = (u["slope_15"] -
                          u.groupby("patient_id")["slope_15"].shift(1))
    # --- v3 Tier-1: per-patient baselines, EXPANDING (past+current only) ---
    # No future rows anywhere: expanding() at row t sees rows <= t only.
    # (Including the current row in its own baseline is standard and uses
    # no future data — same as slope features using current glucose.)
    for _, grp in u.groupby("patient_id"):
        gv = grp["glucose"].to_numpy(dtype=float)
        s = pd.Series(gv)
        mu = s.expanding().mean().to_numpy().copy()
        sd = s.expanding().std().to_numpy().copy()
        sd[sd < 1e-9] = np.nan  # constant run-in -> NaN, never inf
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (gv - mu) / sd
        z[~np.isfinite(z)] = np.nan
        u.loc[grp.index, "glucose_z"] = z
        hr = grp["hr"].to_numpy(dtype=float)
        hrm = pd.Series(hr).expanding().mean().to_numpy()
        u.loc[grp.index, "hr_vs_baseline"] = grp["hr_30"].to_numpy() - hrm
        # post-meal rise baseline: delta_tau = g[tau]-g[tau+12] at past meals,
        # usable only once both endpoints are recorded (tau+12 <= t).
        mc = grp["meal_carbs"].to_numpy()
        n = len(grp)
        delta = np.full(n, np.nan)
        if n > 12:
            meals = np.where(mc[:-12] > 0.5)[0]
            delta[meals] = gv[meals] - gv[meals + 12]
        med = pd.Series(delta).expanding().median().shift(12).to_numpy()
        u.loc[grp.index, "pm_rise_baseline"] = med
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


def main(tag=None):
    """tag=None writes the legacy v2 paths. Any tag (e.g. 'v3') redirects
    ALL outputs to tagged paths so frozen v2 artifacts are never touched."""
    suf = "" if tag is None else f"_{tag}"
    paths = {
        "metrics": f"results/lgbm_metrics{suf}.json",
        "oof": f"results/oof_{'frozen' if tag is None else tag}.parquet",
        "reg": f"models/lgbm_reg60{suf}.txt",
        "clf": f"models/lgbm_event{suf}.txt",
        "shap": f"results/shap_summary{suf}.png",
    }
    if tag is not None:
        print(f"tag={tag} -> {paths}")
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
    d["t2d"] = d["a1c"] >= 6.5
    print(f"T2D subset: {d['t2d'].sum()}/{len(d)} rows, "
          f"{d.loc[d['t2d'], 'patient_id'].nunique()} patients")

    def split_metrics(mask, tag):
        o = {}
        for h in H:
            yt = d.loc[mask, f"target_glucose_{h}"].to_numpy()
            yp = oof_reg[h][mask.to_numpy()]
            o[f"{tag}rmse_{h}"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
            o[f"{tag}mae_{h}"] = float(np.mean(np.abs(yt - yp)))
        ye = d.loc[mask, "event_60"].to_numpy()
        pe = oof_clf[mask.to_numpy()]
        o[f"{tag}event_auroc"] = float(roc_auc_score(ye, pe))
        o[f"{tag}event_auprc"] = float(average_precision_score(ye, pe))
        return o

    res = {"n_rows": int(len(d)), "features": FEATS,
           "monotone_constraints": MONO}
    allm = pd.Series(True, index=d.index)
    res.update(split_metrics(allm, ""))
    res.update(split_metrics(d["t2d"], "t2d_"))
    for h in H:
        res[f"rmse_{h}_persistence"] = base["persistence"][f"rmse_{h}"]
    # false alerts per patient-day: alert=prob>=0.5, false=alert & no event
    alert = (oof_clf >= 0.5).astype(int)
    false = ((alert == 1) & (d["event_60"].to_numpy() == 0)).sum()
    patient_days = len(d) / 288.0
    res["false_alerts_per_patient_day"] = float(false / patient_days)
    print(json.dumps({k: v for k, v in res.items() if k != "features"},
                     indent=2))

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open(paths["metrics"], "w") as f:
        json.dump(res, f, indent=2)
    oof = pd.DataFrame({"patient_id": d["patient_id"].to_numpy(),
                        "timestamp": d["timestamp"],
                        "t2d": d["t2d"].to_numpy(),
                        "event": d["event_60"].to_numpy(),
                        "escore": oof_clf})
    for h in H:
        oof[f"y{h}"] = d[f"target_glucose_{h}"].to_numpy()
        oof[f"p{h}"] = oof_reg[h]
    oof.to_parquet(paths["oof"], index=False)
    print(f"saved {paths['oof']} rows={len(oof)}")
    # refit on all data for demo/freeze; save 60-min reg + event clf
    r60 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                            num_leaves=31, min_child_samples=50,
                            monotone_constraints=MONO, verbose=-1)
    r60.fit(X, d["target_glucose_60"].to_numpy())
    r60.booster_.save_model(paths["reg"])
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             num_leaves=31, min_child_samples=100,
                             monotone_constraints=MONO, verbose=-1)
    clf.fit(X, d["event_60"].to_numpy())
    clf.booster_.save_model(paths["clf"])
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
    plt.savefig(paths["shap"], dpi=120)
    plt.close()
    print(f"saved {paths['reg']} {paths['clf']} {paths['shap']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                    help="redirect all outputs to tagged paths (v2-safe)")
    main(**vars(ap.parse_args()))
