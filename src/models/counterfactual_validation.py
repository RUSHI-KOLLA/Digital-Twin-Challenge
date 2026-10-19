"""TASK 3: Counterfactual validation - compare what-if predictions to reality.

Use HR increase as proxy for post-meal activity (more complete than steps).
Find meals with high HR increase vs matched meals with low HR increase.
Measure ACTUAL 60-min glucose rise difference.
Compare to what-if simulator predictions (adding walk = increase HR/activity).
"""
import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

IN = "data/processed/unified.parquet"
MODEL_REG = "models/lgbm_reg60.txt"
OUT_JSON = "results/counterfactual_validation.json"
OUT_PNG = "results/counterfactual.png"

FEATS = [
    "glucose", "slope_15", "slope_30",
    "roll_mean_30", "roll_std_30", "roll_mean_60", "roll_std_60",
    "carbs_60", "carbs_120", "time_since_meal",
    "hr_30", "activity_60",
    "tod_sin", "tod_cos", "dow",
]

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


def find_meal_events(df: pd.DataFrame) -> pd.DataFrame:
    meals = df[df["meal_carbs"] > 10].copy()
    meals = meals.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    
    results = []
    for _, meal in meals.iterrows():
        pid = meal["patient_id"]
        t = meal["timestamp"]
        carbs = meal["meal_carbs"]
        
        g0 = meal["glucose"]
        
        future = df[(df["patient_id"] == pid) & (df["timestamp"] > t) & 
                    (df["timestamp"] <= t + pd.Timedelta(minutes=60))]
        if len(future) == 0:
            continue
        g60 = future["glucose"].iloc[-1]
        
        pre_hr = df[(df["patient_id"] == pid) & (df["timestamp"] <= t)]["hr"].mean()
        post_hr = df[(df["patient_id"] == pid) & (df["timestamp"] > t) & 
                     (df["timestamp"] <= t + pd.Timedelta(minutes=60))]["hr"].mean()
        
        if pd.isna(pre_hr) or pd.isna(post_hr):
            continue
            
        hr_increase = post_hr - pre_hr
        
        post_meal = df[(df["patient_id"] == pid) & (df["timestamp"] > t) & 
                       (df["timestamp"] <= t + pd.Timedelta(minutes=60))]
        total_steps = post_meal["steps"].sum()
        
        results.append({
            "patient_id": pid,
            "meal_time": t,
            "meal_carbs": carbs,
            "glucose_0": g0,
            "glucose_60": g60,
            "glucose_rise": g60 - g0,
            "hr_increase": hr_increase,
            "post_meal_steps": total_steps,
            "pre_hr": pre_hr,
            "post_hr": post_hr,
        })
    
    return pd.DataFrame(results)


def match_meals_propensity(meal_df: pd.DataFrame, hr_threshold: float = 5.0) -> pd.DataFrame:
    active = meal_df[meal_df["hr_increase"] > hr_threshold].copy()
    inactive = meal_df[meal_df["hr_increase"] <= hr_threshold].copy()
    
    if len(active) == 0 or len(inactive) == 0:
        return pd.DataFrame()
    
    print(f"Active meals (HR+ > {hr_threshold}): {len(active)}")
    print(f"Inactive meals (HR+ <= {hr_threshold}): {len(inactive)}")
    
    active = active.copy()
    inactive = inactive.copy()
    active["hod"] = active["meal_time"].dt.hour + active["meal_time"].dt.minute / 60.0
    inactive["hod"] = inactive["meal_time"].dt.hour + inactive["meal_time"].dt.minute / 60.0
    
    match_feats = ["patient_id", "meal_carbs", "glucose_0", "pre_hr", "hod"]
    
    combined = pd.concat([active.assign(treatment=1), inactive.assign(treatment=0)], ignore_index=True)
    
    X = combined[match_feats].fillna(combined[match_feats].median())
    y = combined["treatment"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X_scaled, y)
    
    active["ps"] = ps_model.predict_proba(scaler.transform(active[match_feats].fillna(active[match_feats].median())))[:, 1]
    inactive["ps"] = ps_model.predict_proba(scaler.transform(inactive[match_feats].fillna(inactive[match_feats].median())))[:, 1]
    
    matches = []
    inactive_used = set()
    
    for _, a in active.iterrows():
        candidates = inactive[~inactive.index.isin(inactive_used)].copy()
        if len(candidates) == 0:
            continue
        candidates["ps_dist"] = abs(candidates["ps"] - a["ps"])
        same_patient = candidates[candidates["patient_id"] == a["patient_id"]]
        if len(same_patient) > 0:
            best = same_patient.loc[same_patient["ps_dist"].idxmin()]
        else:
            best = candidates.loc[candidates["ps_dist"].idxmin()]
        
        inactive_used.add(best.name)
        matches.append({
            "patient_id": a["patient_id"],
            "meal_time_active": a["meal_time"],
            "meal_time_inactive": best["meal_time"],
            "carbs_active": a["meal_carbs"],
            "carbs_inactive": best["meal_carbs"],
            "glucose_0_active": a["glucose_0"],
            "glucose_0_inactive": best["glucose_0"],
            "glucose_rise_active": a["glucose_rise"],
            "glucose_rise_inactive": best["glucose_rise"],
            "hr_inc_active": a["hr_increase"],
            "hr_inc_inactive": best["hr_increase"],
            "observed_diff": best["glucose_rise"] - a["glucose_rise"],
            "ps_active": a["ps"],
            "ps_inactive": best["ps"],
        })
    
    return pd.DataFrame(matches)


def main():
    print("Loading data...")
    df = pd.read_parquet(IN)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    print("Adding features...")
    df = add_features(df)
    
    print("Finding meal events...")
    meals = find_meal_events(df)
    print(f"Found {len(meals)} meal events with HR data")
    print(f"  HR increase > 5: {(meals['hr_increase'] > 5).sum()}")
    print(f"  HR increase > 10: {(meals['hr_increase'] > 10).sum()}")
    print(f"  HR increase <= 5: {(meals['hr_increase'] <= 5).sum()}")
    
    threshold = 5.0
    for thresh in [5.0, 7.5, 10.0]:
        print(f"\n--- Threshold: HR increase > {thresh} ---")
        matches = match_meals_propensity(meals, hr_threshold=thresh)
        print(f"Matched pairs: {len(matches)}")
        if len(matches) >= 10:
            threshold = thresh
            break
    
    if len(matches) < 5:
        print("Too few matches, using threshold 5.0")
        matches = match_meals_propensity(meals, hr_threshold=5.0)
        threshold = 5.0
    
    if len(matches) == 0:
        print("No matched pairs found!")
        return
    
    print("Loading frozen model...")
    model = lgb.Booster(model_file=MODEL_REG)
    
    observed_diffs = matches["observed_diff"].values
    print(f"Observed mean diff (inactive - active): {observed_diffs.mean():.2f} mg/dL")
    print(f"Observed std: {observed_diffs.std():.2f} mg/dL")
    
    print("Running what-if simulations...")
    predicted_diffs = []
    valid_observed = []
    
    for _, m in matches.iterrows():
        feat_active = df[(df["patient_id"] == m["patient_id"]) & 
                         (df["timestamp"] == m["meal_time_active"])]
        feat_inactive = df[(df["patient_id"] == m["patient_id"]) & 
                           (df["timestamp"] == m["meal_time_inactive"])]
        
        if len(feat_active) == 0 or len(feat_inactive) == 0:
            continue
            
        feat_active = feat_active.iloc[0].copy()
        feat_inactive = feat_inactive.iloc[0].copy()
        
        # Check for NaN in features
        if feat_inactive[FEATS].isna().any():
            continue
            
        pred_inactive_base = model.predict(feat_inactive[FEATS].to_numpy().reshape(1, -1))[0]
        
        hr_diff = m["hr_inc_active"] - m["hr_inc_inactive"]
        feat_inactive_walk = feat_inactive.copy()
        feat_inactive_walk["hr_30"] += max(0, hr_diff)
        feat_inactive_walk["activity_60"] += 100
        
        if feat_inactive_walk[FEATS].isna().any():
            continue
            
        pred_inactive_walk = model.predict(feat_inactive_walk[FEATS].to_numpy().reshape(1, -1))[0]
        
        walk_effect = pred_inactive_walk - pred_inactive_base
        predicted_diffs.append(-walk_effect)
        valid_observed.append(m["observed_diff"])
    
    predicted_diffs = np.array(predicted_diffs)
    valid_observed = np.array(valid_observed)
    print(f"Valid pairs for comparison: {len(predicted_diffs)}")
    print(f"Predicted walk effect (mean): {predicted_diffs.mean():.2f} mg/dL")
    print(f"Predicted walk effect (std): {predicted_diffs.std():.2f} mg/dL")
    
    valid = ~(np.isnan(valid_observed) | np.isnan(predicted_diffs))
    obs_valid = valid_observed[valid]
    pred_valid = predicted_diffs[valid]
    
    if len(obs_valid) > 1:
        corr = np.corrcoef(obs_valid, pred_valid)[0, 1]
        rmse = np.sqrt(np.mean((obs_valid - pred_valid) ** 2))
        mae = np.mean(np.abs(obs_valid - pred_valid))
        print(f"Correlation (observed vs predicted): {corr:.3f}")
        print(f"RMSE: {rmse:.2f} mg/dL")
        print(f"MAE: {mae:.2f} mg/dL")
    else:
        corr = np.nan
        rmse = np.nan
        mae = np.nan
        print("Insufficient valid pairs for correlation")
    
    os.makedirs("results", exist_ok=True)
    results = {
        "n_matched_pairs": int(len(matches)),
        "n_valid_pairs": int(len(obs_valid)),
        "hr_threshold_used": float(threshold),
        "observed_mean_diff": float(observed_diffs.mean()),
        "observed_std_diff": float(observed_diffs.std()),
        "predicted_effect_mean": float(predicted_diffs.mean()) if len(predicted_diffs) > 0 else None,
        "predicted_effect_std": float(predicted_diffs.std()) if len(predicted_diffs) > 0 else None,
        "correlation": float(corr) if not np.isnan(corr) else None,
        "rmse": float(rmse) if not np.isnan(rmse) else None,
        "mae": float(mae) if not np.isnan(mae) else None,
    }
    
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON}")
    
    if len(obs_valid) > 1:
        plt.figure(figsize=(6, 6))
        plt.scatter(obs_valid, pred_valid, alpha=0.6)
        lims = [min(obs_valid.min(), pred_valid.min()) - 5,
                max(obs_valid.max(), pred_valid.max()) + 5]
        plt.plot(lims, lims, 'r--', alpha=0.5, label='Perfect prediction')
        plt.xlabel('Observed glucose reduction from activity (mg/dL)')
        plt.ylabel('Predicted glucose reduction from walk (mg/dL)')
        plt.title(f'Counterfactual Validation (HR+ > {threshold})\nCorr={corr:.3f}, RMSE={rmse:.1f} mg/dL, n={len(obs_valid)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=150)
        plt.close()
        print(f"Saved {OUT_PNG}")
    
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
