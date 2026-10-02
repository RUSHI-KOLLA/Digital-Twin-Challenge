"""Task D: cross-cohort generalisation — frozen v2 model, zero training.

Train: CGMacros (frozen models/lgbm_reg60.txt + lgbm_event.txt, untouched).
Test: ShanghaiT2DM (Zhao et al., Sci Data 2023, CC-BY 4.0, 109 files,
15-min CGM). CGM-only cohort: no meals/HR (zeros/NaN, documented);
5-min grid via interpolation; same labels; patient-wise aggregation +
bootstrap-by-subject CIs. Saves results/cross_cohort.json.
"""
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from lightgbm_v1 import add_features, add_static  # noqa: E402

# Frozen-v2 feature set (15). The branch's add_features may compute MORE
# (v3 experiments); the frozen model only ever sees these 15.
V2_FEATS = ["glucose", "slope_15", "slope_30",
            "roll_mean_30", "roll_std_30", "roll_mean_60", "roll_std_60",
            "carbs_60", "carbs_120", "time_since_meal",
            "hr_30", "activity_60", "tod_sin", "tod_cos", "dow"]

SRC = "/tmp/sh/x/Shanghai_T2DM"
N_BOOT = 1000


def load_shanghai() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(os.path.join(SRC, "*.xls*"))):
        pid = os.path.basename(f).split("_")[0]
        try:
            d = pd.read_excel(f, sheet_name=0, header=None)
        except Exception as e:
            print("skip", f, str(e)[:80])
            continue
        d = d.iloc[1:, :2]
        d.columns = ["timestamp", "glucose"]
        d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
        d["glucose"] = pd.to_numeric(d["glucose"], errors="coerce")
        d = d.dropna(subset=["timestamp", "glucose"])
        if not len(d):
            continue
        d["patient_id"] = float(pid)
        parts.append(d)
    u = pd.concat(parts, ignore_index=True)
    u["hr"] = np.nan
    u["steps"] = 0.0
    u["meal_carbs"] = 0.0
    u["meal_fat"] = 0.0
    u["meal_protein"] = 0.0
    u["activity_kcal"] = 0.0
    u["mets"] = np.nan
    return u[["patient_id", "timestamp", "glucose", "hr", "steps",
              "meal_carbs", "meal_protein", "meal_fat",
              "activity_kcal", "mets"]]


def to_5min(u: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in u.sort_values("timestamp").groupby("patient_id"):
        g = g.set_index("timestamp").sort_index()
        r = g.resample("5min").agg({"glucose": "mean", "hr": "mean",
                                    "steps": "sum", "meal_carbs": "sum",
                                    "meal_protein": "sum", "meal_fat": "sum",
                                    "activity_kcal": "sum", "mets": "mean",
                                    "patient_id": "first"})
        r["glucose"] = r["glucose"].interpolate(method="linear", limit=3,
                                               limit_area="inside")
        r = r.dropna(subset=["glucose"]).reset_index()
        out.append(r)
    return pd.concat(out, ignore_index=True)


def boot(pids, df, fn):
    rng = np.random.RandomState(0)
    groups = {p: g for p, g in df.groupby("patient_id")}
    vals = [fn(pd.concat([groups[p] for p in
                          rng.choice(pids, size=len(pids), replace=True)]))
            for _ in range(N_BOOT)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main():
    u = load_shanghai()
    print(f"shanghai rows={len(u)} patients={u['patient_id'].nunique()}")
    d = add_static(add_features(to_5min(u)))
    m = d["target_glucose_120"].notna() & d["event_60"].notna()
    d = d.loc[m].reset_index(drop=True)
    d["t2d"] = True  # cohort label: all T2D by study design (no per-row labs)
    X = d[V2_FEATS].to_numpy(dtype=float)
    reg = lgb.Booster(model_file="models/lgbm_reg60.txt")
    clf = lgb.Booster(model_file="models/lgbm_event.txt")
    d["p60"] = reg.predict(X)
    d["escore"] = clf.predict(X)
    pids = np.array(sorted(d["patient_id"].unique()))
    rmse = lambda s: float(np.sqrt(np.mean((s["target_glucose_60"] - s["p60"]) ** 2)))
    res = {
        "n_patients": int(len(pids)), "n_rows": int(len(d)),
        "event_60_rate": float(d["event_60"].mean()),
        "rmse_60": rmse(d),
        "rmse_60_ci": boot(pids, d, rmse),
        "event_auroc": float(roc_auc_score(d["event_60"], d["escore"])),
        "event_auprc": float(average_precision_score(d["event_60"], d["escore"])),
        "method": "frozen v2 (trained on CGMacros, zero refit); CGM-only "
                  "cohort (meals/HR absent -> zeros/NaN); same 5-min labels",
    }
    res["event_auroc_ci"] = boot(pids, d, lambda s: float(
        roc_auc_score(s["event_60"], s["escore"])))
    v2 = json.load(open("results/metrics_ci.json"))
    res["cgmacros_reference"] = {
        "rmse_60": v2["frozen"]["rmse_60"],
        "rmse_60_ci": v2["frozen"]["rmse_60_ci"],
        "auroc": v2["frozen"]["auroc"]}
    res["drop_rmse"] = res["rmse_60"] - res["cgmacros_reference"]["rmse_60"]
    res["drop_auroc"] = res["cgmacros_reference"]["auroc"] - res["event_auroc"]
    json.dump(res, open("results/cross_cohort.json", "w"), indent=2)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in res.items() if k != "cgmacros_reference"},
                     indent=1))
    print("saved results/cross_cohort.json (v2 model files untouched)")


if __name__ == "__main__":
    main()
