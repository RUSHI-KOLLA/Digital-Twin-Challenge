"""Phase 2a: labels + persistence/linear baselines, patient-wise GroupKFold.

Labels:
- event_60 = glucose >180 at any point in next 60 min (t+1..t+12 bins)
- target_glucose_{30,60,120} = glucose at t+6, t+12, t+24

Baselines:
- persistence: future = current glucose
- linear: 30-min slope (t - t-6) capped at +-3 mg/dL/min, extrapolate.

Metrics: RMSE/MAE per horizon + event rate -> results/metrics.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

IN = "data/processed/unified.parquet"
OUT = "results/metrics.json"
H = {30: 6, 60: 12, 120: 24}
CAP = 3.0  # mg/dL/min


def add_labels(u: pd.DataFrame) -> pd.DataFrame:
    u = u.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    for h, steps in H.items():
        u[f"target_glucose_{h}"] = u.groupby("patient_id")["glucose"].shift(-steps)
    # event_60: any of next 12 bins >180
    fut = [u.groupby("patient_id")["glucose"].shift(-i) for i in range(1, 13)]
    futmax = pd.concat(fut, axis=1).max(axis=1)
    u["event_60"] = (futmax > 180).astype(float)
    return u


def predict_linear(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").reset_index(drop=True)
    dt_min = g["timestamp"].diff(6).dt.total_seconds() / 60.0
    dg = g["glucose"] - g["glucose"].shift(6)
    slope = dg / dt_min
    # only trust ~30-min spacing; else fallback slope 0 (=persistence)
    slope = slope.where((dt_min > 25) & (dt_min < 35), 0.0).fillna(0.0)
    slope = slope.clip(-CAP, CAP)
    for h, steps in H.items():
        mins = steps * 5
        g[f"lin_{h}"] = g["glucose"] + slope * mins
    return g


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    u = pd.read_parquet(IN)
    u["timestamp"] = pd.to_datetime(u["timestamp"])
    u = add_labels(u)
    parts = []
    for pid, g in u.groupby("patient_id"):
        parts.append(predict_linear(g))
    u = pd.concat(parts, ignore_index=True).sort_values(
        ["patient_id", "timestamp"]).reset_index(drop=True)
    u["pers_30"] = u["glucose"]
    u["pers_60"] = u["glucose"]
    u["pers_120"] = u["glucose"]

    mask_event = u["event_60"].notna()
    event_rate = float(u.loc[mask_event, "event_60"].mean())
    print(f"event_60 rate={event_rate * 100:.2f}% n={mask_event.sum()}")

    gkf = GroupKFold(n_splits=5)
    groups = u["patient_id"].to_numpy()
    pers = {h: {"rmse": [], "mae": []} for h in H}
    lin = {h: {"rmse": [], "mae": []} for h in H}
    for tr, te in gkf.split(u, groups=groups):
        d = u.iloc[te]
        for h in H:
            m = d[f"target_glucose_{h}"].notna()
            # linear needs slope warmup; fall back to persistence where NaN
            lp = d[f"lin_{h}"].fillna(d["glucose"])
            pers[h]["rmse"].append(rmse(d.loc[m, f"target_glucose_{h}"],
                                        d.loc[m, f"pers_{h}"]))
            pers[h]["mae"].append(mae(d.loc[m, f"target_glucose_{h}"],
                                      d.loc[m, f"pers_{h}"]))
            lin[h]["rmse"].append(rmse(d.loc[m, f"target_glucose_{h}"],
                                       lp.loc[m]))
            lin[h]["mae"].append(mae(d.loc[m, f"target_glucose_{h}"],
                                     lp.loc[m]))

    pers_mean = {}
    lin_mean = {}
    for h in H:
        pers_mean[f"rmse_{h}"] = float(np.mean(pers[h]["rmse"]))
        pers_mean[f"mae_{h}"] = float(np.mean(pers[h]["mae"]))
        lin_mean[f"rmse_{h}"] = float(np.mean(lin[h]["rmse"]))
        lin_mean[f"mae_{h}"] = float(np.mean(lin[h]["mae"]))
    out = {
        "n_rows": int(len(u)),
        "n_patients": int(u["patient_id"].nunique()),
        "event_60_rate": event_rate,
        "persistence": pers_mean,
        "linear_slope30_cap3": lin_mean,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"THE ONE NUMBER persistence RMSE@60 = {out['persistence']['rmse_60']:.2f}")


if __name__ == "__main__":
    main()
