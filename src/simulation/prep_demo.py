"""Phase 5 prep: hero demo window + triage (frozen-model inference only)."""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from lightgbm_v1 import FEATS, add_features, add_static  # noqa: E402

REG = lgb.Booster(model_file="models/lgbm_reg60.txt")
CLF = lgb.Booster(model_file="models/lgbm_event.txt")
with open("results/conformal.json") as f:
    Q90 = json.load(f)["q90_half_width"]


def apply_walk(x0):
    """+15-min walk via dense wearable signals (steps_* dropped: dead feature)."""
    x = x0.copy()
    x[0, FEATS.index("activity_60")] += 40.0
    x[0, FEATS.index("hr_30")] += 10.0
    return x


def apply_ragi(x0, grams=20.0):
    """Swap rice->ragi: -grams carbs from 60/120-min windows (floor 0)."""
    x = x0.copy()
    x[0, FEATS.index("carbs_60")] = max(0.0, x[0, FEATS.index("carbs_60")] - grams)
    x[0, FEATS.index("carbs_120")] = max(0.0, x[0, FEATS.index("carbs_120")] - grams)
    return x


def pick_demo_state(h: pd.DataFrame, reg, q90: float):
    """Earliest subj-39 state: glucose<180, band crosses 180, actual crosses
    in 30-70 min, post-meal, HR present, BOTH interventions lower pred."""
    h = h.sort_values("timestamp").reset_index(drop=True)
    hi = reg.predict(h[FEATS].to_numpy(dtype=float)) + q90
    fut = [h["glucose"].shift(-i) for i in range(1, 13)]
    h = h.copy()
    h["fmax60"] = pd.concat(fut, axis=1).max(axis=1)
    cand = h[(h["glucose"] < 180) & (hi > 180) & (h["fmax60"] > 180)
             & (h["carbs_60"] > 5) & (h["hr_30"].notna())].copy()
    for _, s in cand.sort_values("timestamp").iterrows():
        x0 = s[FEATS].to_numpy(dtype=float).reshape(1, -1)
        p0 = float(reg.predict(x0)[0])
        pw = float(reg.predict(apply_walk(x0))[0])
        pc = float(reg.predict(apply_ragi(x0))[0])
        cross = h[(h["timestamp"] > s["timestamp"]) & (h["glucose"] > 180)]
        lead = int((cross["timestamp"].min() - s["timestamp"]).total_seconds() / 60) \
            if len(cross) else -1
        if pw <= p0 - 2 and pc <= p0 - 2 and 30 <= lead <= 70:
            return s, p0, pw, pc, lead
    raise AssertionError("no demo window where both interventions lower pred")


def main():
    u = pd.read_parquet("data/processed/unified.parquet")
    u["timestamp"] = pd.to_datetime(u["timestamp"])
    d = add_static(add_features(u))
    m = d["target_glucose_60"].notna() & d["event_60"].notna()
    d = d.loc[m].reset_index(drop=True)
    X = d[FEATS].to_numpy(dtype=float)
    d["pred_60"] = REG.predict(X)
    d["risk"] = CLF.predict(X)
    d["hi_60"] = d["pred_60"] + Q90
    h39 = d[d["patient_id"] == 39]
    s, p0, pw, pc, lead = pick_demo_state(h39, REG, Q90)
    demo = {"patient_id": 39, "timestamp": str(s["timestamp"]),
            "glucose_now": round(float(s["glucose"]), 1),
            "pred_60": round(p0, 1),
            "pred_walk": round(pw, 1), "pred_fewer_carbs": round(pc, 1),
            "risk": round(float(s["risk"]), 3),
            "lead_min_to_180": lead}
    print(json.dumps(demo, indent=2))
    with open("results/demo_window.json", "w") as f:
        json.dump(demo, f, indent=2)
    # triage: latest-row 60-min event probability per patient (replay snapshot)
    last = d.sort_values("timestamp").groupby("patient_id").tail(1)
    tri = last[["patient_id", "timestamp", "glucose", "pred_60", "risk"]].copy()
    tri = tri.rename(columns={"risk": "risk_60"})
    tri = tri.sort_values("risk_60", ascending=False).reset_index(drop=True)
    tri.to_csv("results/triage.csv", index=False)
    print(tri.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
