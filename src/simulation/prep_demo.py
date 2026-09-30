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
    # hero demo window: subj 39, glucose<180 now, band crosses 180,
    # actual crosses 180 within 30-70 min, post-meal, HR present, and BOTH
    # frozen-model interventions (walk, -20g carbs) honestly lower pred.
    h = d[d["patient_id"] == 39].sort_values("timestamp").reset_index(drop=True)
    fut = [h["glucose"].shift(-i) for i in range(1, 13)]
    h["fmax60"] = pd.concat(fut, axis=1).max(axis=1)
    cand = h[(h["glucose"] < 180) & (h["hi_60"] > 180) & (h["fmax60"] > 180)
             & (h["carbs_60"] > 5) & (h["hr_30"].notna())].copy()
    i_c60, i_c120 = FEATS.index("carbs_60"), FEATS.index("carbs_120")
    i_s30, i_s60 = FEATS.index("steps_30"), FEATS.index("steps_60")
    i_act, i_hr = FEATS.index("activity_60"), FEATS.index("hr_30")
    picked = None
    for _, s in cand.sort_values("timestamp").iterrows():
        x0 = s[FEATS].to_numpy(dtype=float).reshape(1, -1)
        p0 = float(REG.predict(x0)[0])
        xw = x0.copy()
        xw[0, i_s30] += 1500.0
        xw[0, i_s60] += 1500.0
        xw[0, i_act] += 40.0
        xw[0, i_hr] += 10.0
        xc = x0.copy()
        xc[0, i_c60] = max(0.0, xc[0, i_c60] - 20.0)
        xc[0, i_c120] = max(0.0, xc[0, i_c120] - 20.0)
        pw, pc = float(REG.predict(xw)[0]), float(REG.predict(xc)[0])
        cross = h[(h["timestamp"] > s["timestamp"]) & (h["glucose"] > 180)]
        lead = int((cross["timestamp"].min() - s["timestamp"]).total_seconds() / 60) \
            if len(cross) else -1
        if pw <= p0 - 2 and pc <= p0 - 2 and 30 <= lead <= 70:
            picked = (s, p0, pw, pc, lead)
            break
    assert picked, "no demo window where both interventions lower pred"
    s, p0, pw, pc, lead = picked
    demo = {"patient_id": 39, "timestamp": str(s["timestamp"]),
            "glucose_now": round(float(s["glucose"]), 1),
            "pred_60": round(p0, 1),
            "pred_walk": round(pw, 1), "pred_fewer_carbs": round(pc, 1),
            "risk": round(float(s["risk"]), 3),
            "lead_min_to_180": lead}
    print(json.dumps(demo, indent=2))
    with open("results/demo_window.json", "w") as f:
        json.dump(demo, f, indent=2)
    # triage: latest-row 2-h risk proxy = current event prob per patient
    last = d.sort_values("timestamp").groupby("patient_id").tail(1)
    tri = last[["patient_id", "timestamp", "glucose", "pred_60", "risk"]].copy()
    tri = tri.rename(columns={"risk": "risk_2h"})
    tri = tri.sort_values("risk_2h", ascending=False).reset_index(drop=True)
    tri.to_csv("results/triage.csv", index=False)
    print(tri.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
