"""Phase 3-ML: personalization lift + what-if simulate + conformal band.

1. Personalization: T2D subset (bio A1c>=6.5). Per patient: cohort model
   trained on all OTHER patients -> RMSE on patient's 2nd half; then
   fine-tune (init_model, +50 rounds on 1st half) -> RMSE on 2nd half.
   Reports mean lift. Saves results/personalization.json.
2. What-if: predict(state) vs simulate(state, -20g carbs / +15-min walk)
   with frozen models/lgbm_reg60.txt. Monotone (+1 carbs/-1 steps) keeps
   curves sane. Saves results/whatif.json. GATE: walk lowers peak >=1 case.
3. Conformal: split-conformal 90% band from OOF residuals
   (same principle as MAPIE/crepes, no new dep). Saves results/conformal.json.
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from lightgbm_v1 import FEATS, MONO, add_features, add_static

PARAMS = {"objective": "regression", "metric": "rmse", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50,
          "monotone_constraints": MONO}
N_ROUNDS = 200
FT_ROUNDS = 50


def personalization(d: pd.DataFrame, t2d: list) -> dict:
    rows = []
    for pid in t2d:
        p = d[d["patient_id"] == pid].sort_values("timestamp").reset_index(drop=True)
        if len(p) < 200:
            continue
        cut = len(p) // 2
        first, second = p.iloc[:cut], p.iloc[cut:]
        Xc = d[d["patient_id"] != pid][FEATS].to_numpy()
        yc = d[d["patient_id"] != pid]["target_glucose_60"].to_numpy()
        tr = lgb.Dataset(Xc, label=yc)
        cohort = lgb.train(PARAMS, tr, num_boost_round=N_ROUNDS)
        Xs = second[FEATS].to_numpy()
        ys = second["target_glucose_60"].to_numpy()
        rmse_cohort = float(np.sqrt(np.mean((ys - cohort.predict(Xs)) ** 2)))
        ft = lgb.Dataset(first[FEATS].to_numpy(),
                         label=first["target_glucose_60"].to_numpy())
        pers = lgb.train(PARAMS, ft, num_boost_round=FT_ROUNDS,
                         init_model=cohort)
        rmse_pers = float(np.sqrt(np.mean((ys - pers.predict(Xs)) ** 2)))
        rows.append({"patient_id": float(pid), "n_second": int(len(second)),
                     "rmse_cohort": rmse_cohort, "rmse_personal": rmse_pers,
                     "lift": rmse_cohort - rmse_pers})
        print(rows[-1])
    lifts = [r["lift"] for r in rows]
    return {"patients": rows, "mean_lift": float(np.mean(lifts)) if lifts else 0.0,
            "n": len(rows)}


def whatif_demo(d: pd.DataFrame) -> dict:
    reg = lgb.Booster(model_file="models/lgbm_reg60.txt")
    # pick a real high-risk pre-meal state: carbs>0 in last hour, glucose rising
    cand = d[(d["carbs_60"] > 10) & (d["slope_30"] > 0.3)].copy()
    assert len(cand), "no high-risk state found"
    s = cand.sort_values("glucose", ascending=False).iloc[0]
    x0 = s[FEATS].to_numpy(dtype=float).reshape(1, -1)
    p0 = float(reg.predict(x0)[0])
    i_carbs60 = FEATS.index("carbs_60")
    i_carbs120 = FEATS.index("carbs_120")
    i_steps30 = FEATS.index("steps_30")
    i_steps60 = FEATS.index("steps_60")
    # -20g carbs (floor 0)
    x1 = x0.copy()
    x1[0, i_carbs60] = max(0.0, x1[0, i_carbs60] - 20.0)
    x1[0, i_carbs120] = max(0.0, x1[0, i_carbs120] - 20.0)
    p1 = float(reg.predict(x1)[0])
    # +15-min walk ~= +1500 steps +40 kcal activity +10 bpm HR.
    # NOTE: raw steps exist for only 1/45 patients, so the model learned
    # ~zero gain on steps_* (monotone-flat, never wrong-way). The dense
    # wearable signals (activity_60, hr_30) carry the walk effect.
    x2 = x0.copy()
    x2[0, i_steps30] += 1500.0
    x2[0, i_steps60] += 1500.0
    x2[0, FEATS.index("activity_60")] += 40.0
    x2[0, FEATS.index("hr_30")] = float(x2[0, FEATS.index("hr_30")] or 0) + 10.0
    p2 = float(reg.predict(x2)[0])
    out = {"patient_id": float(s["patient_id"]),
           "timestamp": str(s["timestamp"]),
           "glucose_now": float(s["glucose"]),
           "pred_base": p0, "pred_fewer_carbs": p1, "pred_walk": p2,
           "walk_lowers": bool(p2 < p0), "carbs_lower": bool(p1 <= p0)}
    print(json.dumps(out, indent=2))
    assert out["walk_lowers"], "GATE FAIL: walk did not lower predicted peak"
    return out


def conformal(d: pd.DataFrame) -> dict:
    # split-conformal: fit on 80% patients, calibrate on 20% (by patient)
    pids = np.array(sorted(d["patient_id"].unique()))
    rng = np.random.RandomState(0)
    rng.shuffle(pids)
    cut = int(len(pids) * 0.8)
    tr_p, ca_p = set(pids[:cut]), set(pids[cut:])
    tr = d[d["patient_id"].isin(tr_p)]
    ca = d[d["patient_id"].isin(ca_p)]
    r = lgb.LGBMRegressor(n_estimators=N_ROUNDS, learning_rate=0.05,
                          num_leaves=31, min_child_samples=50,
                          monotone_constraints=MONO, verbose=-1)
    r.fit(tr[FEATS].to_numpy(), tr["target_glucose_60"].to_numpy())
    resid = np.abs(ca["target_glucose_60"].to_numpy() - r.predict(ca[FEATS].to_numpy()))
    q90 = float(np.quantile(resid, 0.90))
    # coverage check on calibration set
    lo = r.predict(ca[FEATS].to_numpy()) - q90
    hi = r.predict(ca[FEATS].to_numpy()) + q90
    cov = float(np.mean((ca["target_glucose_60"].to_numpy() >= lo) &
                        (ca["target_glucose_60"].to_numpy() <= hi)))
    out = {"q90_half_width": q90, "coverage_cal": cov,
           "n_cal": int(len(ca)), "method": "split-conformal (MAPIE-equivalent)"}
    print(json.dumps(out, indent=2))
    return out


def main():
    u = pd.read_parquet("data/processed/unified.parquet")
    u["timestamp"] = pd.to_datetime(u["timestamp"])
    u = add_features(u)
    u = add_static(u)
    m = u["target_glucose_60"].notna() & u["event_60"].notna()
    d = u.loc[m].reset_index(drop=True)
    t2d = sorted(d[d["a1c"] >= 6.5]["patient_id"].unique().tolist())
    print(f"T2D subset (a1c>=6.5): {len(t2d)} patients {t2d}")
    pers = personalization(d, t2d)
    wi = whatif_demo(d)
    cf = conformal(d)
    os.makedirs("results", exist_ok=True)
    with open("results/personalization.json", "w") as f:
        json.dump(pers, f, indent=2)
    with open("results/whatif.json", "w") as f:
        json.dump(wi, f, indent=2)
    with open("results/conformal.json", "w") as f:
        json.dump(cf, f, indent=2)
    print(f"mean personalization lift = {pers['mean_lift']:.2f} mg/dL RMSE "
          f"(n={pers['n']})")


if __name__ == "__main__":
    main()
