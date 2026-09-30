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
import sys

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
    arr = np.array(lifts)
    return {"patients": rows,
            "mean_lift": float(np.mean(arr)) if len(arr) else 0.0,
            "median_lift": float(np.median(arr)) if len(arr) else 0.0,
            "q1_lift": float(np.quantile(arr, 0.25)) if len(arr) else 0.0,
            "q3_lift": float(np.quantile(arr, 0.75)) if len(arr) else 0.0,
            "n_improved": int((arr > 0).sum()),
            "n_worse": int((arr < 0).sum()),
            "n": len(rows)}


def whatif_demo(d: pd.DataFrame, q90: float) -> dict:
    """GATE check on the shared hero demo state (same pick as the dashboard):
    -20g carbs (monotone +1, always sane) and +15-min walk via dense
    wearable signals (activity/HR; steps_* dropped as dead)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
    from prep_demo import pick_demo_state  # noqa: E402
    reg = lgb.Booster(model_file="models/lgbm_reg60.txt")
    h39 = d[d["patient_id"] == 39]
    s, p0, pw, pc, lead = pick_demo_state(h39, reg, q90)
    out = {"patient_id": 39.0, "timestamp": str(s["timestamp"]),
           "glucose_now": float(s["glucose"]),
           "pred_base": p0, "pred_fewer_carbs": pc, "pred_walk": pw,
           "walk_lowers": bool(pw < p0), "carbs_lower": bool(pc <= p0),
           "lead_min_to_180": lead}
    print(json.dumps(out, indent=2))
    assert out["walk_lowers"], "GATE FAIL: walk did not lower predicted peak"
    return out


def conformal(d: pd.DataFrame) -> dict:
    # split-conformal, 3-way by patient: 60% fit / 20% calibrate / 20% test.
    # Coverage on the held-out TEST set (not just calibration) is reported.
    pids = np.array(sorted(d["patient_id"].unique()))
    rng = np.random.RandomState(0)
    rng.shuffle(pids)
    n = len(pids)
    tr_p, ca_p, te_p = set(pids[:int(n * 0.6)]), \
        set(pids[int(n * 0.6):int(n * 0.8)]), set(pids[int(n * 0.8):])
    tr = d[d["patient_id"].isin(tr_p)]
    ca = d[d["patient_id"].isin(ca_p)]
    te = d[d["patient_id"].isin(te_p)]
    r = lgb.LGBMRegressor(n_estimators=N_ROUNDS, learning_rate=0.05,
                          num_leaves=31, min_child_samples=50,
                          monotone_constraints=MONO, verbose=-1)
    r.fit(tr[FEATS].to_numpy(), tr["target_glucose_60"].to_numpy())
    resid = np.abs(ca["target_glucose_60"].to_numpy() - r.predict(ca[FEATS].to_numpy()))

    def coverage(df, q):
        p = r.predict(df[FEATS].to_numpy())
        y = df["target_glucose_60"].to_numpy()
        return float(np.mean((y >= p - q) & (y <= p + q)))

    q90 = float(np.quantile(resid, 0.90))
    q50 = float(np.quantile(resid, 0.50))
    out = {"q90_half_width": q90, "q50_half_width": q50,
           "coverage_cal_90": coverage(ca, q90),
           "coverage_test_90": coverage(te, q90),
           "coverage_test_50": coverage(te, q50),
           "n_cal": int(len(ca)), "n_test": int(len(te)),
           "method": "split-conformal, 3-way by patient (MAPIE-equivalent)"}
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
    cf = conformal(d)
    wi = whatif_demo(d, cf["q90_half_width"])
    os.makedirs("results", exist_ok=True)
    with open("results/personalization.json", "w") as f:
        json.dump(pers, f, indent=2)
    with open("results/whatif.json", "w") as f:
        json.dump(wi, f, indent=2)
    with open("results/conformal.json", "w") as f:
        json.dump(cf, f, indent=2)
    print(f"personalization lift mean={pers['mean_lift']:.2f} "
          f"median={pers['median_lift']:.2f} "
          f"(+{pers['n_improved']}/-{pers['n_worse']}, n={pers['n']})")


if __name__ == "__main__":
    main()
