"""Task C: judge-facing evaluation artifacts (no training, v2 OOF only).

C1. Isotonic calibration of the event head: fit on 50% of patients (seed 1),
    evaluated on held-out patients. Saves results/calibration.json
    (Brier before/after, slope/intercept, precision at the op point
    before/after) + results/reliability.png.
C2. Clarke Error Grid for the +60 min forecast (v2 OOF y60/p60):
    saves results/error_grid.json + results/clarke.png.
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")

SEED = 1
Q90 = json.load(open("results/conformal.json"))["q90_half_width"]
OP_T = json.load(open("results/operating_point.json"))["best"]["t"]


def clarke_zone(x, y):
    """Standard Clarke EGA zones (x=reference, y=prediction, mg/dL).

    Boundaries: 20% lines (A); hypo box; x=70 / x=240 verticals with
    y=70/180 bands (D vs E); y=x+110 upper overcorrection line (C);
    lower C/B line through (130,90)-(180,60). Points exactly on a
    boundary are measure-zero; order: E, D, A, C, else B.
    """
    # E: dangerous opposite treatment
    if (x < 70 and y > 180) or (x > 240 and y < 70):
        return "E"
    # D: dangerous failure to detect hypo/hyper
    if (x < 70 and 70 <= y <= 180) or (x > 240 and 70 <= y <= 180):
        return "D"
    # A: clinically accurate
    if (x >= 70 and 0.8 * x <= y <= 1.2 * x) or (x < 70 and y < 70):
        return "A"
    # C: overcorrection (unnecessary treatment)
    if 70 <= x <= 290 and y > x + 110:
        return "C"
    if x >= 130 and y < 90 - 0.6 * (x - 130):
        return "C"
    return "B"


def _check_battery():
    battery = [(100, 100, "A"), (100, 115, "A"), (60, 60, "A"), (50, 55, "A"),
               (100, 140, "B"), (100, 70, "B"), (150, 100, "B"), (200, 100, "B"),
               (60, 100, "D"), (60, 200, "E"), (250, 250, "A"),
               (250, 150, "D"), (250, 50, "E"), (300, 150, "D"),
               (150, 300, "C"), (120, 250, "C")]
    for x, y, want in battery:
        got = clarke_zone(x, y)
        assert got == want, f"clarke({x},{y})={got}, want {want}"
    print(f"clarke battery: {len(battery)}/{len(battery)} known-answer points pass")


def calibration():
    oof = pd.read_parquet("results/oof_frozen.parquet")
    pids = np.array(sorted(oof["patient_id"].unique()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(pids)
    fit_p = set(pids[:len(pids) // 2])
    fit = oof[oof["patient_id"].isin(fit_p)]
    ev = oof[~oof["patient_id"].isin(fit_p)].reset_index(drop=True)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(fit["escore"].to_numpy(), fit["event"].to_numpy())
    p_raw = ev["escore"].to_numpy()
    p_cal = iso.predict(p_raw)
    y = ev["event"].to_numpy()
    brier_before = float(np.mean((p_raw - y) ** 2))
    brier_after = float(np.mean((p_cal - y) ** 2))
    eps = 1e-6
    logit = np.log(np.clip(p_cal, eps, 1 - eps) / (1 - np.clip(p_cal, eps, 1 - eps)))
    lr = LogisticRegression().fit(logit.reshape(-1, 1), y)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    # precision at the frozen operating point (AND t, band from v2 Q90)
    band = (ev["p60"].to_numpy() + Q90 > 180).astype(int)
    out = {}
    for name, s in [("before", p_raw), ("after", p_cal)]:
        a = np.minimum(band, (s >= OP_T).astype(int))
        tp = int(((a == 1) & (y == 1)).sum())
        fp = int(((a == 1) & (y == 0)).sum())
        out[name] = {"precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                     "n_alerts": int(a.sum())}
    # reliability curve on eval half (calibrated)
    import matplotlib.pyplot as plt
    qs = np.quantile(p_cal, np.linspace(0, 1, 11))
    cx, cy = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (p_cal >= lo) & (p_cal <= hi) if hi > lo else (p_cal == lo)
        if m.sum():
            cx.append(float(p_cal[m].mean()))
            cy.append(float(y[m].mean()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(cx, cy, "o-", label="isotonic (held-out patients)")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction positive")
    ax.set_title("Reliability — event head (held-out patients)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/reliability.png", dpi=150)
    plt.close()
    res = {"brier_before": brier_before, "brier_after": brier_after,
           "slope": slope, "intercept": intercept,
           "precision_before": out["before"]["precision"],
           "precision_after": out["after"]["precision"],
           "n_fit_patients": len(fit_p), "n_eval_patients": len(pids) - len(fit_p),
           "method": "isotonic fit on 50% patients, evaluated on held-out 50%"}
    json.dump(res, open("results/calibration.json", "w"), indent=2)
    print("calibration:", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                      for k, v in res.items()}, indent=1))
    return res


def error_grid():
    oof = pd.read_parquet("results/oof_frozen.parquet")
    _check_battery()
    x = oof["y60"].to_numpy()
    y = oof["p60"].to_numpy()
    zones = np.array([clarke_zone(a, b) for a, b in zip(x, y)])
    counts = {z: int((zones == z).sum()) for z in "ABCDE"}
    pct = {z: float(counts[z] / len(zones) * 100) for z in "ABCDE"}
    json.dump({"counts": counts,
               "percent": {z: round(pct[z], 2) for z in "ABCDE"},
               "n": int(len(zones)),
               "method": "Clarke EGA on v2 OOF +60min forecast (see clarke_zone)"},
              open("results/error_grid.json", "w"), indent=2)
    print("clarke %:", {z: round(pct[z], 2) for z in "ABCDE"})
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(0)
    take = np.zeros(len(x), dtype=bool)
    take[rng.choice(len(x), min(5000, len(x)), replace=False)] = True
    take |= np.isin(zones, ["C", "D", "E"])  # show every tail point
    col = {"A": "#34a853", "B": "#1a73e8", "C": "#fbbc04",
           "D": "#ea4335", "E": "#7b1fa2"}
    fig, ax = plt.subplots(figsize=(7, 7))
    gx = np.linspace(0, 400, 200)
    ax.plot(gx, 1.2 * gx, "k-", lw=0.8)
    ax.plot(gx, 0.8 * gx, "k-", lw=0.8)
    ax.plot([70, 290], [180, 400], "k-", lw=0.8)
    ax.plot([130, 400], [90, 90 - 0.6 * (400 - 130)], "k-", lw=0.8)
    ax.axvline(70, ymin=70 / 400, ymax=180 / 400, color="k", lw=0.8)
    ax.axvline(240, ymin=70 / 400, ymax=180 / 400, color="k", lw=0.8)
    ax.axhline(180, xmin=0, xmax=70 / 400, color="k", lw=0.8)
    ax.axhline(70, xmin=240 / 400, xmax=1, color="k", lw=0.8)
    for z in "ABCDE":
        m = take & (zones == z)
        ax.scatter(x[m], y[m], s=3, c=col[z], label=f"{z} {pct[z]:.1f}%", alpha=0.6)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("reference glucose (mg/dL)")
    ax.set_ylabel("predicted +60 min (mg/dL)")
    ax.set_title("Clarke Error Grid — v2 +60min forecast (GroupKFold OOF)")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig("results/clarke.png", dpi=150)
    plt.close()
    print("saved results/error_grid.json + results/clarke.png")
    return pct


if __name__ == "__main__":
    calibration()
    error_grid()
