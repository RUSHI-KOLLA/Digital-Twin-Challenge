"""Bootstrap-by-subject CIs + glucose event baseline + alert operating point.

Reads OOF parquets (frozen + ablation variants). All resampling by patient.
- metrics_ci.json: 95% CIs (1000 bootstraps) for RMSE@60, AUROC, AUPRC.
- Glucose-only event baseline (score=current glucose; rule glucose>150).
- Operating point sweep over alert rules OR(t)/AND(t) with
  precision/recall/F1/false-alerts/median-lead/miss-rate.
No training. Saves results/metrics_ci.json + results/operating_point.json.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

N_BOOT = 1000
SEED = 0
Q90 = json.load(open("results/conformal.json"))["q90_half_width"]


def load_oof():
    out = {"frozen": pd.read_parquet("results/oof_frozen.parquet")}
    for v in ["A_cgm", "B_meals", "C_wearable", "D_ehr"]:
        out[v] = pd.read_parquet(f"results/oof_ablation_{v}.parquet")
    u = pd.read_parquet("data/processed/unified.parquet")
    u["timestamp"] = pd.to_datetime(u["timestamp"])
    for k, df in out.items():
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        m = df.merge(u[["patient_id", "timestamp", "glucose"]],
                     on=["patient_id", "timestamp"], how="left", validate="many_to_one")
        assert m["glucose"].notna().all(), f"glucose merge failed for {k}"
        out[k] = m
    return out


def boot_metric(pids, df, fn, rng):
    vals = []
    for _ in range(N_BOOT):
        samp = rng.choice(pids, size=len(pids), replace=True)
        sub = pd.concat([df[df["patient_id"] == p] for p in samp])
        vals.append(fn(sub))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def rmse60(sub):
    return float(np.sqrt(np.mean((sub["y60"] - sub["p60"]) ** 2)))


def auroc(sub):
    return float(roc_auc_score(sub["event"], sub["escore"]))


def auprc(sub):
    return float(average_precision_score(sub["event"], sub["escore"]))


def pr_at(sub, alert):
    a = alert(sub)
    y = sub["event"].to_numpy()
    tp = int(((a == 1) & (y == 1)).sum())
    fp = int(((a == 1) & (y == 0)).sum())
    fn = int(((a == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def lead_stats(df, alert_col):
    """Median lead (min) before upward 180-crossings; miss rate."""
    leads = []
    missed = 0
    total = 0
    for _, g in df.sort_values("timestamp").groupby("patient_id"):
        g = g.reset_index(drop=True)
        cross = g[(g["glucose"].shift(1) <= 180) & (g["glucose"] > 180)].index
        for ci in cross:
            total += 1
            lo = max(0, ci - 24)  # prior 120 min
            pre = g.iloc[lo:ci]
            hits = pre[pre[alert_col] == 1]
            if len(hits):
                # lead = FIRST alert before the crossing (not the last)
                leads.append((g.loc[ci, "timestamp"] - hits.iloc[0]["timestamp"])
                             .total_seconds() / 60.0)
            else:
                missed += 1
    return (float(np.median(leads)) if leads else 0.0,
            float(missed / total) if total else 1.0, total)


def main():
    oof = load_oof()
    fr = oof["frozen"]
    pids = np.array(sorted(fr["patient_id"].unique()))
    rng = np.random.RandomState(SEED)
    ci = {}
    for k, df in oof.items():
        ci[k] = {"rmse_60_ci": boot_metric(pids, df, rmse60, rng),
                 "auroc_ci": boot_metric(pids, df, auroc, rng),
                 "auprc_ci": boot_metric(pids, df, auprc, rng),
                 "rmse_60": rmse60(df), "auroc": auroc(df), "auprc": auprc(df)}
    # persistence OOF (analytic) + glucose event baseline
    fr2 = fr.copy()
    fr2["p60"] = fr2["glucose"]
    ci["persistence"] = {"rmse_60_ci": boot_metric(pids, fr2, rmse60, rng),
                         "rmse_60": rmse60(fr2)}
    ci["glucose_baseline"] = {
        "auroc": float(roc_auc_score(fr["event"], fr["glucose"])),
        "auprc": float(average_precision_score(fr["event"], fr["glucose"])),
        "auroc_ci": boot_metric(pids, fr, lambda s: float(
            roc_auc_score(s["event"], s["glucose"])), rng),
        "auprc_ci": boot_metric(pids, fr, lambda s: float(
            average_precision_score(s["event"], s["glucose"])), rng)}
    pg, rg, fg = pr_at(fr, lambda s: (s["glucose"] > 150).astype(int).to_numpy())
    ci["glucose_baseline"].update(
        {"rule": "glucose>150", "precision": pg, "recall": rg, "f1": fg})
    # T2D-split CIs for the frozen model
    t2d = fr[fr["t2d"]]
    tpids = np.array(sorted(t2d["patient_id"].unique()))
    ci["frozen_t2d"] = {"rmse_60_ci": boot_metric(tpids, t2d, rmse60, rng),
                        "auroc_ci": boot_metric(tpids, t2d, auroc, rng),
                        "auprc_ci": boot_metric(tpids, t2d, auprc, rng),
                        "rmse_60": rmse60(t2d), "auroc": auroc(t2d),
                        "auprc": auprc(t2d), "n_patients": int(len(tpids))}
    with open("results/metrics_ci.json", "w") as f:
        json.dump(ci, f, indent=2)
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in ci.items()},
                     indent=2))

    # operating-point sweep on frozen OOF
    band = (fr["p60"] + Q90 > 180).astype(int).to_numpy()
    best, rows = None, []
    for rule in ["OR", "AND"]:
        for t in np.arange(0.10, 0.91, 0.05):
            t = round(float(t), 2)
            a = np.maximum(band, (fr["escore"] >= t).astype(int).to_numpy()) \
                if rule == "OR" else \
                np.minimum(band, (fr["escore"] >= t).astype(int).to_numpy())
            prec, rec, f1 = pr_at(fr, lambda s, a=a: a)
            tmp = fr.copy()
            tmp["alert"] = a
            med_lead, miss, n_cross = lead_stats(tmp, "alert")
            fa = float(((a == 1) & (fr["event"].to_numpy() == 0)).sum()
                       / (len(fr) / 288.0))
            rows.append({"rule": rule, "t": t, "precision": round(prec, 3),
                         "recall": round(rec, 3), "f1": round(f1, 3),
                         "false_alerts_pd": round(fa, 2),
                         "median_lead_min": round(med_lead, 1),
                         "miss_rate": round(miss, 3), "n_crossings": n_cross})
    fmax = max(r["f1"] for r in rows)
    # max F1; ties (within .001) broken toward fewer false alerts
    cands = [r for r in rows if fmax - r["f1"] <= 0.001]
    best = sorted(cands, key=lambda r: r["false_alerts_pd"])[0]
    op = {"best": best, "all": sorted(rows, key=lambda r: (-r["f1"], r["false_alerts_pd"])),
          "note": "max-F1 operating point, ties -> fewer false alerts; "
                  "alert evaluated per 5-min bin"}
    with open("results/operating_point.json", "w") as f:
        json.dump(op, f, indent=2)
    print("BEST:", json.dumps(best, indent=2))
    print("saved results/metrics_ci.json + results/operating_point.json")


if __name__ == "__main__":
    main()
