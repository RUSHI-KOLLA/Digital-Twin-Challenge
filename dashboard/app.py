"""Phase 5: GlucoTwin doctor dashboard (Streamlit + Plotly, offline).

Single screen: patient header, live CGM trace + forecast band + 180 line +
alert marker, alert card, what-if [walk]/[rice->ragi], triage panel.
Footer badge on every screen: synthetic composite / open data.
Frozen-model inference only (no training post-freeze).
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from lightgbm_v1 import FEATS, add_features, add_static  # noqa: E402

st.set_page_config(page_title="GlucoTwin — T2D Digital Twin", layout="wide")
FOOTER = ("Synthetic composite patient / open data — no open Indian CGM data "
          "exists. Advisory only, clinician-in-the-loop, never automates "
          "insulin delivery.")


@st.cache_resource
def load_all():
    with open("data/synthetic/personas.json") as f:
        personas = json.load(f)["personas"]
    feat = "data/processed/features.parquet"
    try:
        d = pd.read_parquet(feat)
    except FileNotFoundError:
        # fresh clone: build feature cache from unified (no training)
        u = pd.read_parquet("data/processed/unified.parquet")
        u["timestamp"] = pd.to_datetime(u["timestamp"])
        d = add_static(add_features(u))
        d.to_parquet(feat, index=False)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    reg = lgb.Booster(model_file="models/lgbm_reg60.txt")
    clf = lgb.Booster(model_file="models/lgbm_event.txt")
    with open("results/conformal.json") as f:
        q90 = json.load(f)["q90_half_width"]
    with open("results/operating_point.json") as f:
        op = json.load(f)["best"]  # tuned AND rule: band AND risk>=t
    with open("results/demo_window.json") as f:
        demo = json.load(f)
    tri = pd.read_csv("results/triage.csv")
    abl = pd.read_csv("results/ablation.csv")
    return personas, d, reg, clf, q90, demo, tri, abl, op


personas, D, REG, CLF, Q90, DEMO, TRI, ABL, OP = load_all()
names = [p["name"] for p in personas]
sel = st.sidebar.selectbox("Patient", names, index=0)
P = next(p for p in personas if p["name"] == sel)
pid = float(P["trajectory_patient_id"])

st.title("GlucoTwin — rehearsal space for this patient's metabolic future")
st.subheader(f"{P['name']}, {P['persona_age']}, {P['city']}  |  "
             f"HbA1c {P['persona_hba1c']}%  |  "
             f"{' + '.join(P['medications'])}")
st.caption(f"{FOOTER} Trajectory: {P['linked_trajectory']}.")

sub = D[D["patient_id"] == pid].sort_values("timestamp").reset_index(drop=True)
days = sorted(sub["timestamp"].dt.date.astype(str).unique().tolist())
default_day = str(pd.to_datetime(DEMO["timestamp"]).date()) if sel == "Ramesh" else days[-1]
day = st.sidebar.selectbox("Replay day", days,
                           index=days.index(default_day) if default_day in days else 0)
rows = sub[sub["timestamp"].dt.date.astype(str) == day].reset_index(drop=True)
X = rows[FEATS].to_numpy(dtype=float)
rows["pred_60"] = REG.predict(X)
rows["risk"] = CLF.predict(X)
rows["hi_60"] = rows["pred_60"] + Q90
rows["lo_60"] = rows["pred_60"] - Q90
band = rows["hi_60"] > 180
if OP["rule"] == "AND":
    rows["alert"] = band & (rows["risk"] >= OP["t"])
else:
    rows["alert"] = band | (rows["risk"] >= OP["t"])

i = st.sidebar.slider("Replay time (5-min bins)", 0, len(rows) - 1,
                      value=int(rows[rows["timestamp"] <= DEMO["timestamp"]].shape[0]) - 1
                      if sel == "Ramesh" else 0)
play = st.sidebar.button("▶ Play day at 30×")
slot_chart = st.empty()
slot_alert = st.empty()


@st.cache_data
def day_drivers(_pid: float, _day: str):
    rws = D[(D["patient_id"] == _pid) &
            (D["timestamp"].dt.date.astype(str) == _day)].sort_values("timestamp")
    # LightGBM pred_contrib = TreeSHAP values, one vectorized call (no training)
    Xd = rws[FEATS].to_numpy(dtype=float)
    contrib = np.asarray(REG.predict(Xd, pred_contrib=True))[:, :-1]
    plain = {"glucose": "current glucose", "slope_15": "15-min slope",
             "slope_30": "rising slope", "roll_mean_30": "recent average",
             "roll_std_30": "glucose variability", "roll_mean_60": "1-hour average",
             "roll_std_60": "1-hour variability", "carbs_60": "carb load",
             "carbs_120": "2-hour carb load", "activity_60": "low activity",
             "hr_30": "heart rate", "tod_sin": "time of day",
             "tod_cos": "time of day", "dow": "day of week",
             "time_since_meal": "recent meal"}
    out = {}
    band = REG.predict(Xd) + Q90 > 180
    erk = CLF.predict(Xd) >= OP["t"]
    fire = (band & erk) if OP["rule"] == "AND" else (band | erk)
    for (_, row), sv, f in zip(rws.iterrows(), contrib, fire):
        if f:
            top = __import__("numpy").argsort(-abs(sv))[:3]
            out[str(row["timestamp"])] = [plain.get(FEATS[j], FEATS[j]) for j in top]
    return out


DRV = day_drivers(pid, day)


def drivers_for(row, k=3):
    return DRV.get(str(row["timestamp"]), ["recent trend"])


def render(k):
    r = rows.iloc[k]
    hist = rows.iloc[max(0, k - 96):k + 1]  # last 8 h
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["glucose"],
                             mode="lines", name="CGM"))
    fig.add_trace(go.Scatter(x=[r["timestamp"], r["timestamp"] + pd.Timedelta(minutes=60)],
                             y=[r["pred_60"], r["pred_60"]],
                             mode="lines+markers", name="+60 min forecast"))
    fig.add_trace(go.Scatter(
        x=[r["timestamp"], r["timestamp"] + pd.Timedelta(minutes=60),
           r["timestamp"] + pd.Timedelta(minutes=60), r["timestamp"]],
        y=[r["lo_60"], r["lo_60"], r["hi_60"], r["hi_60"]],
        fill="toself", opacity=0.2, name="90% band"))
    fig.add_hline(y=180, line_dash="dash", annotation_text="180 mg/dL")
    if bool(r["alert"]):
        fig.add_vline(x=r["timestamp"], line_color="red",
                      annotation_text="ALERT")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="time", yaxis_title="mg/dL")
    slot_chart.plotly_chart(fig, use_container_width=True)
    if bool(r["alert"]):
        fut = rows.iloc[k + 1:k + 13]
        cross = fut[fut["glucose"] > 180]
        lead = (f"~{int((cross.iloc[0]['timestamp'] - r['timestamp']).total_seconds() / 60)} min"
                if len(cross) else "~60 min")
        slot_alert.warning(f"Spike in {lead}, {r['risk'] * 100:.0f}% — drivers: "
                           f"{', '.join(drivers_for(r))} (model-attributed, not causal)")
    else:
        slot_alert.info(f"No alert — glucose {r['glucose']:.0f}, "
                        f"60-min risk {r['risk'] * 100:.0f}%.")
    return r


if play:
    for k in range(i, len(rows)):
        render(k)
        time.sleep(0.2)
else:
    r = render(i)

st.divider()
st.subheader("What-if simulator (frozen model — carbs is the hero intervention)")
st.caption("Honest limits: raw steps exist for 1/45 patients, so the walk acts "
           "only via activity/HR and moves less (−4 to −11) than the carb swap "
           "(−15 to −18). Wearable fusion here = HR + activity kcal.")
c1, c2 = st.columns(2)
r = rows.iloc[i]
x0 = r[FEATS].to_numpy(dtype=float).reshape(1, -1)
p0 = float(REG.predict(x0)[0])
if c1.button("🚶 15-min walk", key="walk"):
    # walk acts through dense wearable signals; raw steps exist for 1/45
    # patients so steps_* were dropped as a dead feature (see README)
    x = x0.copy()
    x[0, FEATS.index("activity_60")] += 40.0
    x[0, FEATS.index("hr_30")] += 10.0
    st.success(f"Walk: predicted +60 min {p0:.0f} → {float(REG.predict(x)[0]):.0f} mg/dL")
if c2.button("🍚 Swap rice → ragi (−20 g carbs, IFCT 2017)", key="ragi"):
    x = x0.copy()
    x[0, FEATS.index("carbs_60")] = max(0.0, x[0, FEATS.index("carbs_60")] - 20.0)
    x[0, FEATS.index("carbs_120")] = max(0.0, x[0, FEATS.index("carbs_120")] - 20.0)
    st.success(f"Ragi swap: predicted +60 min {p0:.0f} → {float(REG.predict(x)[0]):.0f} mg/dL")

st.divider()
st.subheader("Triage — patients by current 60-min event risk (precomputed replay snapshot)")
st.dataframe(TRI.head(10), use_container_width=True)
with st.expander("Proof: ablation + grouped metrics (95% bootstrap-by-subject CIs)"):
    st.dataframe(ABL, use_container_width=True)
    st.write("Frozen B+wearable-lite RMSE@60 25.66 [23.6, 27.9] vs persistence "
             "29.73 [27.1, 32.5] (no CI overlap) | event AUROC 0.951, AUPRC 0.895 "
             "(glucose-only 0.923/0.865 — not just a threshold rule) | T2D-14: "
             "RMSE 31.17, AUROC 0.954, AUPRC 0.950 | alert AND-rule ≥0.5: "
             "precision 0.907, recall 0.724, 4.25 false alerts/patient-day, "
             "median lead 45 min, miss 7.8% | personalization median lift +1.19 "
             "(+9/−5 of 14 T2D) | conformal 90% band ±37.9, held-out coverage 0.888.")
st.caption(FOOTER)
