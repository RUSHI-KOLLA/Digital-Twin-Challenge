"""Phase 3-DEMO: FastAPI + replay (offline, frozen LightGBM).

POST /predict {patient_id, timestamp} -> forecast + risk + drivers
WS /replay?patient_id=&date=&speed=30 -> streams held-out day at 30x,
  calling the same predict path as data 'arrives'.
GET / -> minimal browser replay viewer (GATE: watch predictions update).
"""
import asyncio
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from lightgbm_v1 import FEATS, add_features, add_static  # noqa: E402

app = FastAPI(title="GlucoTwin replay API")
import yaml
with open("configs/model.yaml") as f:
    CFG = yaml.safe_load(f)
REG = lgb.Booster(model_file=CFG["reg_model"])
CLF = lgb.Booster(model_file=CFG["clf_model"])
with open(CFG["conformal"]) as f:
    Q90 = json.load(f)["q90_half_width"]
with open(CFG["operating_point"]) as f:
    OP = json.load(f)["best"]  # miss-minimising default (see operating_point.json)
with open(CFG["demo_window"]) as f:
    WHATIF = json.load(f)


def is_alert(pred: float, risk: float) -> bool:
    band = pred + Q90 > 180
    if OP["rule"] == "AND":
        return bool(band and risk >= OP["t"])
    return bool(band or risk >= OP["t"])

FEAT_CACHE = "data/processed/features.parquet"
if os.path.exists(FEAT_CACHE):
    D = pd.read_parquet(FEAT_CACHE)
    D["timestamp"] = pd.to_datetime(D["timestamp"])
else:
    _u = pd.read_parquet("data/processed/unified.parquet")
    _u["timestamp"] = pd.to_datetime(_u["timestamp"])
    D = add_static(add_features(_u))
    D.to_parquet(FEAT_CACHE, index=False)
D = D.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
D["ts"] = D["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

PLAIN = {
    "glucose": "current glucose", "slope_15": "15-min slope",
    "slope_30": "rising slope", "roll_mean_30": "recent average",
    "roll_std_30": "glucose variability", "roll_mean_60": "1-hour average",
    "roll_std_60": "1-hour variability", "carbs_60": "carb load",
    "carbs_120": "2-hour carb load", "time_since_meal": "time since meal",
    "steps_30": "recent steps", "steps_60": "activity (steps)",
    "hr_30": "heart rate", "activity_60": "activity level",
    "tod_sin": "time of day", "tod_cos": "time of day", "dow": "day of week",
    "age": "age", "bmi": "BMI", "a1c": "HbA1c",
    "fasting_glu": "fasting glucose",
}


def predict_row(row: pd.Series) -> dict:
    x = row[FEATS].to_numpy(dtype=float).reshape(1, -1)
    pred = float(REG.predict(x)[0])
    risk = float(CLF.predict(x)[0])
    alert = is_alert(pred, risk)
    return {"timestamp": str(row["timestamp"]), "glucose_now": float(row["glucose"]),
            "pred_60": pred, "lo_60": pred - Q90, "hi_60": pred + Q90,
            "risk_event_60": risk, "alert": alert}


def drivers(row: pd.Series, k: int = 3) -> list:
    import shap
    global _EX
    try:
        _EX
    except NameError:
        _EX = shap.TreeExplainer(REG)
    x = row[FEATS].to_numpy(dtype=float).reshape(1, -1)
    sv = _EX.shap_values(x)[0]
    idx = np.argsort(-np.abs(sv))[:k]
    out = []
    for i in idx:
        f = FEATS[i]
        direction = "high" if sv[i] > 0 else "low"
        out.append(f"{PLAIN.get(f, f)} ({direction}, "
                   f"model-attributed driver)")
    return out


class PredictIn(BaseModel):
    patient_id: float
    timestamp: str  # ISO; snapped to nearest 5-min bin


@app.post("/predict")
def predict(inp: PredictIn):
    sub = D[D["patient_id"] == inp.patient_id]
    if not len(sub):
        return {"error": "unknown patient_id"}
    ts = pd.to_datetime(inp.timestamp)
    i = (sub["timestamp"] - ts).abs().argmin()
    row = sub.iloc[int(i)]
    r = predict_row(row)
    r["drivers"] = drivers(row)
    r["patient_id"] = float(row["patient_id"])
    return r


@app.websocket("/replay")
async def replay(ws: WebSocket):
    await ws.accept()
    q = dict(ws.query_params)
    pid = float(q.get("patient_id", WHATIF["patient_id"]))
    day = q.get("date")  # YYYY-MM-DD; default: patient's highest-risk day
    speed = float(q.get("speed", 30))
    sub = D[D["patient_id"] == pid].copy()
    if day is None:
        sub["risk_tmp"] = CLF.predict(sub[FEATS].to_numpy(dtype=float))
        day = sub.sort_values("risk_tmp", ascending=False).iloc[0]["timestamp"].date().isoformat()
    rows = sub[sub["timestamp"].dt.date.astype(str) == day]
    if not len(rows):
        await ws.send_json({"error": f"no data for {day}"})
        await ws.close()
        return
    interval = 300.0 / speed  # 5-min bins at Nx
    for _, row in rows.iterrows():
        r = predict_row(row)
        if r["alert"]:
            r["drivers"] = drivers(row)
        await ws.send_json(r)
        await asyncio.sleep(min(interval, 2.0))
    await ws.close()


@app.get("/", response_class=HTMLResponse)
def index():
    return """<html><body><h3>GlucoTwin replay (offline)</h3>
<p>Streams a held-out day at 30x. Alert when band/prob crosses 180.</p>
<pre id="log"></pre>
<script>
const pid=%s, log=document.getElementById('log');
const ws=new WebSocket(`ws://${location.host}/replay?patient_id=${pid}&speed=30`);
ws.onmessage=e=>{const r=JSON.parse(e.data);
 log.textContent+=`${r.timestamp} g=${r.glucose_now?.toFixed(0)} pred=${r.pred_60?.toFixed(0)} [${r.lo_60?.toFixed(0)}-${r.hi_60?.toFixed(0)}] risk=${r.risk_event_60?.toFixed(2)}${r.alert?' ALERT':''}\n`;
 if(r.drivers) log.textContent+='  drivers: '+r.drivers.join(' | ')+'\n';
 window.scrollTo(0,document.body.scrollHeight);};
</script></body></html>""" % int(WHATIF["patient_id"])
