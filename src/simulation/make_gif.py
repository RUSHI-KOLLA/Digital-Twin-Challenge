"""Generate docs/demo.gif — animated hero demo (CGM + band + alert + what-if).

Frozen-model inference only. Run: python src/simulation/make_gif.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from lightgbm_v1 import FEATS, add_features, add_static  # noqa: E402
from prep_demo import apply_walk, apply_ragi  # noqa: E402

import lightgbm as lgb  # noqa: E402

with open("configs/model.yaml") as f:
    import yaml
    CFG = yaml.safe_load(f)
REG = lgb.Booster(model_file=CFG["reg_model"])
CLF = lgb.Booster(model_file=CFG["clf_model"])
Q90 = json.load(open(CFG["conformal"]))["q90_half_width"]
DEMO = json.load(open(CFG["demo_window"]))


def main():
    u = pd.read_parquet("data/processed/unified.parquet")
    u["timestamp"] = pd.to_datetime(u["timestamp"])
    d = add_static(add_features(u))
    m = d["target_glucose_60"].notna() & d["event_60"].notna()
    d = d.loc[m].reset_index(drop=True)
    h = d[d["patient_id"] == 39].sort_values("timestamp").reset_index(drop=True)
    X = h[FEATS].to_numpy(dtype=float)
    h["pred"] = REG.predict(X)
    h["risk"] = CLF.predict(X)
    h["hi"] = h["pred"] + Q90
    h["lo"] = h["pred"] - Q90
    h["alert"] = (h["hi"] > 180) & (h["risk"] >= 0.7)

    t0 = pd.to_datetime(DEMO["timestamp"])
    i0 = int(h.index[h["timestamp"] == t0][0])
    # window: 2h before demo to 2h after
    lo_i = max(0, i0 - 24)
    hi_i = min(len(h) - 1, i0 + 24)
    seg = h.iloc[lo_i:hi_i + 1].reset_index(drop=True)
    j0 = i0 - lo_i  # demo position within seg

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 2])
    fig.suptitle("GlucoTwin — Ramesh, 52, HbA1c 8.2% (synthetic composite, open data)",
                 fontsize=12)

    def draw(frame):
        ax.clear()
        ax2.clear()
        k = min(frame, len(seg) - 1)
        cur = seg.iloc[:k + 1]
        ax.plot(cur["timestamp"], cur["glucose"], color="#1a73e8", lw=1.5, label="CGM")
        r = seg.iloc[k]
        # forecast band from current point
        ax.plot([r["timestamp"], r["timestamp"] + pd.Timedelta(minutes=60)],
                [r["pred"], r["pred"]], color="#ea4335", lw=2, label="+60 forecast")
        ax.fill_between([r["timestamp"], r["timestamp"] + pd.Timedelta(minutes=60)],
                        [r["lo"], r["lo"]], [r["hi"], r["hi"]],
                        color="#ea4335", alpha=0.2, label="90% band")
        ax.axhline(180, color="orange", ls="--", lw=1, label="180 mg/dL")
        if bool(r["alert"]):
            ax.axvline(r["timestamp"], color="red", lw=1.5)
            ax.text(r["timestamp"], ax.get_ylim()[1] * 0.95, "ALERT",
                    color="red", fontsize=9, ha="center")
        ax.set_ylabel("mg/dL")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{r['timestamp']}  glucose {r['glucose']:.0f}  "
                     f"risk {r['risk']*100:.0f}%", fontsize=10)

        # what-if panel at the demo state
        if k >= j0:
            x0 = r[FEATS].to_numpy(dtype=float).reshape(1, -1)
            p0 = float(REG.predict(x0)[0])
            pw = float(REG.predict(apply_walk(x0))[0])
            pc = float(REG.predict(apply_ragi(x0))[0])
            ax2.bar(["base", "walk", "ragi"], [p0, pw, pc],
                    color=["#ea4335", "#34a853", "#fbbc04"])
            ax2.axhline(180, color="orange", ls="--", lw=1)
            for i, v in enumerate([p0, pw, pc]):
                ax2.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=9)
            ax2.set_ylabel("pred +60 (mg/dL)")
            ax2.set_title("What-if: both levers lower the peak", fontsize=10)
        else:
            ax2.text(0.5, 0.5, "what-if appears at the alert…",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=10)
            ax2.set_xticks([])
            ax2.set_yticks([])
        return ax, ax2

    n_frames = len(seg)
    ani = FuncAnimation(fig, draw, frames=n_frames, interval=120, blit=False)
    ani.save("docs/demo.gif", writer=PillowWriter(fps=8))
    print(f"saved docs/demo.gif ({n_frames} frames)")


if __name__ == "__main__":
    main()
