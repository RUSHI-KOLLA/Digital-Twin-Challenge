# GlucoTwin — T2D Digital Twin (PoC)

> We don't predict diabetes. We predict what is about to happen to this patient.

Doctor-facing twin that warns ~60 min in advance when glucose will exceed
180 mg/dL after a meal — and simulates what would prevent it. Built for the
Happiest Health Digital Twin Challenge 2026. `AGENTS.md` is the single source
of truth.

![Architecture](docs/architecture.png)

## One-command run (frozen model)

```
docker compose up
```
Dashboard → http://localhost:8501, replay API → http://localhost:8000.
On a fresh clone the entrypoint downloads open CGMacros (~627 MB, no login)
and rebuilds processed data automatically. Hero replay: Ramesh, 2025-02-16 —
alert at 12:05 (glucose 153, risk 82%), spike 30 min later. Demo script:
`docs/demo.md`.

## Metrics (patient-wise GroupKFold, n=45; 95% bootstrap-by-subject CIs)

| Model | RMSE@60 [95% CI] | Event AUROC | Event AUPRC |
|---|---|---|---|
| Persistence | 29.73 [27.1, 32.5] | — | — |
| Glucose-only (event) | — | 0.923 [0.898, 0.943] | 0.865 |
| LightGBM frozen (B+wearable-lite) | **25.66 [23.6, 27.9]** | **0.951 [0.937, 0.962]** | **0.895** |
| T2D subset (n=14) | 31.17 [27.5, 34.6] | 0.954 | 0.950 |

Regression also beats persistence at +30 (17.54 vs 20.18) and +120 (32.99 vs 40.20).
Ablation RMSE@60: A CGM 26.83 → B +meals 25.46 → C +wearable 25.67 → D +EHR
25.74 — all CIs overlap B, so fusion adds no measurable gain at this n
(honest small-n read, not "fusion hurts"). Frozen = B + HR/activity (dead
steps_* and flat static dropped; CIs overlap B).

Alert operating point (tuned, AND rule: band-cross AND risk≥0.5): precision
0.907, recall 0.724, 4.25 false alerts/patient-day, median lead 45 min over
923 crossings, miss rate 7.8%. Personalization: median lift +1.19 RMSE
(+9/−5 of 14 T2D — negatives shown). Conformal 90% band ±37.9, held-out
patient coverage 0.888 (50% band ±10.4, coverage 0.458).

## Honest limits (say it ourselves)

- Raw steps exist for 1/45 patients → steps features were dead weight and
  dropped. Wearable fusion here = heart rate + activity kcal. The walk
  what-if moves less (−4 to −11) than the carb swap (−15 to −18):
  **carbs is the hero intervention**.
- No open Indian CGM data exists — patients are labeled synthetic composites
  (Synthea history + real open trajectory), badged on every screen.

## Reproduce (local)

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
aws s3 sync --no-sign-request s3://physionet-open/cgmacros/1.0.0/ data/raw/cgmacros/
python -c "import zipfile; zipfile.ZipFile('data/raw/cgmacros/CGMacros_dateshifted365.zip').extractall('data/raw/cgmacros/unzipped')"
python src/preprocessing/build_unified.py   # Phase 1
python src/models/baselines.py              # Phase 2a (THE number: RMSE@60 29.73)
python src/models/lightgbm_v1.py            # refreeze (frozen v2; do not retrain post-freeze)
python -m unittest discover -s tests        # smoke tests
```

## Responsible AI

Decision support, not diagnosis. Small T2D sample, honest intervals. DPDP Act
2023 / CDSCO SaMD acknowledged; prospective validation is future work. Never
automates insulin delivery.
