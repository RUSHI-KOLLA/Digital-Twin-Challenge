# GlucoTwin — T2D Digital Twin (PoC)

> We don't predict diabetes. We predict what is about to happen to this patient.

Doctor-facing twin that warns ~60 min in advance when glucose will exceed
180 mg/dL after a meal — and simulates what would prevent it. Built for the
Happiest Health Digital Twin Challenge 2026. `AGENTS.md` is the single source
of truth.

## One-command run (offline, frozen model)

```
docker compose up
```
Dashboard → http://localhost:8501, replay API → http://localhost:8000.
Hero replay: Ramesh, 2025-02-17 — alert at 09:20 (glucose 113, risk 80%),
spike 55 min later. Demo script: `docs/demo.md`.

## Metrics (patient-wise GroupKFold, n=45, T2D subset n=14)

| Model | RMSE@60 | Event AUROC | Event AUPRC |
|---|---|---|---|
| Persistence | 29.73 | — | — |
| LightGBM (frozen) | 25.83 | 0.949 | 0.894 |

Ablation RMSE@60: CGM 26.83 → +meals 25.49 → +wearable 25.69 → +EHR 25.83.
Personalization lift +2.97 RMSE (14 T2D). Conformal 90% band ±42.0, coverage 0.90.

## Reproduce (local)

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
aws s3 sync --no-sign-request s3://physionet-open/cgmacros/1.0.0/ data/raw/cgmacros/
python src/preprocessing/build_unified.py   # Phase 1
python src/models/baselines.py              # Phase 2a (THE number: RMSE@60 29.73)
python src/models/lightgbm_v1.py            # Phase 2b (frozen; do not retrain)
```

## Responsible AI

Decision support, not diagnosis. No open Indian CGM data exists — patients are
labeled synthetic composites (Synthea history + real open trajectory). Small T2D
sample, honest intervals. DPDP Act 2023 / CDSCO SaMD acknowledged; prospective
validation is future work. Never automates insulin delivery.
