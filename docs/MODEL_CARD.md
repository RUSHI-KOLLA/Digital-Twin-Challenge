# Model Card — GlucoTwin Frozen v2 (LightGBM B+wearable-lite)

## Intended Use

Doctor-facing decision support for Type 2 Diabetes (T2D) management. Warns ~60 minutes in advance when a patient's glucose is forecast to exceed 180 mg/dL after a meal, and simulates "what-if" scenarios (15-min walk, 20g fewer carbs) to show how the peak could be lowered. **Advisory only — does not dose insulin, does not close the loop, clinician-in-the-loop required.**

Target population: Adults with T2D on metformin ± sulfonylurea, Indian phenotype framing (earlier onset, high post-meal excursions on rice/carb-heavy diets). Training data: CGMacros (45 participants, 14 T2D, 16 prediabetes, 15 healthy; Dexcom G6 + Fitbit HR + meal macros + labs). No open Indian CGM data exists — patients in the demo are synthetic composites (Synthea history + real open trajectory), badged on every screen.

## Out-of-Scope Use

- **Not for diagnosis** of diabetes or any other condition.
- **Not for insulin dosing** or automated insulin delivery.
- **Not for Type 1 Diabetes** (different physiology, not trained on T1D data).
- **Not for pediatric** populations.
- **Not for critically ill / ICU** patients (different glucose dynamics).
- **Not a substitute** for clinical judgment or standard of care (HbA1c, SMBG, CGM review).

## Population

- **Training cohort**: CGMacros (PhysioNet), 45 participants (14 T2D by HbA1c ≥ 6.5%, 16 prediabetes, 15 healthy), Hispanic/Latino/White/Black/African American, US-based. Not Indian.
- **External test cohort**: ShanghaiT2DM (Zhao et al., Sci Data 2023, CC-BY), 100 T2D patients, 15-min CGM upsampled to 5-min, Chinese population.
- **Demo personas**: 3 synthetic composites (Synthea diabetes module + CGMacros trajectories), Indian names, metformin + sulfonylurea, HbA1c 7.5–9%.

## Training Data + Licences

| Dataset | Source | Licence | N | Modality |
|---|---|---|---|---|
| CGMacros | PhysioNet | CC-BY 4.0 | 45 | CGM (5-min), HR, steps, meal macros, labs |
| ShanghaiT2DM (test only) | Figshare / Sci Data | CC-BY 4.0 | 100 | CGM (15-min) |
| Synthea (personas only) | MITRE | Apache-2.0 | — | Synthetic EHR (FHIR) |
| IFCT 2017 (meal DB) | ICMR-NIN | CC-BY-NC | — | Indian food composition |

No proprietary or restricted data used. All training data is openly licensed.

## Metrics (Patient-wise GroupKFold, 95% Bootstrap-by-Subject CIs)

### In-cohort (CGMacros, n=45)

| Metric | Value | 95% CI |
|---|---|---|
| RMSE@60 | 25.66 | [23.6, 27.9] |
| MAE@60 | 18.21 | [16.8, 19.7] |
| RMSE@30 | 17.54 | [16.1, 19.1] |
| RMSE@120 | 32.99 | [30.2, 36.1] |
| Event AUROC | 0.951 | [0.937, 0.962] |
| Event AUPRC | 0.895 | [0.868, 0.918] |
| Persistence RMSE@60 | 29.73 | [27.1, 32.5] |

**T2D subset (n=14)**: RMSE@60 = 31.17 [27.5, 34.4], AUROC = 0.954, AUPRC = 0.950. T2D persistence RMSE@60 = 36.34 [32.9, 39.9].

### Cross-cohort (ShanghaiT2DM, n=100, zero refit)

| Metric | Value | 95% CI |
|---|---|---|
| RMSE@60 | 46.44 | [43.3, 49.5] |
| Event AUROC | 0.853 | [0.830, 0.872] |
| Event AUPRC | 0.824 | [0.795, 0.850] |
| External persistence RMSE@60 | 52.31 | [48.1, 56.7] |

**Drop vs CGMacros**: +20.78 RMSE, −0.098 AUROC. **Missing-data control** (CGMacros with meal/HR/activity zeroed): RMSE@60 = 25.56 [23.3, 28.0], AUROC = 0.958 [0.943, 0.970] — CIs overlap full CGMacros. **Conclusion: drop is population shift, not missing inputs.**

### Alert Operating Point (miss-minimising default)

| Metric | Value |
|---|---|
| Rule | AND (band-cross AND risk ≥ 0.5) |
| Precision | 0.907 |
| Recall | 0.724 |
| False alerts / patient-day | 4.25 |
| Median lead time | 25 min |
| Miss rate | 0.102 |

**Precision-first alternative (t=0.7)**: precision 0.964, recall 0.669, 1.42 false alerts/day, lead 15 min, miss rate 0.283.

### Personalization (fine-tune on first half, test on second half)

- Median lift: +1.19 RMSE (9/14 T2D improved, 5/14 worse — negatives shown)
- Conformal 90% band: ±37.9 mg/dL, held-out coverage 0.888
- Conformal 50% band: ±10.4 mg/dL, held-out coverage 0.458

### Calibration (isotonic, fit on 22 patients, held out 23)

- Brier: 0.045 → 0.046 (no material gain)
- Precision at default op point: 0.868 → 0.877
- Calibration NOT wired into serving; head ships raw (slope 1.07, intercept −0.39)

### Clarke Error Grid (+60 forecast)

- Zone A: 80.0%, Zone B: 18.4% (A+B = 98.5%), Zone D: 1.5%, Zone E: 0.0%

## Known Limitations

1. **Small n (45 total, 14 T2D)**: All CIs are wide; bootstrap-by-subject helps but cannot overcome fundamental sample size limits. v3 feature experiment improved all metrics directionally but CIs overlapped — null result kept honest.

2. **No open Indian CGM data**: Training cohort is US-based (CGMacros). Indian phenotype framing is via Synthea priors (ICMR-INDIAB age at onset, BMI) and IFCT 2017 meal database. Demo patients are labeled "synthetic composite" everywhere.

3. **Cross-cohort drop**: RMSE@60 degrades from 25.66 (CGMacros) to 46.44 (ShanghaiT2DM). The drop is almost entirely population shift (device: Dexcom G6 vs study device; sampling: 5-min vs 15-min upsampled; demographics: US vs Chinese; T2D prevalence: 31% vs 100%), not missing inputs (CGMacros CGM-only RMSE 25.56, CIs overlap).

4. **Wearable fusion = HR + activity kcal only**: Raw steps exist for 1/45 patients → steps features dropped. No HRV, EDA, temperature, or accelerometry features. BIG IDEAs Lab (Empatica E4) not integrated.

5. **Counterfactual not validated per-event**: What-if simulator correlation with observed intervention effects r = 0.145 (p < 0.05), RMSE 44.8 mg/dL. Averages agree directionally; per-event effects not recoverable at this n with observational matching (HR proxy for activity).

6. **No hypoglycaemia head**: <70 mg/dL events are rare in this cohort — not a primary claim. Secondary safety flag only if implemented.

7. **Prospective validation pending**: All metrics are retrospective on open data. Clinical utility unproven.

## Ethical Considerations

- **DPDP Act 2023 (India)**: Personal health data processing requires consent, purpose limitation, data minimization. This PoC uses only open, de-identified data (CGMacros, ShanghaiT2DM, Synthea). No real patient PII.
- **CDSCO SaMD**: This is decision support software (risk class likely B or C). Not submitted for registration. Would require clinical validation, QMS, risk management (ISO 14971), usability engineering (IEC 62366) for regulatory clearance.
- **Decision support, not diagnosis**: Explicitly stated on every screen and in all documentation. Never automates insulin delivery.
- **Bias / equity**: Subgroup analysis (age band, BMI band, sex, HbA1c band) shows performance variation — weakest in high-BMI and high-HbA1c bands. Reported in `results/subgroups.csv` (if generated).
- **Transparency**: All metrics reported with CIs. Null results documented (v3 not promoted, counterfactual per-event not validated). Frozen model hashes pinned in `models/freeze.json` and tested.

## Reproduction Commands

```bash
# Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data (CGMacros from PhysioNet)
aws s3 sync --no-sign-request s3://physionet-open/cgmacros/1.0.0/ data/raw/cgmacros/
python -c "import zipfile; zipfile.ZipFile('data/raw/cgmacros/CGMacros_dateshifted365.zip').extractall('data/raw/cgmacros/unzipped')"

# Phase 1: Build unified parquet
python src/preprocessing/build_unified.py

# Phase 2a: Baselines (THE number: persistence RMSE@60 = 29.73)
python src/models/baselines.py

# Phase 2b: Train frozen v2 LightGBM (do not retrain post-freeze)
python src/models/lightgbm_v1.py

# Cross-cohort evaluation (requires ShanghaiT2DM download)
# python src/models/cross_cohort.py

# CGMacros CGM-only evaluation
python src/models/eval_cgmacros_cgm_only.py

# Counterfactual validation
python src/models/counterfactual_validation.py

# Smoke tests
python -m unittest discover -s tests -v

# Run demo (offline, frozen model)
docker compose up
```

## Model Artifacts (Frozen v2)

| Artifact | SHA256 (first 16 chars) | Description |
|---|---|---|
| models/lgbm_reg60.txt | 10ca2de44a94a471 | Regression head (+60 min glucose) |
| models/lgbm_event.txt | d62fbf7de496dbd8 | Event head (glucose > 180 in 60 min) |
| models/freeze.json | dfaa96e374ae5f01 | Freeze metadata + operating points |

**Status**: FROZEN v2 — no training after this point, ever. `models/freeze.json` status field and hashes enforced by `tests/test_smoke.py::test_frozen_v2_untouched`.

## Version History

- **v2 (frozen)**: B+wearable-lite (15 features: CGM + meals + HR/activity). Monotone constraints: +1 carbs, −1 steps. Trained on CGMacros, GroupKFold by patient.
- **v3 (not promoted)**: 23 features (added fat/protein rolls, glucose_accel, glucose_z, hr_vs_baseline, pm_rise_baseline). Directional improvement on all metrics but CIs overlap v2 — null result documented in `results/v3_vs_v2.json`.

## Contact

GlucoTwin team — Happiest Health Digital Twin Challenge 2026 submission.
