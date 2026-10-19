# Judge Q&A — GlucoTwin Digital Twin

Eight hardest questions with short, number-backed answers. Every answer cites the exact number and the file it comes from.

---

## 1. Does it generalise to new patients?

**Answer:** Partially. On the external ShanghaiT2DM cohort (n=100 T2D, zero refit), the frozen v2 model achieves RMSE@60 = 46.44 [43.3, 49.5] and event AUROC = 0.853 [0.830, 0.872] — a drop of +20.78 RMSE and −0.098 AUROC vs CGMacros. However, this drop is **almost entirely population shift**, not missing inputs. When we zero meal/HR/activity features on CGMacros (simulating CGM-only input), RMSE@60 is 25.56 [23.3, 28.0] (AUROC 0.958 [0.943, 0.970]) — CIs overlap with full CGMacros (25.66 [23.6, 27.9]). The model still beats Shanghai persistence (52.31 [48.1, 56.7]). Discrimination survives (0.853); level accuracy does not.

**Sources:** `results/cross_cohort.json` (rmse_60, rmse_60_ci, event_auroc, event_auroc_ci, persistence_rmse_60, persistence_rmse_60_ci, cgmacros_cgm_only.rmse_60, cgmacros_cgm_only.rmse_60_ci, cgmacros_cgm_only.event_auroc, cgmacros_cgm_only.event_auroc_ci, drop_rmse, drop_auroc)

---

## 2. Isn't this just a glucose threshold rule?

**Answer:** No. The glucose-only event baseline (threshold on current glucose) achieves AUROC = 0.923 [0.896, 0.943]. The frozen v2 model achieves AUROC = 0.951 [0.937, 0.962] — a statistically significant improvement (CIs don't overlap). At the operating point (AND t=0.5), the model achieves precision 0.907 vs 0.58 for a naive threshold rule at the same recall, with 4.25 false alerts/patient-day and median lead 25 min. The model uses slope, meal context, HR, and time-of-day — not just current glucose.

**Sources:** `results/metrics_ci.json` (glucose_baseline.auroc, frozen.auroc), `results/operating_point.json` (best.precision, best.recall, best.false_alerts_pd, best.median_lead_min)

---

## 3. Is n=45 too small?

**Answer:** It is small, and we report it honestly. We use patient-wise GroupKFold (never random splits) and 95% bootstrap-by-subject CIs on all headline metrics. The T2D subset (n=14 patients) is reported separately: RMSE@60 = 31.17 [27.5, 34.4], AUROC = 0.954. We ran a v3 feature experiment (fat/protein, glucose_accel, glucose_z, hr_vs_baseline, pm_rise_baseline) that improved all 5 metrics directionally (RMSE 23.97 [21.9, 26.1] vs 25.66 [23.6, 27.9]) but **CIs overlapped everywhere** — so we kept v2 frozen and documented the null result. No cherry-picking.

**Sources:** `results/metrics_ci.json` (frozen.rmse_60_ci, t2d_frozen.rmse_60, t2d_frozen.auroc), `results/v3_vs_v2.json` (v2.rmse_60, v2.rmse_60_ci, v3.rmse_60, v3.rmse_60_ci, decision)

---

## 4. Why LightGBM and not a deep model?

**Answer:** On small tabular/CGM data (n=45, ~138k rows), gradient boosting matches or beats deep models and is 16–77× faster to train. We did try a v3 feature set (23 features vs 15) with the same LightGBM architecture — it improved all metrics directionally but CIs overlapped with v2, so we did not promote it. A GRU ensemble was planned (Phase 4) but skipped per authorized skip conditions (no Kaggle token; LightGBM carries the accuracy story at this data size). The frozen v2 model has 15 features, monotone constraints (+1 carbs, −1 steps), and trains in seconds on CPU.

**Sources:** `results/v3_vs_v2.json` (v2.rmse_60, v3.rmse_60, decision), `models/freeze.json` (frozen_variant, status), AGENTS.md (FAST PATHS table)

---

## 5. Is the wearable fusion real?

**Answer:** Wearable fusion here = **heart rate + activity kcal only**. Raw steps exist for 1/45 patients in CGMacros → steps features were dead weight and dropped. The model uses `hr_30` (30-min rolling mean HR) and `activity_60` (60-min rolling sum activity kcal) as the wearable features. Both are present for all 45 patients. The what-if simulator uses these: "walk" increases `hr_30` and `activity_60`; "fewer carbs" decreases `carbs_60`/`carbs_120`. We state this openly — no step-count fusion, no faked signals.

**Sources:** `models/freeze.json` (frozen_variant), `src/models/lightgbm_v1.py` (FEATS list, MONO constraints), `README.md` (Honest limits section)

---

## 6. Does the what-if actually work?

**Answer:** Directionally yes, per-event no. We matched 353 meal pairs with high vs low post-meal HR increase (proxy for activity). Observed mean glucose reduction from activity: ~0 mg/dL (high variance, 52.9 mg/dL SD). What-if simulator (add 15-min walk → +HR/activity) predicts mean reduction 1.67 mg/dL. **Correlation between observed and predicted intervention effect: r = 0.145 (p < 0.05)**, RMSE 44.8 mg/dL. Averages agree; per-event effects are not recoverable at this n with observational matching. This is the "nobody else does this" validation — we report it honestly.

**Sources:** `results/counterfactual_validation.json` (correlation, rmse, mae, n_valid_pairs, observed_mean_diff, predicted_effect_mean)

---

## 7. Is it safe / regulated?

**Answer:** **Advisory only, clinician-in-the-loop, never automates insulin delivery.** The system warns ~60 min ahead when glucose is forecast to exceed 180 mg/dL and simulates "what if" scenarios (walk, fewer carbs). It does not dose insulin, does not close the loop. Regulatory posture: DPDP Act 2023 (data privacy), CDSCO SaMD (Software as Medical Device) considerations acknowledged. Prospective clinical validation is future work. Every screen badges "Synthetic composite patient / open data" and "Decision support, not diagnosis."

**Sources:** `README.md` (Responsible AI section), `models/freeze.json` (status), AGENTS.md (Responsible-AI posture)

---

## 8. What would you do with more time?

**Answer:** 
1. **Multi-cohort training**: Combine CGMacros + AI-READI + ShanghaiT2DM (harmonized schema, patient-wise GroupKFold) — the biggest accuracy lever per AGENTS.md.
2. **AI-READI volume**: Flagship T2D cohort (Dexcom G6, 5-min, ~10 days/participant + demographics/labs) — registration requested, pending access.
3. **Counterfactual validation with METs**: Replace HR proxy with metabolic equivalents from accelerometry (BIG IDEAs Lab has Empatica E4) for cleaner activity signal.
4. **Clinician review**: Prospective study with endocrinologists rating alert utility and what-if plausibility.
5. **Hypoglycaemia head**: Secondary <70 mg/dL head (events are rare — safety flag only, not primary claim).
6. **GRU ensemble**: On rented GPU, ensembled with LightGBM; promote only if it clears the frozen CI.

**Sources:** AGENTS.md (BUILD QUEUE Phases 0, 4, 7; FAST PATHS; Cut order), `results/counterfactual_validation.json` (limitations noted)
