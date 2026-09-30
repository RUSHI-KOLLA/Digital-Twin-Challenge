# GlucoTwin — Master Context & Build Contract (AGENTS.md)

You are assisting on **GlucoTwin**, a proof-of-concept digital twin for Type 2
Diabetes, built for the Happiest Health "Reimagining and Reforming Healthcare in
India Summit 2026" Digital Twin Challenge (submission target 19 Oct 2026 —
confirm exact deadline and deliverables on Unstop in Phase 0).

This single file contains everything: the locked design, and the phase queue
you execute. There is no other source of truth. Work through the phases
top-to-bottom, verify every GATE, and do not deviate.

## OPERATING CONTRACT (binding rules for every interaction)

1. **Speed is the priority. Always choose the fastest path that does not
   compromise evaluation integrity.** The fast paths are listed below; they are
   evidence-backed, not shortcuts on quality.
2. **The accuracy guarantees are never cut, simplified, or deferred:**
   patient-wise (grouped) train/test splits, the persistence baseline, the
   ablation table, honest small-n reporting. These ARE the accuracy. Cutting
   them produces fake numbers and a dead submission.
3. **Do not redesign.** The architecture, datasets, event definition, and stack
   are locked. If something fails, fix the implementation, not the plan.
4. **Smallest working increment first.** Never emit large untested code. Each
   task block below is one increment; verify the GATE, then move on.
5. **If it isn't in git, it doesn't exist.** Commit after every GATE.
6. **Model freeze at the end of Phase 4 — no training after that, ever.**
7. When this file and your training data disagree, this file wins. When you
   believe the file is wrong, say so and stop — do not silently work around it.

## One-line summary

A doctor-facing digital twin that warns, ~60 minutes in advance, when a T2D
patient's glucose will exceed 180 mg/dL after a meal — and simulates what would
prevent it.

Pitch line: *"We don't predict diabetes. We predict what is about to happen to
this patient."* Closing line: *"Not a predictor — a rehearsal space for this
patient's metabolic future, personalized by their medical record and updated by
their wearable."*

## FAST PATHS (chosen once — follow them, they do not cost accuracy)

| Decision | Fast path | Why it doesn't cost accuracy |
|---|---|---|
| Dataset | Start on CGMacros (instant download); AI-READI/DiaGame/ShanghaiT2DM requested in Phase 0 and added only if they arrive before the model freeze | Real T2D data day one; more cohorts are a bonus, not a blocker |
| Core model | LightGBM first, always | On small tabular/CGM data, gradient boosting matches or beats deep models and is 16–77× faster. GRU is an optional add-on, never the foundation |
| Validation | Patient-wise GroupKFold from the very first model | Costs nothing in time; random splits are what produce fake accuracy |
| What-if | Monotone constraints in LightGBM (+1 carbs, −1 steps) | One parameter; guarantees sane counterfactuals; zero training cost |
| Demo | Replay mode (recorded stream, frozen model) | Removes live-data risk; inference is one forward pass |
| UI | Streamlit + Plotly | Hours, not days; clinician-grade forecast bands |
| Tuning | ONE Optuna sweep, ≤20 trials, Phase 4 only | Features beat hyperparameters on small data |

**MVP path:** CGMacros → labels → LightGBM → Streamlit. That is a demoable
product even if every optional upgrade fails. Everything else enhances it.

## Locked design decisions

- **Condition:** Type 2 Diabetes, Indian phenotype framing (earlier onset,
  high post-meal excursions on rice/carb-heavy diets, metformin +
  sulfonylurea therapy).
- **Single adverse event:** glucose > 180 mg/dL within the next 60 minutes.
  Forecasts also shown at +30 and +120 min, but it is ONE event. Hypoglycemia
  is future work only.
- **Non-negotiables:** real open data only for training/evaluation; never
  relabel T1D data as T2D; no scripted "injected spike" synthetic CGM;
  patient-wise validation; persistence baseline; fusion ablation; live-replay
  demo; what-if simulator; full submission package.

## Data stack

| Role | Dataset | Use |
|---|---|---|
| Core T2D CGM | AI-READI Flagship T2D (Fairhub) | Dexcom G6, 5-min, ~10 days/participant + demographics/labs. Registration required — request in Phase 0. |
| Day-one starter | CGMacros (PhysioNet) | 45 participants (14 T2D, 16 prediabetes, 15 healthy); CGM + Fitbit HR + meal macros + labs. Instant download. BUILD ON THIS. |
| Cross-cohort test | ShanghaiT2DM | ~100 T2D patients, 15-min CGM. Held-out cohort for the generalization test (add only if cheap). |
| Wearable enrichment (optional) | BIG IDEAs Lab (PhysioNet) | Dexcom G6 + Empatica E4 (HR, IBI/HRV, EDA, temp, accelerometer). |
| Synthetic EHR | Synthea (diabetes module) → FHIR JSON | Indian priors (ICMR-INDIAB age at onset, BMI). Personas + ABDM-friendly FHIR output. |
| Indian meals | IFCT 2017 | Carb/protein/fat for idli, dosa, rice, roti, ragi, biryani — powers the what-if simulator. |
| Benchmarks + code | GlucoBench (IrinaStatsLab) | Published CGM forecasting numbers and reusable loaders/metrics. Fork it; don't rebuild. |

**Composite-patient rule:** where a real dataset ships its own demographics and
labs (CGMacros, AI-READI), those are the real static features. Synthea supplies
population profiles and demo personas. A persona = Synthea history + a real
open CGM/wearable trajectory, labeled "synthetic composite" everywhere,
including a footer badge on every dashboard screen. State openly that no open
dataset contains Indian CGM data — saying it yourself reads as rigor.

**Skip list:** OhioT1DM, D1NAMO, HUPA-UCM (Type 1; OhioT1DM needs a signed
agreement), MIMIC (ICU, off-story), synthetic Kaggle CGM sets, simglucose as
primary evidence.

## Model and twin architecture (implement, don't redesign)

- **Static branch:** age, sex, BMI, diabetes duration, HbA1c, fasting glucose,
  eGFR, medications, comorbidities → small MLP → patient embedding.
- **Dynamic branch:** last 2–4 h at 5-min steps of CGM (level, slope, rolling
  variance), HR/HRV (where available), steps, time since meal, meal
  carbs/fat/protein, time-of-day (sin/cos), prior-night sleep → small TCN or
  GRU. Keep it tiny (~50–100k params); the data is small.
- **Fusion:** FiLM conditioning — the EHR embedding scales and shifts the
  temporal features ("the same CGM trace means something different at HbA1c
  9.2 vs 6.4").
- **Heads:** glucose forecast at +30/+60/+120 + excursion probability.
  **Alert rule:** analytic — fire when the forecast's upper quantile for +60
  crosses 180 mg/dL. No separate alert classifier.
- **Ensemble:** LightGBM on engineered features blended with the TCN/GRU; lead
  with whichever performs better. LightGBM monotone constraints: +1 on carbs
  features, −1 on steps features — what-if curves must always move the right way.
- **Physiological guardrail:** cap predicted rate of change at ~3 mg/dL/min
  (tune on data).
- **Uncertainty:** conformal prediction intervals (MAPIE or crepes) — the
  forecast is a band, not a line.
- **Personalization:** pretrain on cohort, fine-tune on each patient's first
  half of data, evaluate on the second half; report the lift (the twin
  "learning" the patient).
- **Drift flag:** if recent error grows, show "twin out of sync, recalibrating".
- **Explainability:** SHAP drivers in plain language, labeled
  "model-attributed drivers" (never causal).
- **Optional last (cut first):** LLM narrator for SBAR-style summaries. It only
  rephrases structured outputs — never generates predictions or clinical claims.

## Evaluation spec (the credibility core)

- Leave-one-subject-out or grouped CV. Never random row splits. Report the T2D
  subset separately with bootstrap-by-subject confidence intervals.
- Cross-cohort check when available: train on AI-READI/CGMacros, test on
  ShanghaiT2DM, report the drop honestly.
- Regression: MAE/RMSE at +30/+60/+120 vs persistence + Clarke/Parkes error grid.
- Alert quality: calibration curve, false alerts per patient-day (alert
  fatigue), median lead time.
- **Ablation (the proof slide):** A: CGM only → B: +meals → C: +wearable →
  D: +EHR/FiLM → E: +personalization. With tiny n the EHR gain may be small —
  report honestly with intervals.

## Tech stack (pinned)

Python 3.11; pandas/Polars; LightGBM, PyTorch, scikit-learn, SHAP, MAPIE or
crepes; FastAPI (+ WebSocket replay), parquet/SQLite; Streamlit + Plotly;
Docker Compose; Kaggle as rented GPU only.

## Repo structure

```
digital-twin-t2d/
├── data/raw, processed, synthetic
├── src/ingestion, preprocessing, features, models, personalization, explainability, simulation
├── api/main.py
├── dashboard/app.py
├── configs/  notebooks/  tests/  docker/  docs/
└── README.md, requirements.txt, Dockerfile, docker-compose.yml
```

## Workflow rules

- Local development is the source of truth; Kaggle is only a rented GPU:
  upload data once as private Kaggle datasets, train via
  `kaggle kernels push` (GPU enabled), pull outputs with
  `kaggle kernels output`, commit results to git. Interactive Kaggle notebooks
  are for data exploration only — anything that survives goes into the repo
  immediately.
- Pin all package versions in requirements.txt the day the environment works.
- All demo modes run fully offline (replay of held-out data, frozen model).

---

# BUILD QUEUE — execute top to bottom, verify every GATE

Two parallel tracks: **ML track** (Phases 1→4) and **DEMO track** (starts as
soon as Phase 1's parquet exists). Parallelism is the single biggest speed
multiplier.

## PHASE 0 — Setup (do immediately)

```
1. Create GitHub repo: glucotwin. Clone locally. Put THIS file at repo root.
2. python -m venv .venv && source .venv/bin/activate
3. pip install pandas numpy scikit-learn lightgbm pyarrow fastapi uvicorn streamlit plotly shap joblib
4. pip freeze > requirements.txt ; commit.
5. Download CGMacros from PhysioNet NOW (instant). Request AI-READI access NOW (slow).
6. Confirm the Unstop deadline and exact deliverables.
7. kaggle: create API token, pip install kaggle, upload CGMacros as PRIVATE dataset.
```
**GATE:** repo exists, CGMacros CSVs on disk, first commit pushed.

## PHASE 1 — Data in, schema out  [ML track]

> Read data/raw/cgmacros/*.csv into one unified table: patient_id, timestamp,
> glucose, hr, steps, meal_carbs, meal_fat, meal_protein. Resample to a 5-min
> grid, interpolate glucose gaps ≤15 min, meals and steps summed per bin. Save
> data/processed/unified.parquet. Print rows per patient and the event rate.

**GATE:** unified.parquet exists; per-patient row counts printed; one patient's
glucose curve looks like a real CGM day.

## PHASE 2 — Labels, baselines, LightGBM v1, ablation  [ML track]

**2a — Labels + baselines:**
> Add labels: event_60 = glucose exceeds 180 mg/dL at any point in the next
> 60 min; target_glucose_{30,60,120} = glucose shifted by horizon. Evaluate two
> baselines with patient-wise GroupKFold: (a) persistence (future = current),
> (b) linear extrapolation of 30-min slope capped at 3 mg/dL/min. Report RMSE
> and MAE at +30/+60/+120 and the event rate. Save results/metrics.json.

**GATE:** metrics.json exists. Record persistence RMSE@60 — every model must
beat it. Event rate between 5% and 60% (if 0% or 100%, fix labels first).

**2b — LightGBM v1:**
> Train LightGBM on engineered features (glucose, slope_15, slope_30, rolling
> mean/std 30/60, carbs last 60/120 min, time since meal, steps last 30/60,
> hr mean 30, tod sin/cos, day of week, plus static HbA1c, BMI, age, meds).
> GroupKFold by patient, monotone_constraints +1 carbs / −1 steps. Report
> RMSE/MAE vs baselines, event AUROC/AUPRC, false alerts per patient-day. Save
> the model + a SHAP summary plot.

**GATE:** LightGBM beats persistence at +60. If not, add features (not model
complexity) — max one retry, then move on anyway.

**2c — Ablation table (the proof slide):**
> Four identical-setting variants: A: CGM only. B: + meals. C: + wearable
> (steps/hr). D: + static EHR. Columns: RMSE@60, AUROC, AUPRC. Save
> results/ablation.csv + a table image.

**GATE:** ablation.csv done. Don't polish it yet — the demo consumes it later.

## PHASE 3 — Twin layer + product skeleton (parallel from here)  [ML + DEMO]

**3-ML — twin behaviors:**
> 1. Personalization: fine-tune on each test patient's first half of data,
>    evaluate on the second half; report the lift. 2. What-if: predict(state)
>    and simulate(state, intervention) with +20 g fewer carbs and +15-min walk;
>    monotone constraints keep curves sane. 3. Conformal intervals on the
>    forecast (MAPIE/crepes).

**GATE:** for at least one real patient, simulate(walk) lowers the predicted
peak, and intervals render as a band.

**3-DEMO — API + replay:**
> FastAPI app: POST /predict (patient window → forecast + risk + drivers) and
> WebSocket /replay streaming a held-out patient's day at 30× speed, calling
> /predict as data "arrives". SHAP top-3 drivers as plain-language strings.
> Runs offline from parquet + the frozen model; until the freeze, use
> precomputed predictions.

**GATE:** open the replay stream in a browser tab and watch predictions update.

## PHASE 4 — Optional upgrades, then MODEL FREEZE  [ML track, Kaggle GPU]

> Push to Kaggle: small PyTorch GRU (2 layers, <100k params) over the last
> 2–4 h of 5-min windows (glucose, carbs, steps, hr, tod) with the static-EHR
> patient embedding concatenated at each timestep (FiLM or concat). Ensemble
> with LightGBM by averaging. ONE Optuna sweep, ≤20 trials. Pull weights, commit.

**Skip conditions (finish faster, lose nothing material):**
- GRU/ensemble: skip if Phase 2–3 aren't complete — LightGBM carries the
  accuracy story at this data size.
- Extra datasets: add only if already downloaded and the loader is a <2 h job.
- Optuna sweep: skip if time is tight; default LightGBM params are near-optimal.

**GATE (HARD): model frozen at the end of this phase. No more training, ever.**

## PHASE 5 — Dashboard + personas + demo script  [DEMO track]

> Streamlit + Plotly single-screen doctor dashboard: 1. Patient header (name,
> age, HbA1c, meds from the Synthea persona). 2. Live CGM trace + forecast band
> + 180 line + alert marker. 3. Alert card: "Spike in ~47 min, 87% — drivers:
> carb load, rising slope, low activity". 4. What-if panel: buttons
> [15-min walk] [swap rice→ragi] → curves update. 5. Triage panel: patients
> ranked by current 2-h risk (precomputed). Footer on every screen:
> "Synthetic composite patient / open data".

Then: 3 Synthea T2D personas (Indian names, metformin + sulfonylurea, HbA1c
7.5–9) linked to real CGMacros trajectories as labeled composites. Hero:
"Ramesh, 52, Bengaluru, HbA1c 8.2%, metformin + glimepiride."

**GATE:** `docker compose up` shows everything on one screen, offline; the demo
runs start-to-finish twice without error.

## PHASE 6 — Package + submit (keep a buffer at the end)

> README with a demo GIF at the top + 4-line metrics table + one-command run.
> Architecture diagram (one image). Record the 3-minute video with the alert
> firing in the first 30 seconds, before any intro. Backup video offline. Submit.

**GATE:** submitted. Anything after is bonus polish only.

## 5-minute demo flow (fixed script)

1. Problem, 30 s: India's diabetes burden vs quarterly clinic visits.
2. Meet Ramesh, 60 s: EHR panel, live CGM replay starts.
3. Alert fires ~60 min ahead, 60 s: confidence band + plain-language drivers.
4. What-if, 60 s: swap rice for ragi or add a 15-min walk; peak drops on screen.
5. Proof, 60 s: ablation table, subject-wise metrics, alert burden.
6. Responsible AI, 30 s: "advisory only, clinician-in-the-loop, never automates
   insulin delivery."

## Cut order if slipping

Optuna sweep → GRU ensemble → triage panel → extra datasets.
**Never cut:** patient-wise validation, ablation table, replay demo, what-if
panel, submission package.

## THE ONE NUMBER TO REMEMBER

Persistence RMSE@60 from Phase 2a. Everything you build must visibly beat it,
and that comparison goes on the proof slide of the demo.

## Responsible-AI posture (use this exact framing)

Decision support, not diagnosis. No open Indian CGM data exists — say so. Small
T2D sample — present as proof of concept with honest intervals. Composite
patients are labeled. DPDP Act 2023 and CDSCO SaMD considerations acknowledged.
Prospective validation is future work.
