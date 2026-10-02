# GlucoTwin 5-minute demo (fixed script)

One command: `docker compose up` → dashboard http://localhost:8501, API http://localhost:8000.

1. **Problem, 30 s:** India's diabetes burden vs quarterly clinic visits.
   Quarterly HbA1c misses daily post-meal excursions on carb-heavy diets.
2. **Meet Ramesh, 60 s:** header (52, Bengaluru, HbA1c 8.2%, metformin +
   glimepiride — synthetic composite, open-data trajectory). Start replay of
   2025-02-16: CGM trace runs, forecast band + 180 line render.
3. **Alert fires ~30 min ahead, 60 s:** at 12:05, glucose 153, band crosses
   180, risk 82% — drivers: carb load, rising slope (model-attributed,
   not causal). Actual crosses 180 at ~12:35. Confidence band, not a line.
   (Median lead across 923 crossings: 45 min.)
4. **What-if, 60 s:** two levers, both lower the predicted peak. Swap
   rice→ragi (−20 g carbs, IFCT 2017) → 235 to 231; 15-min walk → 235 to 228.
   Say it openly: raw steps exist for 1/45 patients, so wearable fusion here
   is HR + activity kcal; the relative size varies by state, so the UI shows
   both numbers and doesn't rank them. Curves update on screen.
5. **Proof, 60 s:** ablation with bootstrap CIs (meals help; wearable/EHR
   overlap B — no measurable gain at this n, not "fusion hurts"), grouped
   metrics (RMSE@60 25.66 [23.6, 27.9] vs persistence 29.73; T2D-14: 31.17
   vs T2D persistence 36.34; AUROC 0.951 vs glucose-only 0.923 — not just a
   threshold rule), personalization median +1.19 (+9/−5 shown), clinical
   alert (AND t=0.7): precision 0.964, 1.42 false alerts/day, median lead
   15 min, miss 28%.
6. **Responsible AI, 30 s:** advisory only, clinician-in-the-loop, never
   automates insulin delivery. No open Indian CGM data exists — composites labeled.
