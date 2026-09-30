# GlucoTwin 5-minute demo (fixed script)

One command: `docker compose up` → dashboard http://localhost:8501, API http://localhost:8000.

1. **Problem, 30 s:** India's diabetes burden vs quarterly clinic visits.
   Quarterly HbA1c misses daily post-meal excursions on carb-heavy diets.
2. **Meet Ramesh, 60 s:** header (52, Bengaluru, HbA1c 8.2%, metformin +
   glimepiride — synthetic composite, open-data trajectory). Start replay of
   2025-02-17: CGM trace runs, forecast band + 180 line render.
3. **Alert fires ~55 min ahead, 60 s:** at 09:20, glucose 113, band crosses
   180, risk 80% — drivers: carb load, rising slope (model-attributed).
   Actual crosses 180 at ~10:15. Confidence band, not a line.
4. **What-if, 60 s:** 15-min walk → 193 to 189; swap rice→ragi (−20 g carbs,
   IFCT 2017) → 193 to 178. Curves update on screen.
5. **Proof, 60 s:** ablation (meals help; wearable/EHR flat at this n),
   grouped metrics (RMSE@60 25.83 vs persistence 29.73; AUROC 0.949),
   personalization lift +2.97 (14 T2D), alert burden; honest small-n intervals.
6. **Responsible AI, 30 s:** advisory only, clinician-in-the-loop, never
   automates insulin delivery. No open Indian CGM data exists — composites labeled.
