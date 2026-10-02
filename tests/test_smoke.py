"""Smoke tests: guard the headline claims (no training, files only)."""
import json
import os
import unittest

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(name):
    with open(os.path.join(ROOT, "results", name)) as f:
        return json.load(f)


class TestHeadlines(unittest.TestCase):
    def test_freeze(self):
        with open(os.path.join(ROOT, "models", "freeze.json")) as f:
            fr = json.load(f)
        self.assertIn("FROZEN", fr["status"])
        for a in fr["frozen_artifacts"]:
            self.assertTrue(os.path.exists(os.path.join(ROOT, a)), a)

    def test_beats_persistence(self):
        m = load("lgbm_metrics.json")
        self.assertLess(m["rmse_60"], m["rmse_60_persistence"])
        self.assertLess(m["t2d_rmse_60"], 40.0)

    def test_fusion_no_gain_claim(self):
        ci = load("metrics_ci.json")
        blo, bhi = ci["B_meals"]["rmse_60_ci"]
        for v in ["C_wearable", "D_ehr", "frozen"]:
            lo, hi = ci[v]["rmse_60_ci"]
            self.assertTrue(lo < bhi and blo < hi, f"{v} CI outside B range?")

    def test_event_not_just_threshold(self):
        ci = load("metrics_ci.json")
        self.assertGreater(ci["frozen"]["auroc"], ci["glucose_baseline"]["auroc"])

    def test_operating_point(self):
        op = load("operating_point.json")["best"]
        self.assertEqual(op["rule"], "AND")
        self.assertEqual(op["t"], 0.5)
        self.assertLessEqual(op["false_alerts_pd"], 5.0)
        self.assertGreaterEqual(op["median_lead_min"], 20)
        self.assertLessEqual(op["miss_rate"], 0.15)

    def test_demo_window(self):
        demo = load("demo_window.json")
        wi = load("whatif.json")
        self.assertEqual(wi["timestamp"], demo["timestamp"])
        self.assertTrue(30 <= demo["lead_min_to_180"] <= 70)
        self.assertLess(demo["pred_walk"], demo["pred_60"])
        self.assertLess(demo["pred_fewer_carbs"], demo["pred_60"])

    def test_triage_label(self):
        tri = pd.read_csv(os.path.join(ROOT, "results", "triage.csv"))
        self.assertIn("risk_60", tri.columns)
        self.assertNotIn("risk_2h", tri.columns)

    def test_personalization_honest(self):
        p = load("personalization.json")
        self.assertIn("median_lift", p)
        self.assertIn("n_worse", p)
        self.assertEqual(p["n_improved"] + p["n_worse"] + sum(
            1 for r in p["patients"] if r["lift"] == 0), p["n"])

    def test_conformal_heldout(self):
        c = load("conformal.json")
        self.assertIn("coverage_test_90", c)
        self.assertTrue(0.80 <= c["coverage_test_90"] <= 0.95)
        self.assertIn("q50_half_width", c)

    def test_frozen_v2_untouched(self):
        import hashlib
        want = {"models/lgbm_reg60.txt": "10ca2de44a94a471ab538ae471507b52",
                "models/lgbm_event.txt": "d62fbf7de496dbd871fc2ba02b3b6cc0",
                "models/freeze.json": "c8484a163c08fbc0e0e81e3869281ebb"}
        for rel, h in want.items():
            with open(os.path.join(ROOT, rel), "rb") as f:
                self.assertEqual(hashlib.md5(f.read()).hexdigest(), h, rel)

    def test_calibration_artifact(self):
        c = load("calibration.json")
        for k in ["brier_before", "brier_after", "slope", "intercept",
                  "precision_before", "precision_after"]:
            self.assertIn(k, c)
        self.assertTrue(0.0 <= c["brier_after"] <= 0.25)
        self.assertGreater(c["slope"], 0.0)
        self.assertTrue(os.path.exists(
            os.path.join(ROOT, "results", "reliability.png")))

    def test_clarke_grid(self):
        eg = load("error_grid.json")
        self.assertTrue(os.path.exists(
            os.path.join(ROOT, "results", "clarke.png")))
        self.assertGreaterEqual(eg["percent"]["A"] + eg["percent"]["B"], 90.0)

    def test_v3_documented_not_promoted(self):
        doc = load("v3_vs_v2.json")
        self.assertEqual(doc["decision"], "KEEP v2 (no promotion)")
        self.assertLess(doc["v3"]["rmse_60"], doc["v2"]["rmse_60"])  # direction
        lo2, hi2 = doc["v2"]["rmse_60_ci"]
        lo3, hi3 = doc["v3"]["rmse_60_ci"]
        self.assertTrue(lo3 < hi2 and lo2 < hi3)  # CIs overlap: null result

    def test_cross_cohort(self):
        c = load("cross_cohort.json")
        self.assertGreaterEqual(c["n_patients"], 90)
        self.assertIn("drop_rmse", c)  # gap documented, not hidden
        lo, hi = c["rmse_60_ci"]
        self.assertLess(lo, hi)


if __name__ == "__main__":
    unittest.main()
