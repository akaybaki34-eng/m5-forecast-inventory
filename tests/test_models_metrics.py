"""Baselines, metrics, segmentation and decision-analysis helpers."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_inventory import metrics  # noqa: E402
from forecast_inventory.experiments import holm, pareto_front, regret  # noqa: E402
from forecast_inventory.models.baselines import (croston_sba, ets_ana, moving_average,  # noqa: E402
                                                 seasonal_naive, tsb)
from forecast_inventory.segmentation import abc_classify, adi_cv2, classify_demand  # noqa: E402


class TestBaselines(unittest.TestCase):
    def test_seasonal_naive_repeats_last_week(self):
        Y = np.arange(21, dtype=float)[None, :]
        f = seasonal_naive(Y, 10)
        np.testing.assert_array_equal(f[0], [14, 15, 16, 17, 18, 19, 20, 14, 15, 16])

    def test_moving_average_ignores_inactive(self):
        Y = np.array([[np.nan] * 20 + [2.0] * 8 + [4.0] * 0])
        self.assertEqual(moving_average(Y, 3)[0, 0], 2.0)

    def test_ets_recovers_pure_weekly_pattern(self):
        week = np.array([5, 1, 1, 2, 8, 9, 3], dtype=float)
        Y = np.tile(week, 60)[None, :]
        f = ets_ana(Y, 7)
        np.testing.assert_allclose(f[0], week, atol=0.05)

    def test_intermittent_methods_on_constant_series(self):
        Y = np.full((1, 200), 3.0)
        self.assertAlmostEqual(tsb(Y, 5)[0, 0], 3.0, places=4)
        f = croston_sba(Y, 5)[0, 0]
        self.assertTrue(2.5 < f <= 3.0)  # SBA deflates by (1 - alpha/2)

    def test_croston_rate_on_every_other_day(self):
        Y = np.tile([0.0, 4.0], 200)[None, :]
        self.assertAlmostEqual(tsb(Y, 1)[0, 0], 2.0, delta=0.3)


class TestMetrics(unittest.TestCase):
    def test_wape_bias(self):
        a = np.array([[1.0, 0.0, 3.0, np.nan]])
        f = np.array([[2.0, 1.0, 1.0, 100.0]])
        self.assertAlmostEqual(metrics.wape(a, f), 4 / 4)
        self.assertAlmostEqual(metrics.bias(a, f), 0.0)
        self.assertAlmostEqual(metrics.mae(a, f), 4 / 3)


class TestSegmentation(unittest.TestCase):
    def test_classify_quadrants(self):
        cls = classify_demand(np.array([1.0, 1.0, 2.0, 2.0]), np.array([0.1, 0.9, 0.1, 0.9]))
        self.assertEqual(list(cls), ["smooth", "erratic", "intermittent", "lumpy"])

    def test_adi_counts_only_active_days(self):
        y = np.array([[np.nan, np.nan, 0, 2, 0, 2]])
        s = adi_cv2(y, 6)
        self.assertEqual(s["adi"][0], 2.0)
        self.assertEqual(s["cv2"][0], 0.0)

    def test_abc(self):
        sales = np.array([[80.0], [15.0], [5.0]])
        abc = abc_classify(sales, np.ones_like(sales), end=1)
        self.assertEqual(list(abc["abc"]), ["A", "B", "C"])


class TestDecisionHelpers(unittest.TestCase):
    def test_pareto(self):
        df = pd.DataFrame({"x": [1, 2, 3, 2.5], "y": [3, 2, 1, 2.5]}, index=list("abcd"))
        self.assertEqual(list(pareto_front(df, "x", "y")), [True, True, True, False])

    def test_regret(self):
        c = pd.DataFrame({"m1": [100.0, 90.0], "m2": [110.0, 90.0]})
        r = regret(c)
        self.assertEqual(r.loc[0, "m2"], 0.1)
        self.assertEqual(r.loc[1, "m1"], 0.0)

    def test_holm_monotone(self):
        adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)


if __name__ == "__main__":
    unittest.main()
