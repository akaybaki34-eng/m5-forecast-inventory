"""Hand-checkable tests for the lost-sales (R, S_t) simulator and safety-stock rules."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_inventory.inventory import (Policy, cumulative_errors, order_up_to, portfolio_kpis,  # noqa: E402
                                          simulate, ss_empirical_quantile)


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.pol = Policy(review_period=7, lead_time=2, service_level=0.95, holding_rate=0.365,
                          shortage_multiplier=1.0, order_cost=1.0)

    def test_hand_computed_example(self):
        # 1 SKU, 10 days, demand 3/day, price 1, start with 5 units, single review on day 0
        D = np.full((1, 10), 3.0)
        P = np.ones((1, 10))
        S = np.array([[20.0]])
        r = simulate(D, P, [0], S, self.pol, initial_on_hand=np.array([5.0]))
        # day0: order 15 (arrives day 2); days 0-1 sell 3,2 (1 lost on day1); day2 +15 -> 15
        # days 2..6 sell 3 each -> stock 0 after day 6; days 7..9 lose 3 each
        self.assertEqual(r["sold"][0], 3 + 2 + 15)
        self.assertEqual(r["lost"][0], 1 + 9)
        self.assertEqual(r["n_orders"][0], 1)
        end_stock = [2, 0, 12, 9, 6, 3, 0, 0, 0, 0]
        self.assertAlmostEqual(r["holding_cost"][0], sum(end_stock) * 0.365 / 365)
        self.assertAlmostEqual(r["shortage_cost"][0], 10.0)

    def test_conservation_and_non_negativity(self):
        rng = np.random.default_rng(0)
        D = rng.poisson(4, size=(20, 56)).astype(float)
        P = rng.uniform(1, 5, size=(20, 56))
        fc = np.full((20, 70), 4.0)
        rd = list(range(0, 56, 7))
        S = order_up_to(fc, rd, self.pol, ss=np.full(20, 3.0))
        r = simulate(D, P, rd, S, self.pol, initial_on_hand=S[:, 0], keep_trajectory=True)
        np.testing.assert_allclose(r["sold"] + r["lost"], D.sum(axis=1))
        self.assertTrue((r["on_hand"] >= 0).all())
        k = portfolio_kpis(r)
        self.assertGreater(k["fill_rate"], 0.8)
        self.assertAlmostEqual(k["total_cost"], k["holding_cost"] + k["shortage_cost"] + k["order_cost"])

    def test_perfect_forecast_no_lost_sales(self):
        D = np.tile(np.array([2.0, 5, 1, 0, 3, 4, 2]), (3, 8))  # (3, 56)
        P = np.ones_like(D)
        rd = list(range(0, 56, 7))
        fc = np.concatenate([D, D[:, :14]], axis=1)  # oracle forecast
        S = order_up_to(fc, rd, self.pol, ss=np.zeros(3))
        r = simulate(D, P, rd, S, self.pol, initial_on_hand=S[:, 0])
        self.assertEqual(r["lost"].sum(), 0)

    def test_warmup_not_counted(self):
        D = np.full((1, 20), 1.0)
        r_all = simulate(D, np.ones((1, 20)), [0, 7, 14], np.full((1, 3), 10.0), self.pol, np.array([10.0]))
        r_w = simulate(D, np.ones((1, 20)), [0, 7, 14], np.full((1, 3), 10.0), self.pol, np.array([10.0]),
                       account_from=7)
        self.assertEqual(r_w["demand"][0], 13)
        self.assertLessEqual(r_w["n_orders"][0], r_all["n_orders"][0])

    def test_cost_optimal_service_level(self):
        p = Policy(review_period=7, holding_rate=0.25, shortage_multiplier=1.0)
        self.assertAlmostEqual(p.cost_optimal_service_level(), 1 / (1 + 0.25 * 7 / 365))
        self.assertGreater(p.cost_optimal_service_level(), 0.95)

    def test_empirical_quantile_ss(self):
        rng = np.random.default_rng(3)
        err = rng.normal(0, 1, size=(50, 400))
        ss = ss_empirical_quantile(err, np.ones(50), np.array(["a"] * 50), Policy(service_level=0.95))
        self.assertAlmostEqual(ss[0], 1.645, delta=0.06)

    def test_cumulative_errors_shape_and_value(self):
        a = np.ones((2, 28))
        f = np.zeros((2, 40))
        ce = cumulative_errors(a, f, [0, 7, 21], H=12)
        self.assertEqual(ce.shape, (2, 3))
        self.assertEqual(ce[0, 0], 12)
        self.assertTrue(np.isnan(ce[0, 2]))  # block runs past the observed window


if __name__ == "__main__":
    unittest.main()
