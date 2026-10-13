"""Leakage and train/inference-consistency tests for the feature pipeline."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecast_inventory.features import BASE_FEATURES, day_features  # noqa: E402
from forecast_inventory.models.ml import recursive_forecast  # noqa: E402


def synthetic(n=5, T=120, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.poisson(5, size=(n, T)).astype(float)
    Y[0, :10] = np.nan  # an item that becomes active later
    P = np.full((n, T), 2.0)
    P[0, :10] = np.nan
    cal = rng.integers(0, 2, size=(T, 9)).astype(float)
    return Y, P, cal


class MeanOfLags:
    """Tiny deterministic 'model': prediction = mean of lag_1 and rolling_mean_7."""

    def predict(self, X):
        return np.nan_to_num((X[:, 1] + X[:, 5]) / 2)


class TestFeatures(unittest.TestCase):
    def test_feature_count_matches_names(self):
        Y, P, cal = synthetic()
        self.assertEqual(day_features(Y, P, cal, 50).shape, (Y.shape[0], len(BASE_FEATURES)))

    def test_day_features_ignore_target_day_and_future(self):
        Y, P, cal = synthetic()
        t = 60
        before = day_features(Y, P, cal, t)
        Y2 = Y.copy()
        Y2[:, t:] = 9999.0  # tamper with the target day and everything after it
        np.testing.assert_array_equal(before, day_features(Y2, P, cal, t))

    def test_rolling_matches_pandas_shift_rolling(self):
        Y, P, cal = synthetic()
        t = 70
        X = day_features(Y, P, cal, t)
        s = pd.Series(Y[1])
        names = BASE_FEATURES
        self.assertAlmostEqual(X[1, names.index("rolling_mean_7")], s.shift(1).rolling(7).mean().iloc[t])
        self.assertAlmostEqual(X[1, names.index("rolling_std_28")], s.shift(1).rolling(28).std().iloc[t])
        self.assertEqual(X[1, names.index("lag_7")], Y[1, t - 7])

    def test_short_history_gives_nan_not_partial_window(self):
        Y, P, cal = synthetic()
        X = day_features(Y, P, cal, 15)  # item 0 active from day 10 -> only 5 days of history
        self.assertTrue(np.isnan(X[0, BASE_FEATURES.index("rolling_mean_7")]))
        self.assertFalse(np.isnan(X[1, BASE_FEATURES.index("rolling_mean_7")]))

    def test_recursive_forecast_never_reads_actuals_after_origin(self):
        Y, P, cal = synthetic()
        origin, h = 80, 28
        f1 = recursive_forecast(MeanOfLags(), Y, P, cal, origin, h)
        Y2 = Y.copy()
        Y2[:, origin:] = np.random.default_rng(1).poisson(50, size=Y2[:, origin:].shape)
        f2 = recursive_forecast(MeanOfLags(), Y2, P, cal, origin, h)
        np.testing.assert_array_equal(f1, f2)
        self.assertTrue((f1 >= 0).all())

    def test_recursive_feeds_own_forecasts(self):
        Y, P, cal = synthetic()
        origin = 80
        f, feats = recursive_forecast(MeanOfLags(), Y, P, cal, origin, 3, keep_features=True)
        # lag_1 at step 2 must equal the step-1 forecast, not the actual demand
        np.testing.assert_allclose(feats[1][:, BASE_FEATURES.index("lag_1")], f[:, 0])


if __name__ == "__main__":
    unittest.main()
