"""Tests for demand-forecast/forecaster.py — EnsembleForecaster."""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demand-forecast"))
from forecaster import EnsembleForecaster


def _make_data(n=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.standard_normal(n) * 0.5
    return X, y


class TestEnsembleForecaster:
    def test_fit_predict_shape(self):
        X, y = _make_data(80)
        ef = EnsembleForecaster(n_bootstrap=20, random_state=0)
        ef.fit(X[:60], y[:60])
        preds = ef.predict(X[60:])
        assert preds.shape == (20,)

    def test_predict_with_intervals_shape(self):
        X, y = _make_data(80)
        ef = EnsembleForecaster(n_bootstrap=30, random_state=0)
        ef.fit(X[:60], y[:60])
        point, lower, upper = ef.predict_with_intervals(X[60:], confidence=0.9)
        assert point.shape == lower.shape == upper.shape == (20,)

    def test_intervals_ordering(self):
        X, y = _make_data(100)
        ef = EnsembleForecaster(n_bootstrap=50, random_state=1)
        ef.fit(X[:80], y[:80])
        point, lower, upper = ef.predict_with_intervals(X[80:], confidence=0.9)
        assert np.all(lower <= point + 1e-9), "lower bound exceeds point forecast"
        assert np.all(upper >= point - 1e-9), "upper bound below point forecast"

    def test_wider_interval_at_higher_confidence(self):
        X, y = _make_data(100)
        ef = EnsembleForecaster(n_bootstrap=100, random_state=2)
        ef.fit(X[:80], y[:80])
        _, l90, u90 = ef.predict_with_intervals(X[80:], confidence=0.90)
        _, l99, u99 = ef.predict_with_intervals(X[80:], confidence=0.99)
        width_90 = (u90 - l90).mean()
        width_99 = (u99 - l99).mean()
        assert width_99 >= width_90, "99% CI should be wider than 90% CI"

    def test_feature_importances(self):
        X, y = _make_data(80)
        ef = EnsembleForecaster(n_bootstrap=10, random_state=0)
        ef.fit(X[:60], y[:60])
        imp = ef.feature_importances()
        assert imp is not None
        assert imp.shape == (4,)
        assert abs(imp.sum() - 1.0) < 0.01

    def test_small_dataset(self):
        """Should not crash with fewer than 5 samples."""
        X, y = _make_data(10)
        ef = EnsembleForecaster(n_bootstrap=10, random_state=0)
        ef.fit(X[:8], y[:8])
        point, lower, upper = ef.predict_with_intervals(X[8:])
        assert point.shape == (2,)
