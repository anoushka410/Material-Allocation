"""Tests for monitoring/drift.py — ForecastDriftDetector."""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from monitoring.drift import ForecastDriftDetector, DriftAlert


def _add_observations(detector, store_id, product_id, n_baseline=30, n_post=10,
                       baseline_error=0.5, post_error=3.0, seed=0):
    """Add baseline + post observations, with inflated error in post-period."""
    rng = np.random.default_rng(seed)
    for i in range(n_baseline):
        actual = 10.0 + rng.standard_normal() * baseline_error
        pred = 10.0
        detector.add_observation(store_id, product_id, f"2024-01-{i+1:02d}", actual, pred)
    for i in range(n_post):
        actual = 10.0 + rng.standard_normal() * post_error
        pred = 10.0
        detector.add_observation(store_id, product_id, f"2024-02-{i+1:02d}", actual, pred)


class TestAddObservation:
    def test_adds_single_observation(self):
        d = ForecastDriftDetector()
        d.add_observation(1, 100, "2024-01-01", 10.0, 9.5)
        assert (1, 100) in d._obs
        assert len(d._obs[(1, 100)]) == 1

    def test_error_computed_correctly(self):
        d = ForecastDriftDetector()
        d.add_observation(1, 100, "2024-01-01", 10.0, 8.0)
        obs = d._obs[(1, 100)][0]
        assert abs(obs["error"] - 2.0) < 1e-9

    def test_bulk_add_from_df(self):
        d = ForecastDriftDetector()
        df = pd.DataFrame({
            "store_id": [1, 1, 2],
            "product_id": [100, 100, 200],
            "date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "actual": [10.0, 11.0, 5.0],
            "predicted": [9.5, 10.5, 5.5],
        })
        d.add_observations_from_df(df)
        assert len(d._obs[(1, 100)]) == 2
        assert len(d._obs[(2, 200)]) == 1


class TestSlidingWindowDetection:
    def test_detects_drift_with_inflated_error(self):
        d = ForecastDriftDetector(baseline_window=30, alert_window=7, mae_threshold=2.0)
        _add_observations(d, 1, 100, n_baseline=30, n_post=10,
                          baseline_error=0.1, post_error=5.0)
        alerts = d.get_alerts()
        sw_alerts = [a for a in alerts if a.method == "sliding_window"]
        assert len(sw_alerts) >= 1

    def test_no_drift_when_error_stable(self):
        d = ForecastDriftDetector(baseline_window=30, alert_window=7, mae_threshold=2.0)
        _add_observations(d, 1, 100, n_baseline=30, n_post=10,
                          baseline_error=0.5, post_error=0.5)
        sw_alerts = [a for a in d.get_alerts() if a.method == "sliding_window"]
        assert len(sw_alerts) == 0

    def test_not_enough_obs_returns_no_alert(self):
        d = ForecastDriftDetector(baseline_window=30, alert_window=7)
        d.add_observation(1, 100, "2024-01-01", 10.0, 9.0)
        assert d.get_alerts() == []


class TestCUSUMDetection:
    def test_detects_mean_shift(self):
        """A sudden, persistent positive shift in errors should trigger CUSUM."""
        d = ForecastDriftDetector(
            baseline_window=30, alert_window=7,
            cusum_k=0.5, cusum_h=2.0,
        )
        rng = np.random.default_rng(42)
        # Baseline: small errors
        for i in range(30):
            err = rng.standard_normal() * 0.2
            d.add_observation(1, 200, f"2024-01-{i+1:02d}", 10.0 + err, 10.0)
        # Post: large persistent positive bias
        for i in range(20):
            err = 3.0 + rng.standard_normal() * 0.1   # strong upward shift
            d.add_observation(1, 200, f"2024-02-{i+1:02d}", 10.0 + err, 10.0)
        cusum_alerts = [a for a in d.get_alerts() if a.method == "cusum"]
        assert len(cusum_alerts) >= 1


class TestAlertFormat:
    def test_alert_has_required_fields(self):
        d = ForecastDriftDetector(baseline_window=30, alert_window=7, mae_threshold=2.0)
        _add_observations(d, 5, 999, baseline_error=0.1, post_error=8.0)
        alerts = d.get_alerts()
        if alerts:
            a = alerts[0]
            assert isinstance(a, DriftAlert)
            assert a.store_id == 5
            assert a.product_id == 999
            assert a.baseline_mae >= 0
            assert a.recent_mae >= 0

    def test_to_dict(self):
        d = ForecastDriftDetector(baseline_window=30, alert_window=7, mae_threshold=1.5)
        _add_observations(d, 1, 100, baseline_error=0.1, post_error=5.0)
        alerts = d.get_alerts()
        if alerts:
            d_dict = alerts[0].to_dict()
            assert "store_id" in d_dict
            assert "method" in d_dict
            assert "message" in d_dict


class TestSummary:
    def test_summary_dataframe_shape(self):
        d = ForecastDriftDetector()
        _add_observations(d, 1, 100, n_baseline=10, n_post=5, post_error=1.0)
        _add_observations(d, 2, 200, n_baseline=10, n_post=5, post_error=1.0)
        df = d.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "mae" in df.columns
        assert "rmse" in df.columns

    def test_get_alerts_df_shape(self):
        d = ForecastDriftDetector(baseline_window=30, alert_window=7, mae_threshold=2.0)
        _add_observations(d, 1, 100, baseline_error=0.1, post_error=5.0)
        df = d.get_alerts_df()
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "store_id" in df.columns
            assert "method" in df.columns
