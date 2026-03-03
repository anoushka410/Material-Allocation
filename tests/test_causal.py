"""Tests for demand-forecast/causal.py — Interrupted Time Series."""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demand-forecast"))
from causal import estimate_promo_lift, ITSResult


def _make_series_with_lift(n=60, intervention=40, lift=5.0, seed=0):
    """Generate a series with a known step lift at intervention_index."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(n):
        y[t] = 10.0 + 0.1 * t + (lift if t >= intervention else 0.0) + rng.standard_normal() * 0.5
    return y


class TestITSEstimation:
    def test_returns_its_result(self):
        y = _make_series_with_lift()
        result = estimate_promo_lift(y, intervention_index=40)
        assert isinstance(result, ITSResult)

    def test_step_effect_positive(self):
        """With a known positive lift, step_effect should be positive."""
        y = _make_series_with_lift(lift=5.0)
        result = estimate_promo_lift(y, intervention_index=40)
        assert result.step_effect > 0

    def test_step_effect_negative_for_dip(self):
        """With a known negative shock, step_effect should be negative."""
        y = _make_series_with_lift(lift=-5.0)
        result = estimate_promo_lift(y, intervention_index=40)
        assert result.step_effect < 0

    def test_r_squared_reasonable(self):
        """R² should be reasonable (> 0.5) for clean synthetic data."""
        y = _make_series_with_lift(lift=10.0)
        result = estimate_promo_lift(y, intervention_index=40)
        assert result.r_squared > 0.5

    def test_counts(self):
        y = _make_series_with_lift(n=60, intervention=40)
        result = estimate_promo_lift(y, intervention_index=40)
        assert result.n_pre == 40
        assert result.n_post == 20

    def test_summary_contains_key_terms(self):
        y = _make_series_with_lift()
        result = estimate_promo_lift(y, 40)
        s = result.summary()
        assert "Step effect" in s
        assert "Cumulative lift" in s
        assert "R²" in s

    def test_too_few_pre_raises(self):
        with pytest.raises(ValueError, match="3 pre-period"):
            estimate_promo_lift([1, 2, 10, 11, 12], intervention_index=2)

    def test_too_few_post_raises(self):
        y = list(range(50))
        with pytest.raises(ValueError, match="3 post-period"):
            estimate_promo_lift(y, intervention_index=48)

    def test_accepts_list(self):
        y = list(_make_series_with_lift())
        result = estimate_promo_lift(y, intervention_index=40)
        assert result.n_pre == 40

    def test_cumulative_lift_sign(self):
        """Cumulative lift sign should match direction of step_effect."""
        y = _make_series_with_lift(lift=8.0)
        result = estimate_promo_lift(y, intervention_index=40)
        assert (result.cumulative_lift > 0) == (result.step_effect > 0)
