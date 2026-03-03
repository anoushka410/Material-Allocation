"""Tests for demand-forecast/correlations.py — VAR and correlation matrix."""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demand-forecast"))
from correlations import compute_correlation_matrix, fit_var_model, var_forecast, VARModel


def _make_wide_df(n_stores=5, n_products=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_stores + 1):
        for p in range(1, n_products + 1):
            row = {"store_id": s, "product_id": p}
            for d in range(1, 8):
                row[f"day+{d}"] = float(rng.uniform(1.0, 20.0))
            rows.append(row)
    return pd.DataFrame(rows)


def _make_demand_matrix(T=50, K=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, K)).cumsum(axis=0)


class TestCorrelationMatrix:
    def test_returns_square_dataframe(self):
        df = _make_wide_df(n_stores=5)
        corr = compute_correlation_matrix(df)
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (5, 5)

    def test_diagonal_is_one(self):
        df = _make_wide_df(n_stores=4)
        corr = compute_correlation_matrix(df)
        diag = np.diag(corr.values)
        assert np.allclose(diag, 1.0, atol=1e-9)

    def test_symmetric(self):
        df = _make_wide_df(n_stores=4)
        corr = compute_correlation_matrix(df)
        assert np.allclose(corr.values, corr.values.T, atol=1e-9)

    def test_values_in_range(self):
        df = _make_wide_df(n_stores=4)
        corr = compute_correlation_matrix(df)
        assert corr.values.min() >= -1.0 - 1e-9
        assert corr.values.max() <= 1.0 + 1e-9

    def test_unsupported_aggregate_raises(self):
        df = _make_wide_df()
        with pytest.raises(ValueError):
            compute_correlation_matrix(df, aggregate_by="product")

    def test_missing_day_cols_raises(self):
        df = pd.DataFrame({"store_id": [1, 2], "product_id": [1, 1], "val": [5.0, 6.0]})
        with pytest.raises(ValueError):
            compute_correlation_matrix(df)


class TestVARModel:
    def test_fit_returns_var_model(self):
        mat = _make_demand_matrix(T=30, K=3)
        model = fit_var_model(mat, maxlags=2)
        assert isinstance(model, VARModel)
        assert model.lags >= 1

    def test_intercepts_shape(self):
        mat = _make_demand_matrix(T=30, K=3)
        model = fit_var_model(mat, maxlags=1)
        assert model.intercepts.shape == (3,)

    def test_coef_matrices_count(self):
        mat = _make_demand_matrix(T=30, K=3)
        model = fit_var_model(mat, maxlags=2)
        assert len(model.coef_matrices) == model.lags

    def test_coef_matrix_shape(self):
        mat = _make_demand_matrix(T=30, K=3)
        model = fit_var_model(mat, maxlags=2)
        for A in model.coef_matrices:
            assert A.shape == (3, 3)

    def test_forecast_shape(self):
        mat = _make_demand_matrix(T=30, K=3)
        model = fit_var_model(mat, maxlags=2)
        preds = model.forecast(mat, steps=7)
        assert preds.shape == (7, 3)

    def test_var_forecast_wrapper_shape(self):
        mat = _make_demand_matrix(T=30, K=4)
        model = fit_var_model(mat, maxlags=1, store_ids=["s1", "s2", "s3", "s4"])
        df = var_forecast(model, mat, steps=5)
        assert df.shape == (5, 4)
        assert list(df.columns) == ["s1", "s2", "s3", "s4"]

    def test_too_few_obs_raises(self):
        mat = _make_demand_matrix(T=5, K=4)
        with pytest.raises(ValueError, match="Not enough observations"):
            fit_var_model(mat, maxlags=2)

    def test_dataframe_input(self):
        df = pd.DataFrame(_make_demand_matrix(T=30, K=3), columns=["a", "b", "c"])
        model = fit_var_model(df, maxlags=1)
        assert model.store_ids == ["a", "b", "c"]
