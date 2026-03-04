"""Tests for demand-forecast/reconciler.py — Hierarchical Reconciliation."""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demand-forecast"))
from reconciler import reconcile_forecasts


def _make_forecast_df(n_stores=3, n_products=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_stores + 1):
        for p in range(1, n_products + 1):
            row = {"store_id": s, "product_id": p}
            for d in range(1, 8):
                row[f"day+{d}"] = float(rng.uniform(1.0, 10.0))
            rows.append(row)
    return pd.DataFrame(rows)


class TestBottomUpReconciliation:
    def test_returns_dataframe(self):
        df = _make_forecast_df()
        out = reconcile_forecasts(df, method="bottom_up")
        assert isinstance(out, pd.DataFrame)

    def test_has_rec_columns(self):
        df = _make_forecast_df()
        out = reconcile_forecasts(df, method="bottom_up")
        assert "rec_day+1" in out.columns

    def test_contains_product_total_rows(self):
        df = _make_forecast_df(n_stores=2, n_products=3)
        out = reconcile_forecasts(df, method="bottom_up")
        assert "product_total" in out["level"].values

    def test_contains_grand_total_row(self):
        df = _make_forecast_df(n_stores=2, n_products=3)
        out = reconcile_forecasts(df, method="bottom_up")
        assert "grand_total" in out["level"].values

    def test_grand_total_equals_sum_of_bottom(self):
        df = _make_forecast_df(n_stores=2, n_products=3)
        out = reconcile_forecasts(df, method="bottom_up")
        grand = out[out["level"] == "grand_total"]["rec_day+1"].values[0]
        bottom_sum = out[out["level"] == "bottom"]["rec_day+1"].sum()
        assert abs(grand - bottom_sum) < 1e-6

    def test_product_total_sums_match(self):
        df = _make_forecast_df(n_stores=2, n_products=3)
        out = reconcile_forecasts(df, method="bottom_up")
        for pid in [1, 2, 3]:
            prod_total = out[
                (out["level"] == "product_total") & (out["product_id"] == pid)
            ]["rec_day+1"].values[0]
            bottom_sum = out[
                (out["level"] == "bottom") & (out["product_id"] == pid)
            ]["rec_day+1"].sum()
            assert abs(prod_total - bottom_sum) < 1e-6

    def test_bottom_level_unchanged(self):
        df = _make_forecast_df(n_stores=2, n_products=3)
        out = reconcile_forecasts(df, method="bottom_up")
        bottom = out[out["level"] == "bottom"]
        for _, row in bottom.iterrows():
            assert abs(row["day+1"] - row["rec_day+1"]) < 1e-9


class TestMinTraceReconciliation:
    def test_returns_dataframe(self):
        df = _make_forecast_df(n_stores=2, n_products=3)
        out = reconcile_forecasts(df, method="mintrace_ols")
        assert isinstance(out, pd.DataFrame)

    def test_has_rec_columns(self):
        df = _make_forecast_df(n_stores=2, n_products=3)
        out = reconcile_forecasts(df, method="mintrace_ols")
        assert "rec_day+1" in out.columns

    def test_grand_total_coherent_with_bottom(self):
        """Grand total should equal sum of reconciled bottom-level."""
        df = _make_forecast_df(n_stores=2, n_products=2)
        out = reconcile_forecasts(df, method="mintrace_ols")
        grand = out[out["level"] == "grand_total"]["rec_day+1"].values[0]
        bottom_sum = out[out["level"] == "bottom"]["rec_day+1"].sum()
        assert abs(grand - bottom_sum) < 1e-4

    def test_invalid_method_raises(self):
        df = _make_forecast_df()
        with pytest.raises(ValueError, match="Unknown reconciliation method"):
            reconcile_forecasts(df, method="unknown_method")

    def test_missing_day_cols_raises(self):
        df = pd.DataFrame({"store_id": [1], "product_id": [1], "forecast": [5.0]})
        with pytest.raises(ValueError, match="day\\+"):
            reconcile_forecasts(df, method="bottom_up")
