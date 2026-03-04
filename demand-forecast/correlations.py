"""
Cross-Store Demand Correlation and VAR Modelling.

Provides:
  1. compute_correlation_matrix — Pearson correlation of store-level demand.
  2. fit_var_model              — Reduced-form VAR(p) via OLS per equation.
  3. var_forecast               — h-step-ahead forecast from a fitted VAR.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ── Correlation ───────────────────────────────────────────────────────────────

def compute_correlation_matrix(
    forecasts_wide: pd.DataFrame,
    aggregate_by: str = "store",
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix of demand across stores.

    Parameters
    ----------
    forecasts_wide : DataFrame with columns [store_id, product_id, day+1, ..., day+7]
    aggregate_by   : 'store' aggregates by summing across products per store.

    Returns
    -------
    DataFrame (store × store) of Pearson correlations.
    """
    day_cols = [c for c in forecasts_wide.columns if c.startswith("day+")]
    if not day_cols:
        raise ValueError("forecasts_wide must contain 'day+N' columns.")

    if aggregate_by == "store":
        store_demand = (
            forecasts_wide.groupby("store_id")[day_cols]
            .sum()
        )
        # Transpose so rows=days, cols=stores — then correlate stores
        corr = store_demand.T.corr()
        return corr
    else:
        raise ValueError(f"aggregate_by={aggregate_by!r} not supported; use 'store'.")


# ── VAR model ─────────────────────────────────────────────────────────────────

@dataclass
class VARModel:
    """Minimal reduced-form VAR(p) estimated by OLS."""
    lags: int
    coef_matrices: list[np.ndarray] = field(default_factory=list)  # A_1, ..., A_p
    intercepts: np.ndarray = field(default_factory=lambda: np.array([]))
    resid_cov: np.ndarray = field(default_factory=lambda: np.array([]))
    store_ids: list = field(default_factory=list)
    n_obs_fit: int = 0

    def forecast(self, history: np.ndarray, steps: int = 7) -> np.ndarray:
        """
        h-step ahead forecast.

        Parameters
        ----------
        history : ndarray shape (T, K) — last ``lags`` rows are used.
        steps   : forecast horizon.

        Returns
        -------
        ndarray shape (steps, K)
        """
        K = history.shape[1]
        buf = list(history[-self.lags :])  # sliding window
        preds = []
        for _ in range(steps):
            y_next = self.intercepts.copy()
            for lag_i, A in enumerate(self.coef_matrices):
                y_next += A @ buf[-(lag_i + 1)]
            preds.append(y_next)
            buf.append(y_next)
        return np.array(preds)


def fit_var_model(
    demand_matrix: np.ndarray | pd.DataFrame,
    maxlags: int = 2,
    store_ids: list | None = None,
) -> VARModel:
    """
    Fit a VAR(maxlags) model by OLS (one equation per variable).

    Parameters
    ----------
    demand_matrix : (T, K) array or DataFrame — rows=time, cols=stores.
    maxlags       : number of lags p.
    store_ids     : optional list of store labels (length K).

    Returns
    -------
    VARModel instance.
    """
    if isinstance(demand_matrix, pd.DataFrame):
        store_ids = store_ids or list(demand_matrix.columns)
        Y = demand_matrix.values.astype(float)
    else:
        Y = np.asarray(demand_matrix, dtype=float)
        store_ids = store_ids or list(range(Y.shape[1]))

    T, K = Y.shape
    p = min(maxlags, T - K - 1)  # guard: need enough observations

    if p < 1:
        raise ValueError(
            f"Not enough observations (T={T}) to fit VAR({maxlags}) with K={K} variables."
        )

    # Build regressor matrix X and response Y_dep
    # Each row of Y_dep corresponds to time index t = p, p+1, ..., T-1
    Y_dep = Y[p:]                         # (T-p, K)
    X_rows = []
    for t in range(p, T):
        row = [1.0]                        # intercept
        for lag in range(1, p + 1):
            row.extend(Y[t - lag])
        X_rows.append(row)
    X = np.array(X_rows)                  # (T-p, 1 + p*K)

    # OLS: B = (X'X)^{-1} X' Y_dep   (one regression per equation)
    B, _, _, _ = np.linalg.lstsq(X, Y_dep, rcond=None)  # (1+p*K, K)

    intercepts = B[0]                     # (K,)
    coef_matrices = []
    for lag in range(p):
        start = 1 + lag * K
        A = B[start : start + K].T        # (K, K)
        coef_matrices.append(A)

    residuals = Y_dep - X @ B
    resid_cov = (residuals.T @ residuals) / max(T - p - 1, 1)

    return VARModel(
        lags=p,
        coef_matrices=coef_matrices,
        intercepts=intercepts,
        resid_cov=resid_cov,
        store_ids=store_ids,
        n_obs_fit=T - p,
    )


def var_forecast(
    model: VARModel,
    history: np.ndarray | pd.DataFrame,
    steps: int = 7,
) -> pd.DataFrame:
    """
    Convenience wrapper: return h-step forecast as a DataFrame.

    Parameters
    ----------
    model   : fitted VARModel
    history : (T, K) array — T ≥ model.lags
    steps   : forecast horizon

    Returns
    -------
    DataFrame (steps × K), columns = model.store_ids
    """
    if isinstance(history, pd.DataFrame):
        mat = history.values.astype(float)
    else:
        mat = np.asarray(history, dtype=float)

    preds = model.forecast(mat, steps=steps)
    return pd.DataFrame(preds, columns=model.store_ids)
