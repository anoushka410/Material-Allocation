"""
Causal Impact: Interrupted Time Series (ITS) Analysis.

Estimates the causal lift of a promotion by fitting a segmented regression
with a level change (step) and slope change at the intervention point.

Model
-----
  y(t) = β0 + β1·t + β2·D(t) + β3·t·D(t) + ε

where D(t) = 1 if t ≥ intervention_index, 0 otherwise.

  β2 : immediate level change (step effect)
  β3 : change in trend slope after the intervention
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ITSResult:
    """Results from an Interrupted Time Series regression."""
    pre_mean: float
    post_mean: float
    step_effect: float          # β2
    step_effect_se: float
    step_pvalue: float
    slope_change: float         # β3
    slope_change_se: float
    slope_change_pvalue: float
    cumulative_lift: float      # sum of counterfactual gap over post period
    lift_pct: float             # % lift relative to counterfactual
    n_pre: int
    n_post: int
    r_squared: float
    coefficients: dict[str, float]

    def summary(self) -> str:
        sig = "✓ significant" if self.step_pvalue < 0.05 else "✗ not significant"
        return (
            f"ITS Promotion Lift Analysis\n"
            f"  Pre-period mean   : {self.pre_mean:.4f}\n"
            f"  Post-period mean  : {self.post_mean:.4f}\n"
            f"  Step effect (β₂) : {self.step_effect:+.4f}  SE={self.step_effect_se:.4f}  p={self.step_pvalue:.4f}  {sig}\n"
            f"  Slope change (β₃): {self.slope_change:+.4f}  SE={self.slope_change_se:.4f}  p={self.slope_change_pvalue:.4f}\n"
            f"  Cumulative lift   : {self.cumulative_lift:+.4f} units\n"
            f"  Lift %            : {self.lift_pct:+.2f}%\n"
            f"  R²                : {self.r_squared:.4f}  (n_pre={self.n_pre}, n_post={self.n_post})\n"
        )


def estimate_promo_lift(
    series: pd.Series | np.ndarray | list,
    intervention_index: int,
) -> ITSResult:
    """
    Fit an ITS regression and return the causal lift estimate.

    Parameters
    ----------
    series : 1-D array-like of demand observations (chronological order).
    intervention_index : integer index of the first post-intervention period.

    Returns
    -------
    ITSResult dataclass.

    Raises
    ------
    ValueError if there are fewer than 3 pre-period or post-period observations.
    """
    y = np.asarray(series, dtype=float)
    n = len(y)

    if intervention_index < 3:
        raise ValueError("Need at least 3 pre-period observations.")
    if n - intervention_index < 3:
        raise ValueError("Need at least 3 post-period observations.")

    t = np.arange(n, dtype=float)
    D = (t >= intervention_index).astype(float)
    tD = t * D

    # Design matrix: [intercept, t, D, t*D]
    X = np.column_stack([np.ones(n), t, D, tD])

    # OLS via pseudo-inverse
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat

    # Compute SE via OLS formula: Var(β) = σ² (X'X)^{-1}
    n_params = X.shape[1]
    df_resid = n - n_params
    sigma2 = (residuals ** 2).sum() / max(df_resid, 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))

    # t-statistics and p-values
    t_stats = beta / np.where(se > 0, se, 1e-12)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=max(df_resid, 1)))

    # Counterfactual: predicted without intervention (D=0, slope continues)
    t_post = t[intervention_index:]
    y_cf = beta[0] + beta[1] * t_post  # extrapolate pre-trend
    y_actual_post = y[intervention_index:]
    cumulative_lift = float((y_actual_post - y_cf).sum())

    pre_mean = float(y[:intervention_index].mean())
    cf_mean = float(y_cf.mean())
    lift_pct = (cumulative_lift / (cf_mean * len(t_post)) * 100.0) if cf_mean != 0 else 0.0

    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (residuals ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return ITSResult(
        pre_mean=pre_mean,
        post_mean=float(y_actual_post.mean()),
        step_effect=float(beta[2]),
        step_effect_se=float(se[2]),
        step_pvalue=float(p_values[2]),
        slope_change=float(beta[3]),
        slope_change_se=float(se[3]),
        slope_change_pvalue=float(p_values[3]),
        cumulative_lift=cumulative_lift,
        lift_pct=lift_pct,
        n_pre=intervention_index,
        n_post=n - intervention_index,
        r_squared=r2,
        coefficients={
            "intercept": float(beta[0]),
            "trend": float(beta[1]),
            "step": float(beta[2]),
            "slope_change": float(beta[3]),
        },
    )
