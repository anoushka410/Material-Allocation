"""
Ensemble Forecaster with Bootstrap Prediction Intervals.

Implements a three-model ensemble (Random Forest, Gradient Boosting, Ridge)
and uses bootstrap resampling to produce symmetric prediction intervals.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict


class EnsembleForecaster:
    """
    Ensemble of RF + GBM + Ridge with bootstrap prediction intervals.

    Usage
    -----
    >>> ef = EnsembleForecaster(n_bootstrap=50, random_state=42)
    >>> ef.fit(X_train, y_train)
    >>> point, lower, upper = ef.predict_with_intervals(X_test, confidence=0.9)
    """

    def __init__(self, n_bootstrap: int = 100, random_state: int = 0):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self._base_estimators: list = []
        self._bootstrap_residuals: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_estimators(random_state: int) -> list:
        return [
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]),
            RandomForestRegressor(
                n_estimators=100, max_features="sqrt",
                random_state=random_state, n_jobs=-1,
            ),
            GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05,
                max_depth=4, random_state=random_state,
            ),
        ]

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleForecaster":
        """Fit all base estimators and collect in-sample residuals."""
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self._base_estimators = self._make_estimators(self.random_state)
        for est in self._base_estimators:
            est.fit(X, y)

        # Collect leave-one-out residuals via CV for interval calibration
        oof_preds = np.column_stack([
            cross_val_predict(est, X, y, cv=min(5, len(y))) if len(y) >= 5
            else est.predict(X)
            for est in self._base_estimators
        ])
        oof_ensemble = oof_preds.mean(axis=1)
        self._bootstrap_residuals = y - oof_ensemble
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the mean ensemble prediction."""
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([est.predict(X) for est in self._base_estimators])
        return preds.mean(axis=1)

    def predict_with_intervals(
        self, X: np.ndarray, confidence: float = 0.90
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (point_forecast, lower_bound, upper_bound).

        Intervals are constructed by adding bootstrapped residuals to the
        ensemble point forecast, then taking the requested quantiles.
        """
        X = np.asarray(X, dtype=float)
        point = self.predict(X)

        if self._bootstrap_residuals is None or len(self._bootstrap_residuals) == 0:
            return point, point, point

        rng = np.random.default_rng(self.random_state)
        alpha = (1.0 - confidence) / 2.0

        # Bootstrap: draw residuals and add to point forecast
        boot_preds = np.empty((self.n_bootstrap, len(point)))
        for b in range(self.n_bootstrap):
            drawn = rng.choice(self._bootstrap_residuals, size=len(point), replace=True)
            boot_preds[b] = point + drawn

        lower = np.quantile(boot_preds, alpha, axis=0)
        upper = np.quantile(boot_preds, 1.0 - alpha, axis=0)
        return point, lower, upper

    # ------------------------------------------------------------------
    # Feature importance (weighted average over tree models)
    # ------------------------------------------------------------------
    def feature_importances(self) -> np.ndarray | None:
        tree_models = [
            est for est in self._base_estimators
            if hasattr(est, "feature_importances_")
        ]
        if not tree_models:
            return None
        importances = np.stack([m.feature_importances_ for m in tree_models])
        return importances.mean(axis=0)
