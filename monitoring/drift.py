"""
Forecast Drift Detection.

Tracks prediction error over time and raises drift alerts when the
error distribution shifts significantly from the baseline window.

Two detection methods are provided:
  1. sliding_window_mae  — flag when recent-window MAE exceeds threshold × baseline MAE
  2. cusum               — CUSUM chart on signed error for mean-shift detection

Usage
-----
>>> detector = ForecastDriftDetector(baseline_window=30, alert_window=7)
>>> detector.add_observation(store_id=1, product_id=100, date="2024-07-01",
...                          actual=10.5, predicted=9.2)
>>> alerts = detector.get_alerts()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Union

import numpy as np
import pandas as pd


DateLike = Union[str, date, datetime]


@dataclass
class DriftAlert:
    """A single drift-detection alert."""
    store_id: object
    product_id: object
    method: str               # 'sliding_window' or 'cusum'
    detected_at: str          # date string of detection
    baseline_mae: float
    recent_mae: float
    threshold_ratio: float
    message: str

    def to_dict(self) -> dict:
        return {
            "store_id": self.store_id,
            "product_id": self.product_id,
            "method": self.method,
            "detected_at": self.detected_at,
            "baseline_mae": round(self.baseline_mae, 4),
            "recent_mae": round(self.recent_mae, 4),
            "threshold_ratio": round(self.threshold_ratio, 4),
            "message": self.message,
        }


class ForecastDriftDetector:
    """
    Accumulates forecast observations and detects drift.

    Parameters
    ----------
    baseline_window : number of initial observations used to establish baseline MAE
    alert_window    : recent window to compare against baseline
    mae_threshold   : raise alert if recent_MAE / baseline_MAE > mae_threshold
    cusum_k         : CUSUM allowance (slack); in units of baseline σ
    cusum_h         : CUSUM decision interval (threshold); in units of baseline σ
    """

    def __init__(
        self,
        baseline_window: int = 30,
        alert_window: int = 7,
        mae_threshold: float = 1.5,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
    ):
        self.baseline_window = baseline_window
        self.alert_window = alert_window
        self.mae_threshold = mae_threshold
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        # observations: keyed by (store_id, product_id)
        self._obs: dict[tuple, list[dict]] = {}

    # ------------------------------------------------------------------
    # Ingest observations
    # ------------------------------------------------------------------

    def add_observation(
        self,
        store_id: object,
        product_id: object,
        date_key: DateLike,
        actual: float,
        predicted: float,
    ) -> None:
        """Record one actual vs. predicted data point."""
        key = (store_id, product_id)
        if key not in self._obs:
            self._obs[key] = []
        self._obs[key].append({
            "date": str(date_key),
            "actual": float(actual),
            "predicted": float(predicted),
            "error": float(actual) - float(predicted),
        })

    def add_observations_from_df(
        self,
        df: pd.DataFrame,
        store_col: str = "store_id",
        product_col: str = "product_id",
        date_col: str = "date",
        actual_col: str = "actual",
        predicted_col: str = "predicted",
    ) -> None:
        """Bulk-ingest from a DataFrame."""
        for _, row in df.iterrows():
            self.add_observation(
                store_id=row[store_col],
                product_id=row[product_col],
                date_key=row[date_col],
                actual=row[actual_col],
                predicted=row[predicted_col],
            )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _check_sliding_window(self, key: tuple, obs: list[dict]) -> DriftAlert | None:
        n = len(obs)
        if n < self.baseline_window + self.alert_window:
            return None

        errors_baseline = [abs(o["error"]) for o in obs[: self.baseline_window]]
        errors_recent = [abs(o["error"]) for o in obs[-self.alert_window :]]

        mae_base = float(np.mean(errors_baseline))
        mae_recent = float(np.mean(errors_recent))

        if mae_base < 1e-9:
            return None

        ratio = mae_recent / mae_base
        if ratio > self.mae_threshold:
            sid, pid = key
            return DriftAlert(
                store_id=sid,
                product_id=pid,
                method="sliding_window",
                detected_at=obs[-1]["date"],
                baseline_mae=mae_base,
                recent_mae=mae_recent,
                threshold_ratio=ratio,
                message=(
                    f"MAE ratio {ratio:.2f}× exceeds threshold {self.mae_threshold:.1f}×. "
                    f"Baseline MAE={mae_base:.4f}, Recent MAE={mae_recent:.4f}."
                ),
            )
        return None

    def _check_cusum(self, key: tuple, obs: list[dict]) -> DriftAlert | None:
        n = len(obs)
        if n < self.baseline_window + self.alert_window:
            return None

        errors_baseline = np.array([o["error"] for o in obs[: self.baseline_window]])
        mu0 = float(errors_baseline.mean())
        sigma0 = float(errors_baseline.std(ddof=1)) or 1.0

        # CUSUM on remaining observations
        k = self.cusum_k * sigma0
        h = self.cusum_h * sigma0

        S_pos, S_neg = 0.0, 0.0
        for ob in obs[self.baseline_window :]:
            e = ob["error"]
            S_pos = max(0.0, S_pos + (e - mu0) - k)
            S_neg = max(0.0, S_neg - (e - mu0) - k)
            if S_pos > h or S_neg > h:
                # Drift detected
                # Compute a synthetic "recent MAE" for reporting
                recent_errors = [
                    abs(o["error"]) for o in obs[-self.alert_window :]
                ]
                sid, pid = key
                return DriftAlert(
                    store_id=sid,
                    product_id=pid,
                    method="cusum",
                    detected_at=ob["date"],
                    baseline_mae=float(np.mean(np.abs(errors_baseline))),
                    recent_mae=float(np.mean(recent_errors)) if recent_errors else 0.0,
                    threshold_ratio=float(max(S_pos, S_neg) / h),
                    message=(
                        f"CUSUM signal: S+={S_pos:.2f}, S-={S_neg:.2f} "
                        f"(threshold h={h:.2f}). Mean-shift detected from μ₀={mu0:.4f}."
                    ),
                )
        return None

    def get_alerts(self) -> list[DriftAlert]:
        """Run all detectors and return a list of DriftAlert objects."""
        alerts = []
        for key, obs in self._obs.items():
            a1 = self._check_sliding_window(key, obs)
            if a1:
                alerts.append(a1)
            a2 = self._check_cusum(key, obs)
            if a2:
                alerts.append(a2)
        return alerts

    def get_alerts_df(self) -> pd.DataFrame:
        """Return alerts as a DataFrame."""
        alerts = self.get_alerts()
        if not alerts:
            return pd.DataFrame(columns=[
                "store_id", "product_id", "method", "detected_at",
                "baseline_mae", "recent_mae", "threshold_ratio", "message",
            ])
        return pd.DataFrame([a.to_dict() for a in alerts])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return per-(store, product) MAE summary for all tracked pairs."""
        rows = []
        for (sid, pid), obs in self._obs.items():
            if not obs:
                continue
            errors = np.array([o["error"] for o in obs])
            abs_errors = np.abs(errors)
            rows.append({
                "store_id": sid,
                "product_id": pid,
                "n_obs": len(obs),
                "mae": float(abs_errors.mean()),
                "rmse": float(np.sqrt((errors ** 2).mean())),
                "bias": float(errors.mean()),
                "last_date": obs[-1]["date"],
            })
        return pd.DataFrame(rows)
