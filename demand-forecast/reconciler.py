"""
Hierarchical Forecast Reconciliation — MinTrace (OLS).

Reconciles bottom-level (store × product) forecasts to be coherent with
product-level and grand-total aggregates, using the OLS MinTrace method
(Wickramasuriya et al., 2019).

Hierarchy
---------
  Level 0: Grand total (1 series)
  Level 1: Product totals  (P series — sum across stores)
  Level 2: Store×Product   (S×P series — bottom level)

The summation matrix S maps bottom-level series to all levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _build_summation_matrix(
    store_ids: list, product_ids: list
) -> tuple[np.ndarray, list[str]]:
    """
    Build the summation matrix S and return it together with the ordered
    list of all series names (grand-total, product totals, bottom-level).
    """
    n_stores = len(store_ids)
    n_prods = len(product_ids)
    n_bottom = n_stores * n_prods

    s_store_idx = {s: i for i, s in enumerate(store_ids)}
    s_prod_idx = {p: i for i, p in enumerate(product_ids)}

    # Rows: 1 (grand total) + n_prods (product totals) + n_bottom (bottom)
    n_rows = 1 + n_prods + n_bottom
    S = np.zeros((n_rows, n_bottom), dtype=float)

    # Grand total: sum all bottom-level
    S[0, :] = 1.0

    # Product totals: sum across stores for each product
    for prod, pi in s_prod_idx.items():
        for si in range(n_stores):
            col = si * n_prods + pi
            S[1 + pi, col] = 1.0

    # Bottom level: identity block
    S[1 + n_prods :, :] = np.eye(n_bottom)

    # Series labels
    labels = ["grand_total"]
    labels += [f"product_{p}" for p in product_ids]
    labels += [
        f"store_{s}_product_{p}"
        for s in store_ids
        for p in product_ids
    ]
    return S, labels


def reconcile_forecasts(
    base_forecasts: pd.DataFrame,
    method: str = "bottom_up",
) -> pd.DataFrame:
    """
    Reconcile store×product forecasts.

    Parameters
    ----------
    base_forecasts : DataFrame with columns [store_id, product_id, <day+1>, ..., <day+7>]
    method : 'bottom_up' or 'mintrace_ols'

    Returns
    -------
    DataFrame with the same schema plus reconciled forecast columns
    (prefixed ``rec_``) and product-level aggregate rows.
    """
    day_cols = [c for c in base_forecasts.columns if c.startswith("day+")]
    if not day_cols:
        raise ValueError("base_forecasts must contain 'day+N' columns.")

    store_ids = sorted(base_forecasts["store_id"].unique().tolist())
    product_ids = sorted(base_forecasts["product_id"].unique().tolist())

    if method == "bottom_up":
        return _reconcile_bottom_up(base_forecasts, day_cols, store_ids, product_ids)
    elif method == "mintrace_ols":
        return _reconcile_mintrace(base_forecasts, day_cols, store_ids, product_ids)
    else:
        raise ValueError(f"Unknown reconciliation method: {method!r}")


# ── Bottom-up ─────────────────────────────────────────────────────────────────

def _reconcile_bottom_up(
    df: pd.DataFrame,
    day_cols: list[str],
    store_ids: list,
    product_ids: list,
) -> pd.DataFrame:
    """
    Bottom-up: keep bottom-level forecasts unchanged; aggregate upwards.
    Adds product-total rows and a grand-total row.
    """
    result = df.copy()
    # Rename day cols to rec_ prefix (no change for bottom-up)
    for c in day_cols:
        result[f"rec_{c}"] = result[c]

    # Product totals
    prod_totals = (
        df.groupby("product_id")[day_cols]
        .sum()
        .reset_index()
    )
    prod_totals["store_id"] = "ALL"
    prod_totals["level"] = "product_total"
    for c in day_cols:
        prod_totals[f"rec_{c}"] = prod_totals[c]

    # Grand total
    grand = df[day_cols].sum().to_frame().T
    grand["store_id"] = "ALL"
    grand["product_id"] = "ALL"
    grand["level"] = "grand_total"
    for c in day_cols:
        grand[f"rec_{c}"] = grand[c]

    result["level"] = "bottom"
    out = pd.concat([result, prod_totals, grand], ignore_index=True)
    return out


# ── MinTrace OLS ──────────────────────────────────────────────────────────────

def _reconcile_mintrace(
    df: pd.DataFrame,
    day_cols: list[str],
    store_ids: list,
    product_ids: list,
) -> pd.DataFrame:
    """
    OLS MinTrace: ĝ = S (S'S)⁻¹ S' ŷ

    Works day-by-day.  Bottom-level outputs are stored back in the
    original rows; aggregated rows are appended.
    """
    S, labels = _build_summation_matrix(store_ids, product_ids)
    n_bottom = len(store_ids) * len(product_ids)
    St = S.T
    M = S @ np.linalg.lstsq(St @ S, St, rcond=None)[0]  # = S(S'S)^{-1}S'

    # Build a pivot: rows=bottom pairs, cols=day_cols
    pivot = (
        df.set_index(["store_id", "product_id"])[day_cols]
        .reindex(
            pd.MultiIndex.from_product([store_ids, product_ids],
                                       names=["store_id", "product_id"])
        )
        .fillna(0.0)
    )

    # y_hat: shape (n_all_series, n_days)
    bottom_matrix = pivot.values  # (n_bottom, n_days)
    y_hat_all = S @ bottom_matrix  # (n_all, n_days)

    # Reconciled all-level forecasts
    reconciled_all = M @ y_hat_all  # (n_all, n_days) — projection

    # Split back
    reconciled_bottom = reconciled_all[1 + len(product_ids) :, :]  # bottom rows

    # Build output DataFrame
    result = df.copy()
    result["level"] = "bottom"
    # Map reconciled bottom back in same order as pivot index
    bottom_index = pd.MultiIndex.from_product([store_ids, product_ids])
    rec_df = pd.DataFrame(
        reconciled_bottom,
        index=bottom_index,
        columns=[f"rec_{c}" for c in day_cols],
    ).reset_index()
    rec_df.columns = ["store_id", "product_id"] + [f"rec_{c}" for c in day_cols]

    result = result.merge(rec_df, on=["store_id", "product_id"], how="left")

    # Product totals from reconciled
    prod_totals_vals = reconciled_all[1 : 1 + len(product_ids), :]
    prod_rows = []
    for i, pid in enumerate(product_ids):
        row = {"store_id": "ALL", "product_id": pid, "level": "product_total"}
        for j, dc in enumerate(day_cols):
            row[dc] = 0.0
            row[f"rec_{dc}"] = float(prod_totals_vals[i, j])
        prod_rows.append(row)

    grand_vals = reconciled_all[0, :]
    grand_row = {"store_id": "ALL", "product_id": "ALL", "level": "grand_total"}
    for j, dc in enumerate(day_cols):
        grand_row[dc] = 0.0
        grand_row[f"rec_{dc}"] = float(grand_vals[j])

    out = pd.concat(
        [result, pd.DataFrame(prod_rows), pd.DataFrame([grand_row])],
        ignore_index=True,
    )
    return out
