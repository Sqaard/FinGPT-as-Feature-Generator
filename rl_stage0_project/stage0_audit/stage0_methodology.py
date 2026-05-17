from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRAIN_START = pd.Timestamp("2010-01-01")
TRAIN_END = pd.Timestamp("2021-10-01")
FROZEN_TEST_START = pd.Timestamp("2022-01-03")
FROZEN_TEST_END = pd.Timestamp("2023-03-01")

REQUIRED_PANEL_COLUMNS = ["date", "tic", "close"]

FILTERED_WITH_GRU_FEATURES = [
    "Regime_1_Prob",
    "SP500_Trend",
    "forecast_std",
    "gru_return_forecast_1d",
    "gru_return_forecast_5d",
    "turbulence",
    "10Y_Yield",
    "VIX",
    "atr_rel",
    "daily_return",
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "volume_ratio",
    "obv_pct_change",
    "PE_ratio",
    "PB_ratio",
    "dividend_yield",
    "debt_ratio",
    "revenue_growth",
    "EV_EBITDA",
]

INTERPRETABLE_NO_GRU_FEATURES = [
    feature
    for feature in FILTERED_WITH_GRU_FEATURES
    if not feature.startswith("gru_return_forecast_") and feature != "forecast_std"
]


@dataclass(frozen=True)
class WalkForwardFold:
    fold: str
    train_start: str
    train_end_inclusive: str
    validation_start: str
    validation_end_inclusive: str


DEFAULT_ANCHORED_FOLDS = [
    WalkForwardFold("fold_2018", "2010-01-04", "2017-12-29", "2018-01-02", "2018-12-31"),
    WalkForwardFold("fold_2019", "2010-01-04", "2018-12-31", "2019-01-02", "2019-12-31"),
    WalkForwardFold("fold_2020", "2010-01-04", "2019-12-31", "2020-01-02", "2020-12-31"),
    WalkForwardFold("fold_2021", "2010-01-04", "2020-12-31", "2021-01-04", "2021-12-31"),
]


def assert_panel_integrity(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_PANEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if any(col.startswith("Unnamed") for col in df.columns):
        raise ValueError("Dataset contains an exported index column starting with 'Unnamed'.")

    if df[REQUIRED_PANEL_COLUMNS].isna().any().any():
        raise ValueError("Required panel columns contain missing values.")

    if df.duplicated(["date", "tic"]).any():
        raise ValueError("Dataset contains duplicate date/tic rows.")

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty and np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Dataset contains infinite numeric values.")


def next_day_return_by_ticker(df: pd.DataFrame) -> pd.Series:
    sorted_df = df.sort_values(["tic", "date"])
    return sorted_df.groupby("tic")["close"].shift(-1) / sorted_df["close"] - 1.0


def rank_features_train_only(
    df: pd.DataFrame,
    candidate_features: Iterable[str],
    train_start: str | pd.Timestamp = TRAIN_START,
    train_end_exclusive: str | pd.Timestamp = TRAIN_END,
) -> pd.DataFrame:
    """Rank candidate features against next-day return using only the train window."""
    assert_panel_integrity(df[REQUIRED_PANEL_COLUMNS].copy())
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["tic", "date"]).reset_index(drop=True)
    work["next_day_return"] = work.groupby("tic")["close"].shift(-1) / work["close"] - 1.0

    train_mask = (work["date"] >= pd.Timestamp(train_start)) & (
        work["date"] < pd.Timestamp(train_end_exclusive)
    )
    train = work.loc[train_mask]

    rows = []
    for feature in candidate_features:
        if feature not in work.columns:
            rows.append(
                {
                    "feature": feature,
                    "abs_train_corr": np.nan,
                    "train_corr": np.nan,
                    "status": "missing",
                }
            )
            continue
        corr = train[feature].corr(train["next_day_return"])
        rows.append(
            {
                "feature": feature,
                "abs_train_corr": abs(corr) if pd.notna(corr) else np.nan,
                "train_corr": corr,
                "status": "ok" if pd.notna(corr) else "constant_or_nan",
            }
        )
    return pd.DataFrame(rows).sort_values("abs_train_corr", ascending=False)


def export_feature_panel(df: pd.DataFrame, features: Iterable[str], output_path: Path) -> pd.DataFrame:
    """Write a FinRL-compatible panel with explicit date/tic/close columns and no index."""
    columns = REQUIRED_PANEL_COLUMNS + list(features)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot export feature panel; missing columns: {missing}")

    out = df.loc[:, columns].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    assert_panel_integrity(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def walk_forward_folds_dataframe(
    folds: Iterable[WalkForwardFold] = DEFAULT_ANCHORED_FOLDS,
) -> pd.DataFrame:
    return pd.DataFrame([fold.__dict__ for fold in folds])


def walk_forward_selection_score(
    fold_results: pd.DataFrame,
    sharpe_col: str = "validation_sharpe",
    group_cols: Iterable[str] = ("config_name", "checkpoint_step", "feature_set"),
    stability_penalty: float = 0.5,
) -> pd.DataFrame:
    """Compute mean Sharpe minus a cross-fold stability penalty."""
    required = set(group_cols) | {sharpe_col}
    missing = [col for col in required if col not in fold_results.columns]
    if missing:
        raise ValueError(f"Missing walk-forward result columns: {missing}")

    grouped = (
        fold_results.groupby(list(group_cols), dropna=False)[sharpe_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_validation_sharpe",
                "std": "std_validation_sharpe",
                "count": "fold_count",
            }
        )
    )
    grouped["std_validation_sharpe"] = grouped["std_validation_sharpe"].fillna(0.0)
    grouped["selection_score"] = grouped["mean_validation_sharpe"] - (
        stability_penalty * grouped["std_validation_sharpe"]
    )
    return grouped.sort_values("selection_score", ascending=False)


def assert_threshold_policy_consistent(
    train_threshold: float | None,
    validation_threshold: float | None,
    test_threshold: float | None,
) -> None:
    values = {train_threshold, validation_threshold, test_threshold}
    if len(values) != 1:
        raise ValueError(
            "Threshold policy is inconsistent. Train, validation, and test must all use "
            "the same turbulence threshold, or all use None."
        )


def final_acceptance_gate(summary: pd.DataFrame) -> pd.DataFrame:
    """Evaluate final frozen-test acceptance against DJI and equal-weight benchmarks."""
    required = {"strategy_type", "return_pct", "sharpe_ratio"}
    missing = [col for col in required if col not in summary.columns]
    if missing:
        raise ValueError(f"Missing final-test summary columns: {missing}")

    rl = summary[summary["strategy_type"] == "RL"].copy()
    benchmarks = summary[summary["strategy_type"] == "Benchmark"].copy()
    if benchmarks.empty:
        raise ValueError("No benchmark rows found in final-test summary.")

    best_benchmark_return = benchmarks["return_pct"].max()
    best_benchmark_sharpe = benchmarks["sharpe_ratio"].max()

    rl["positive_return"] = rl["return_pct"] > 0
    rl["positive_sharpe"] = rl["sharpe_ratio"] > 0
    rl["beats_best_benchmark_return"] = rl["return_pct"] > best_benchmark_return
    rl["beats_best_benchmark_sharpe"] = rl["sharpe_ratio"] > best_benchmark_sharpe
    rl["passes_acceptance_gate"] = (
        rl["positive_return"]
        & rl["positive_sharpe"]
        & rl["beats_best_benchmark_return"]
        & rl["beats_best_benchmark_sharpe"]
    )
    return rl
