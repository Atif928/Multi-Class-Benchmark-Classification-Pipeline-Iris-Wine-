"""Load, validate, and summarize the California Housing dataset.

The canonical source is :func:`sklearn.datasets.fetch_california_housing`. The
loader can optionally mirror the downloaded frame to a local CSV so later runs do
not need network access. Downloaded data is ignored by Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.datasets import fetch_california_housing

TARGET_COLUMN = "MedHouseVal"
TARGET_UNIT = "$100,000"
TARGET_UNIT_LONG = "hundreds of thousands of US dollars"
DATASET_NAME = "California Housing"
MIN_OBSERVATIONS = 100

FEATURE_COLUMNS: tuple[str, ...] = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be used for training or evaluation."""


def load_california_housing(
    *,
    cache_path: str | Path | None = None,
    download_if_missing: bool = True,
    data_home: str | Path | None = None,
) -> pd.DataFrame:
    """Load the California Housing dataset as a pandas ``DataFrame``.

    Parameters
    ----------
    cache_path:
        Optional CSV path. If the file exists it is read instead of downloading.
        After a successful download the frame is written there, creating parent
        directories as needed.
    download_if_missing:
        Passed to scikit-learn. When ``False``, scikit-learn raises if no cached
        copy is available.
    data_home:
        Optional scikit-learn cache directory.

    Returns
    -------
    pandas.DataFrame
        The eight input features plus the ``MedHouseVal`` target. The target is
        measured in units of **$100,000**.

    Raises
    ------
    RuntimeError
        If the dataset cannot be read or downloaded, with a concise hint about
        network access and cache locations.
    """
    cache = Path(cache_path).expanduser() if cache_path is not None else None

    if cache is not None and cache.is_file():
        try:
            frame = pd.read_csv(cache)
        except Exception as exc:  # pragma: no cover - corrupt cache is unusual
            raise RuntimeError(
                f"Could not read the cached dataset at '{cache}'. "
                "Delete the file and rerun so it can be downloaded again."
            ) from exc
        if TARGET_COLUMN not in frame.columns:
            raise RuntimeError(
                f"Cached dataset at '{cache}' does not contain the target column "
                f"'{TARGET_COLUMN}'. Delete the file and rerun."
            )
        return frame

    try:
        bunch = fetch_california_housing(
            as_frame=True,
            download_if_missing=download_if_missing,
            data_home=str(data_home) if data_home is not None else None,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the California Housing dataset from scikit-learn. "
            "Check your internet connection (the first run downloads the data) "
            "or provide a valid local CSV through the 'cache_path' argument. "
            f"Original error: {exc}"
        ) from exc

    frame = bunch.frame.copy()
    if TARGET_COLUMN not in frame.columns:
        raise RuntimeError(
            "scikit-learn returned a California Housing frame without the expected "
            f"target column '{TARGET_COLUMN}'. Check the scikit-learn installation."
        )

    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache, index=False)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            raise RuntimeError(
                f"The dataset loaded successfully but could not be cached at "
                f"'{cache}'. Check write permissions or omit 'cache_path'. "
                f"Original error: {exc}"
            ) from exc

    return frame


def validate_dataset(
    frame: pd.DataFrame,
    *,
    target: str = TARGET_COLUMN,
    min_observations: int = MIN_OBSERVATIONS,
) -> dict[str, Any]:
    """Validate a housing dataset and return a machine-readable report.

    Checks include target presence and numeric type, minimum row count, target
    validity, missing values, infinite values, duplicate rows, and non-numeric
    feature columns. Missing feature values are reported but are not fatal because
    the preprocessing pipeline imputes them. Missing or invalid *target* values are
    fatal because those rows cannot be used for supervised learning.

    Raises
    ------
    TypeError
        If ``frame`` is not a pandas ``DataFrame``.
    DatasetValidationError
        If one or more blocking data-quality issues are found.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame, got " f"{type(frame).__name__}.")

    errors: list[str] = []
    warnings: list[str] = []

    if frame.empty:
        raise DatasetValidationError("The dataset is empty (0 rows).")
    if target not in frame.columns:
        raise DatasetValidationError(
            f"Target column '{target}' is missing. Available columns: "
            f"{list(frame.columns)}"
        )
    if not is_numeric_dtype(frame[target]):
        raise DatasetValidationError(
            f"Target column '{target}' must be numeric, got dtype "
            f"'{frame[target].dtype}'."
        )

    n_rows = int(len(frame))
    if n_rows < min_observations:
        errors.append(
            f"Only {n_rows} observations are available; at least "
            f"{min_observations} are required."
        )

    target_values = pd.to_numeric(frame[target], errors="coerce")
    target_missing = int(target_values.isna().sum())
    target_non_finite = int(
        np.count_nonzero(
            ~np.isfinite(target_values.to_numpy(dtype=float, na_value=np.nan))
        )
    ) - target_missing
    target_negative = int((target_values < 0).sum())
    valid_target_rows = int(
        np.count_nonzero(
            np.isfinite(target_values.to_numpy(dtype=float, na_value=np.nan))
            & (target_values.to_numpy(dtype=float, na_value=np.nan) >= 0)
        )
    )

    if target_missing:
        errors.append(f"Target column '{target}' contains {target_missing} missing values.")
    if target_non_finite:
        errors.append(
            f"Target column '{target}' contains {target_non_finite} infinite values."
        )
    if target_negative:
        errors.append(
            f"Target column '{target}' contains {target_negative} negative values."
        )
    if valid_target_rows < min_observations:
        errors.append(
            f"Only {valid_target_rows} rows have a valid target; at least "
            f"{min_observations} are required."
        )
    if target_values.nunique(dropna=True) < 2:
        errors.append("The target is constant; regression cannot be fitted.")

    feature_columns = [column for column in frame.columns if column != target]
    if not feature_columns:
        errors.append("No feature columns are available.")

    non_numeric_features = [
        column
        for column in feature_columns
        if not is_numeric_dtype(frame[column])
    ]
    missing_by_column = {
        str(column): int(count)
        for column, count in frame.isna().sum().items()
        if count
    }

    infinite_by_column: dict[str, int] = {}
    for column in frame.columns:
        if is_numeric_dtype(frame[column]):
            values = frame[column].to_numpy(dtype=float, na_value=np.nan)
            count = int(np.count_nonzero(np.isinf(values)))
            if count:
                infinite_by_column[str(column)] = count

    if infinite_by_column:
        errors.append(
            "Infinite values found in: "
            + ", ".join(f"{name} ({count})" for name, count in infinite_by_column.items())
        )

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(
            f"{duplicate_rows} duplicate rows detected. They are kept because "
            "duplicate census records may be legitimate; consider reviewing them."
        )

    report: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "shape": {"rows": n_rows, "columns": int(frame.shape[1])},
        "target_column": target,
        "target_unit": TARGET_UNIT,
        "feature_columns": feature_columns,
        "column_dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {
            "total_missing_cells": int(frame.isna().sum().sum()),
            "columns_with_missing": missing_by_column,
        },
        "infinite_values": infinite_by_column,
        "duplicate_rows": duplicate_rows,
        "valid_target_rows": valid_target_rows,
        "target_summary": {
            "count": int(target_values.count()),
            "mean": float(target_values.mean()),
            "std": float(target_values.std()),
            "min": float(target_values.min()),
            "median": float(target_values.median()),
            "max": float(target_values.max()),
        },
        "categorical_feature_columns": non_numeric_features,
    }

    if errors:
        raise DatasetValidationError(
            "Dataset validation failed:\n- " + "\n- ".join(errors)
        )
    return report


def summarize_dataset(
    frame: pd.DataFrame,
    *,
    target: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Return descriptive statistics for the dataset and target distribution."""
    if target not in frame.columns:
        raise DatasetValidationError(f"Target column '{target}' is not present.")

    numeric_frame = frame.select_dtypes(include=[np.number])
    target_values = frame[target].astype(float)

    feature_statistics: dict[str, dict[str, float]] = {}
    for column in numeric_frame.columns:
        if column == target:
            continue
        series = numeric_frame[column]
        feature_statistics[str(column)] = {
            "count": float(series.count()),
            "missing": int(series.isna().sum()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(0.75)),
            "max": float(series.max()),
        }

    # The public dataset caps the target near 5.00001; report the share at the cap
    # so the limitation is visible rather than silently ignored.
    cap_threshold = float(target_values.max()) - 1e-6
    capped_count = int((target_values >= cap_threshold).sum())

    return {
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "columns": [str(column) for column in frame.columns],
        "feature_statistics": feature_statistics,
        "target": {
            "name": target,
            "unit": TARGET_UNIT,
            "count": int(target_values.count()),
            "mean": float(target_values.mean()),
            "std": float(target_values.std()),
            "min": float(target_values.min()),
            "q25": float(target_values.quantile(0.25)),
            "median": float(target_values.median()),
            "q75": float(target_values.quantile(0.75)),
            "max": float(target_values.max()),
            "capped_count": capped_count,
            "capped_fraction": float(capped_count / len(target_values)),
        },
    }


def summary_to_markdown(report: dict[str, Any], summary: dict[str, Any]) -> str:
    """Render validation and summary dictionaries as a Markdown report."""
    target = summary["target"]
    missing = report["missing_values"]
    lines = [
        "# Dataset Summary — California Housing",
        "",
        "## Validation",
        "",
        f"- Status: **{'PASSED' if report['is_valid'] else 'FAILED'}**",
        f"- Rows: {report['shape']['rows']:,}",
        f"- Columns: {report['shape']['columns']}",
        f"- Feature columns: {len(report['feature_columns'])}",
        f"- Duplicate rows: {report['duplicate_rows']}",
        f"- Missing cells: {missing['total_missing_cells']}",
        f"- Infinite values: {sum(report['infinite_values'].values())}",
        f"- Categorical feature columns: {len(report['categorical_feature_columns'])}",
        "",
        "## Target",
        "",
        f"- Column: `{target['name']}`",
        f"- Unit: **{target['unit']}**",
        f"- Mean: {target['mean']:.4f} ({target['mean'] * 100_000:,.0f} USD)",
        f"- Median: {target['median']:.4f} ({target['median'] * 100_000:,.0f} USD)",
        f"- Standard deviation: {target['std']:.4f}",
        f"- Minimum: {target['min']:.4f}",
        f"- Maximum: {target['max']:.4f}",
        f"- Rows at the upper cap: {target['capped_count']:,} "
        f"({target['capped_fraction']:.2%})",
        "",
        "## Feature statistics",
        "",
        "| Feature | Mean | Std | Min | Median | Max | Missing |",
        "| ------- | ---: | --: | --: | -----: | --: | ------: |",
    ]
    for name, stats in summary["feature_statistics"].items():
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
            f"{stats['min']:.4f} | {stats['median']:.4f} | "
            f"{stats['max']:.4f} | {stats['missing']} |"
        )

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def main() -> None:
    """Download/load the dataset and print a concise validation summary."""
    frame = load_california_housing()
    report = validate_dataset(frame)
    summary = summarize_dataset(frame)
    print(f"{DATASET_NAME}: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"Target: {TARGET_COLUMN} (unit: {TARGET_UNIT})")
    print(f"Missing cells: {report['missing_values']['total_missing_cells']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(
        "Target mean/median/max: "
        f"{summary['target']['mean']:.4f} / "
        f"{summary['target']['median']:.4f} / "
        f"{summary['target']['max']:.4f}"
    )
    print(json.dumps(report, indent=2)[:1500])


if __name__ == "__main__":
    main()
