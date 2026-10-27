"""Feature engineering and multicollinearity utilities.

All engineered features are deterministic, row-wise transformations that never use
the target. They can therefore be computed independently for training and test rows
without leaking information.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SAFE_DIVISION_EPSILON = 1e-9

ENGINEERED_FEATURE_COLUMNS: tuple[str, ...] = (
    "RoomsPerPerson",
    "BedroomRatio",
    "IncomePerPerson",
    "LogPopulation",
)

ENGINEERED_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "RoomsPerPerson": (
        "AveRooms / AveOccup. Approximates living space per occupant and separates "
        "crowded households from spacious ones, which the raw averages mix together."
    ),
    "BedroomRatio": (
        "AveBedrms / AveRooms. Captures the bedroom share of a home; a high ratio "
        "suggests smaller units or studios, while a low ratio suggests larger homes."
    ),
    "IncomePerPerson": (
        "MedInc / AveOccup. Scales neighbourhood income by household size and gives "
        "a rough proxy for individual economic capacity."
    ),
    "LogPopulation": (
        "log1p(Population). Compresses the long right tail of block-group population, "
        "reducing the leverage of very large census areas."
    ),
}

REQUIRED_RAW_FEATURES: tuple[str, ...] = (
    "MedInc",
    "AveRooms",
    "AveBedrms",
    "AveOccup",
    "Population",
)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    epsilon: float = SAFE_DIVISION_EPSILON,
    fill_value: float = np.nan,
) -> pd.Series:
    """Divide two numeric Series without raising on zero denominators.

    Denominators with absolute value at or below ``epsilon`` produce
    ``fill_value`` (NaN by default). Non-numeric inputs are coerced to NaN, and
    infinite results are replaced by ``fill_value``.

    Parameters
    ----------
    numerator, denominator:
        Series with equal length. They are aligned on their index.
    epsilon:
        Non-negative tolerance used to treat very small denominators as zero.
    fill_value:
        Value assigned where division is undefined.

    Raises
    ------
    TypeError
        If either input is not a pandas Series.
    ValueError
        If the Series have different lengths or ``epsilon`` is negative.
    """
    if not isinstance(numerator, pd.Series) or not isinstance(denominator, pd.Series):
        raise TypeError("safe_divide expects two pandas Series.")
    if len(numerator) != len(denominator):
        raise ValueError("Numerator and denominator must have the same length.")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    numerator_values = pd.to_numeric(numerator, errors="coerce").astype(float)
    denominator_values = pd.to_numeric(denominator, errors="coerce").astype(float)
    safe_denominator = denominator_values.where(denominator_values.abs() > epsilon)
    result = numerator_values.divide(safe_denominator)
    return result.replace([np.inf, -np.inf], fill_value)


def add_engineered_features(
    frame: pd.DataFrame,
    *,
    inplace: bool = False,
    epsilon: float = SAFE_DIVISION_EPSILON,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with the engineered housing features added.

    The function is idempotent: rerunning it recomputes the engineered columns with
    the same values. It never reads or modifies the target column.

    Parameters
    ----------
    frame:
        DataFrame containing the raw California Housing features.
    inplace:
        If ``True``, modify and return the supplied frame. Defaults to ``False``
        to avoid surprising side effects.
    epsilon:
        Tolerance passed to :func:`safe_divide`.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    KeyError
        If a required raw feature is absent.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("add_engineered_features expects a pandas DataFrame.")

    missing = [column for column in REQUIRED_RAW_FEATURES if column not in frame.columns]
    if missing:
        raise KeyError(
            "Cannot engineer features because required columns are missing: "
            f"{missing}"
        )

    result = frame if inplace else frame.copy()
    result["RoomsPerPerson"] = safe_divide(
        result["AveRooms"], result["AveOccup"], epsilon=epsilon
    )
    result["BedroomRatio"] = safe_divide(
        result["AveBedrms"], result["AveRooms"], epsilon=epsilon
    )
    result["IncomePerPerson"] = safe_divide(
        result["MedInc"], result["AveOccup"], epsilon=epsilon
    )
    result["LogPopulation"] = np.log1p(
        pd.to_numeric(result["Population"], errors="coerce").clip(lower=0)
    )
    return result


def variance_inflation_factors(
    frame: pd.DataFrame,
    *,
    low_threshold: float = 5.0,
    high_threshold: float = 10.0,
) -> pd.DataFrame:
    """Compute variance inflation factors (VIF) for numeric features.

    VIF is calculated without statsmodels by regressing each feature on all other
    features and converting the resulting R²:

    ``VIF_i = 1 / (1 - R²_i)``

    A value near 1 indicates little shared linear variance; values above 5 or 10
    are commonly treated as moderate or high multicollinearity flags. Perfectly
    collinear features produce infinite VIF.

    Parameters
    ----------
    frame:
        Feature frame. Non-numeric columns are ignored.
    low_threshold, high_threshold:
        Thresholds used to label severity.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature``, ``r_squared``, ``vif``, and ``severity``, sorted by VIF
        descending.
    """
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 2:
        raise ValueError("At least two numeric features are required to compute VIF.")
    if len(numeric) < numeric.shape[1] + 1:
        raise ValueError(
            "Not enough observations to compute VIF: "
            f"{len(numeric)} rows for {numeric.shape[1]} features."
        )

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    records: list[dict[str, Any]] = []
    for column in numeric.columns:
        others = numeric.drop(columns=column)
        if numeric[column].nunique() <= 1:
            records.append(
                {
                    "feature": str(column),
                    "r_squared": np.nan,
                    "vif": np.inf,
                    "severity": "undefined (constant feature)",
                }
            )
            continue
        model = LinearRegression()
        model.fit(others, numeric[column])
        r_squared = float(model.score(others, numeric[column]))
        if r_squared >= 1.0:
            vif = np.inf
        else:
            vif = 1.0 / max(1.0 - r_squared, np.finfo(float).eps)
        if vif >= high_threshold:
            severity = "high"
        elif vif >= low_threshold:
            severity = "moderate"
        else:
            severity = "low"
        records.append(
            {
                "feature": str(column),
                "r_squared": r_squared,
                "vif": float(vif),
                "severity": severity,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("vif", ascending=False, na_position="first")
        .reset_index(drop=True)
    )
