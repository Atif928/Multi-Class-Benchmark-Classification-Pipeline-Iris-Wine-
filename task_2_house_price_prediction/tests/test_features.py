"""Tests for feature engineering, safe division, and VIF analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ENGINEERED_FEATURE_COLUMNS,
    add_engineered_features,
    safe_divide,
    variance_inflation_factors,
)


def test_add_engineered_features_adds_expected_columns(synthetic_housing_frame):
    original = synthetic_housing_frame.copy(deep=True)
    engineered = add_engineered_features(synthetic_housing_frame)

    assert set(ENGINEERED_FEATURE_COLUMNS).issubset(engineered.columns)
    assert len(engineered) == len(synthetic_housing_frame)

    first = synthetic_housing_frame.iloc[0]
    assert engineered.loc[0, "RoomsPerPerson"] == pytest.approx(
        first["AveRooms"] / first["AveOccup"]
    )
    assert engineered.loc[0, "BedroomRatio"] == pytest.approx(
        first["AveBedrms"] / first["AveRooms"]
    )
    assert engineered.loc[0, "IncomePerPerson"] == pytest.approx(
        first["MedInc"] / first["AveOccup"]
    )
    assert engineered.loc[0, "LogPopulation"] == pytest.approx(
        np.log1p(first["Population"])
    )

    # The default is not to mutate the caller's frame.
    pd.testing.assert_frame_equal(synthetic_housing_frame, original)


def test_add_engineered_features_is_idempotent(synthetic_housing_frame):
    once = add_engineered_features(synthetic_housing_frame)
    twice = add_engineered_features(once)
    pd.testing.assert_frame_equal(once, twice)


def test_engineered_features_do_not_use_the_target(synthetic_housing_frame):
    changed_target = synthetic_housing_frame.copy()
    changed_target["MedHouseVal"] = changed_target["MedHouseVal"] * 10

    base = add_engineered_features(synthetic_housing_frame)
    changed = add_engineered_features(changed_target)
    pd.testing.assert_frame_equal(
        base[list(ENGINEERED_FEATURE_COLUMNS)],
        changed[list(ENGINEERED_FEATURE_COLUMNS)],
    )


def test_safe_divide_handles_zero_negative_and_missing_denominators():
    numerator = pd.Series([1.0, 1.0, 1.0, 1.0, np.nan])
    denominator = pd.Series([0.0, 2.0, -4.0, np.nan, 2.0])
    result = safe_divide(numerator, denominator)

    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.5)
    assert result.iloc[2] == pytest.approx(-0.25)
    assert np.isnan(result.iloc[3])
    assert np.isnan(result.iloc[4])


def test_safe_divide_treats_tiny_denominators_as_zero():
    result = safe_divide(pd.Series([5.0]), pd.Series([1e-12]))
    assert np.isnan(result.iloc[0])


def test_safe_divide_rejects_invalid_inputs():
    with pytest.raises(TypeError):
        safe_divide([1.0, 2.0], [1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        safe_divide(pd.Series([1.0]), pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError):
        safe_divide(pd.Series([1.0]), pd.Series([1.0]), epsilon=-1)


def test_add_engineered_features_reports_missing_columns():
    frame = pd.DataFrame({"MedInc": [1.0], "AveRooms": [2.0]})
    with pytest.raises(KeyError, match="required columns are missing"):
        add_engineered_features(frame)


def test_vif_detects_perfect_collinearity():
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x1 + x2})
    vif = variance_inflation_factors(frame)

    assert set(vif.columns) == {"feature", "r_squared", "vif", "severity"}
    assert (vif["vif"] >= 100).all()
    assert (vif["severity"] == "high").all()


def test_vif_is_low_for_independent_features():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=500),
            "b": rng.normal(size=500),
            "c": rng.normal(size=500),
        }
    )
    vif = variance_inflation_factors(frame)
    assert (vif["vif"] < 2.0).all()
    assert (vif["severity"] == "low").all()


def test_vif_rejects_too_few_features():
    with pytest.raises(ValueError):
        variance_inflation_factors(pd.DataFrame({"only": [1.0, 2.0, 3.0]}))
