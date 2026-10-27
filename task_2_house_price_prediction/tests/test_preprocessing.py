"""Tests for feature typing, splitting, imputation, and preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.data_loader import TARGET_COLUMN
from src.features import ENGINEERED_FEATURE_COLUMNS
from src.preprocessing import (
    build_model_pipeline,
    build_preprocessor,
    get_transformed_feature_names,
    infer_feature_types,
    prepare_datasets,
    split_features_target,
    train_test_split_regression,
)


def test_prepare_datasets_uses_an_80_20_split(prepared_data):
    total = len(prepared_data.X_train) + len(prepared_data.X_test)
    assert len(prepared_data.X_test) / total == pytest.approx(0.20, abs=0.005)
    assert len(prepared_data.y_train) == len(prepared_data.X_train)
    assert len(prepared_data.y_test) == len(prepared_data.X_test)


def test_prepare_datasets_is_reproducible(synthetic_housing_frame):
    first = prepare_datasets(synthetic_housing_frame, random_state=42)
    second = prepare_datasets(synthetic_housing_frame, random_state=42)
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)
    pd.testing.assert_series_equal(first.y_train, second.y_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


def test_prepare_datasets_keeps_target_out_of_features(prepared_data):
    assert TARGET_COLUMN not in prepared_data.X_train.columns
    assert TARGET_COLUMN not in prepared_data.X_test.columns
    assert TARGET_COLUMN not in prepared_data.numeric_features


def test_prepare_datasets_adds_engineered_features_to_both_splits(prepared_data):
    for frame in (prepared_data.X_train, prepared_data.X_test):
        for column in ENGINEERED_FEATURE_COLUMNS:
            assert column in frame.columns
            assert frame[column].notna().all()


def test_prepare_datasets_detects_numeric_and_categorical_columns():
    frame = pd.DataFrame(
        {
            "number": [1.0, 2.0, 3.0, 4.0],
            "category": ["a", "b", "a", "b"],
            "flag": [True, False, True, False],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    prepared = prepare_datasets(
        frame,
        target="target",
        engineer_features=False,
        test_size=0.25,
        random_state=0,
    )
    assert prepared.numeric_features == ("number",)
    assert set(prepared.categorical_features) == {"category", "flag"}


def test_infer_feature_types_rejects_datetime_columns():
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=3),
            "value": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(TypeError, match="Unsupported dtype"):
        infer_feature_types(frame)


def test_split_features_target_rejects_missing_target():
    frame = pd.DataFrame({"value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Target column"):
        split_features_target(frame)


def test_train_test_split_regression_validates_arguments(synthetic_housing_frame):
    features, target = split_features_target(synthetic_housing_frame)
    with pytest.raises(ValueError):
        train_test_split_regression(features, target, test_size=1.0)
    with pytest.raises(ValueError):
        train_test_split_regression(features.iloc[:-1], target)


def test_preprocessor_imputes_missing_numeric_and_categorical_values():
    frame = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, 3.0],
            "cat": ["a", None, "a", "b"],
        }
    )
    preprocessor = build_preprocessor(["num"], ["cat"])
    transformed = preprocessor.fit_transform(frame)

    assert np.isfinite(transformed).all()
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    assert numeric_imputer.statistics_[0] == pytest.approx(3.0)

    categorical_imputer = preprocessor.named_transformers_[
        "categorical"
    ].named_steps["imputer"]
    assert categorical_imputer.statistics_[0] == "a"


def test_scaler_statistics_are_fitted_on_training_data_only():
    train = pd.DataFrame({"num": [0.0, 2.0, 4.0]})
    test = pd.DataFrame({"num": [100.0]})
    preprocessor = build_preprocessor(["num"], [])

    train_scaled = preprocessor.fit_transform(train)
    test_scaled = preprocessor.transform(test)

    assert np.allclose(train_scaled.mean(axis=0), 0.0, atol=1e-12)
    assert test_scaled[0, 0] > 10.0


def test_one_hot_encoder_ignores_unseen_categories():
    train = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    test = pd.DataFrame({"num": [4.0], "cat": ["never_seen"]})
    preprocessor = build_preprocessor(["num"], ["cat"])
    preprocessor.fit(train)

    transformed = preprocessor.transform(test)
    assert np.isfinite(transformed).all()
    assert transformed.shape[0] == 1


def test_fitted_pipeline_preserves_expected_feature_columns(prepared_data):
    pipeline = build_model_pipeline(
        LinearRegression(), prepared_data.numeric_features, prepared_data.categorical_features
    )
    pipeline.fit(prepared_data.X_train, prepared_data.y_train)
    names = get_transformed_feature_names(pipeline)

    for column in prepared_data.X_train.columns:
        assert column in names
    for column in ENGINEERED_FEATURE_COLUMNS:
        assert column in names
    assert TARGET_COLUMN not in names


def test_build_preprocessor_requires_at_least_one_feature():
    with pytest.raises(ValueError, match="At least one"):
        build_preprocessor([], [])
