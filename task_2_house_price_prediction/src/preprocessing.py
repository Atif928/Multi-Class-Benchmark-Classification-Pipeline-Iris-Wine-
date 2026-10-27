"""Leakage-safe preprocessing and train/test splitting.

Every learned transformation lives inside a scikit-learn ``Pipeline`` /
``ColumnTransformer`` so it can be fitted on training data only. The test subset is
created before feature scaling or imputation and is never passed to a transformer's
``fit`` method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import TARGET_COLUMN
from .features import add_engineered_features

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


@dataclass(frozen=True)
class PreparedData:
    """Container for a reproducible, leakage-safe train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    target: str = TARGET_COLUMN
    random_state: int = RANDOM_STATE

    @property
    def all_features(self) -> tuple[str, ...]:
        """Feature names in a stable, documented order."""
        return self.numeric_features + self.categorical_features


def split_features_target(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target after basic validation.

    Raises
    ------
    TypeError
        If ``frame`` is not a DataFrame.
    ValueError
        If the target is missing, duplicated in the feature frame, or unusable.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' is not present.")

    features = frame.drop(columns=[target]).copy()
    target_values = frame[target].copy()

    if target in features.columns:  # defensive; drop above should make this impossible
        raise ValueError("Target leakage detected: target column is still a feature.")
    if target_values.isna().any():
        raise ValueError(f"Target column '{target}' contains missing values.")
    if not is_numeric_dtype(target_values):
        raise ValueError(f"Target column '{target}' must be numeric.")
    finite_target = np.isfinite(target_values.to_numpy(dtype=float, na_value=np.nan))
    if not finite_target.all():
        raise ValueError(f"Target column '{target}' contains non-finite values.")
    if features.empty:
        raise ValueError("No feature columns remain after removing the target.")

    return features, target_values.astype(float)


def train_test_split_regression(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split rows once into reproducible training and test subsets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1.")
    if len(features) != len(target):
        raise ValueError("Features and target must have the same number of rows.")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return X_train, X_test, y_train, y_test


def infer_feature_types(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Programmatically separate numerical and categorical feature columns.

    Boolean, object, string, and pandas categorical columns are treated as
    categorical. Unsupported types (for example datetimes) raise a clear error.

    Raises
    ------
    TypeError
        If a column has an unsupported dtype or if there are no features.
    """
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for column in frame.columns:
        series = frame[column]
        if is_numeric_dtype(series):
            numeric_features.append(str(column))
        elif (
            is_object_dtype(series)
            or is_string_dtype(series)
            or is_bool_dtype(series)
            or is_categorical_dtype(series)
        ):
            categorical_features.append(str(column))
        else:
            raise TypeError(
                f"Unsupported dtype '{series.dtype}' for feature '{column}'. "
                "Convert it to numeric, string, boolean, or categorical."
            )

    if not numeric_features and not categorical_features:
        raise ValueError("No feature columns were found.")
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Build an unfitted preprocessing ``ColumnTransformer``.

    - Numerical: median imputation then ``StandardScaler``.
    - Categorical: most-frequent imputation then one-hot encoding with
      ``handle_unknown="ignore"``.

    All learned statistics are fitted later by the surrounding pipeline on training
    data only.
    """
    categorical_features = list(categorical_features or [])
    numeric_features = list(numeric_features)
    if not numeric_features and not categorical_features:
        raise ValueError("At least one numerical or categorical feature is required.")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float64,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    model: BaseEstimator,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
) -> Pipeline:
    """Attach a model to a fresh preprocessing pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("model", model),
        ]
    )


def prepare_datasets(
    frame: pd.DataFrame,
    *,
    target: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    engineer_features: bool = True,
) -> PreparedData:
    """Split rows first, then add stateless engineered features to each subset.

    The split happens before any preprocessing or engineering. Engineered features
    only use row-wise arithmetic, so computing them separately for train and test
    does not leak information. Feature types are inferred from the training frame.

    Raises
    ------
    ValueError
        If the split is too small to infer feature types or the target is unusable.
    """
    features, target_values = split_features_target(frame, target)
    X_train, X_test, y_train, y_test = train_test_split_regression(
        features,
        target_values,
        test_size=test_size,
        random_state=random_state,
    )

    if engineer_features:
        X_train = add_engineered_features(X_train)
        X_test = add_engineered_features(X_test)

    numeric_features, categorical_features = infer_feature_types(X_train)

    # Feature schemas must match; otherwise the fitted column transformer may
    # silently drop or mis-map test columns.
    missing_in_test = set(X_train.columns).difference(X_test.columns)
    if missing_in_test:
        raise ValueError(
            "Test data is missing training feature columns: "
            f"{sorted(missing_in_test)}"
        )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_features=tuple(numeric_features),
        categorical_features=tuple(categorical_features),
        target=target,
        random_state=random_state,
    )


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return output feature names from a fitted pipeline's preprocessor."""
    if "preprocessor" not in pipeline.named_steps:
        raise ValueError("Pipeline has no 'preprocessor' step.")
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise RuntimeError(
            "Could not retrieve transformed feature names. Fit the pipeline first."
        ) from exc
