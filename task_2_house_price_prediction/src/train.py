"""Model training utilities for OLS, Ridge, and Lasso regression.

The module builds complete preprocessing + estimator pipelines, tunes regularized
models on training data only, and provides a small prediction helper whose output
units are explicit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from .data_loader import TARGET_COLUMN, load_california_housing
from .features import REQUIRED_RAW_FEATURES, add_engineered_features
from .preprocessing import (
    CV_FOLDS,
    RANDOM_STATE,
    PreparedData,
    build_model_pipeline,
    prepare_datasets,
)

ALPHA_GRIDS: dict[str, tuple[float, ...]] = {
    "Ridge": (0.01, 0.1, 1.0, 10.0, 100.0),
    "Lasso": (0.001, 0.01, 0.1, 1.0, 10.0),
}
MODEL_ORDER: tuple[str, ...] = ("OLS", "Ridge", "Lasso")
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LASSO_MAX_ITER = 100_000


@dataclass
class ModelResult:
    """A fitted model pipeline plus its training/CV metadata."""

    name: str
    pipeline: Pipeline
    estimator_name: str
    best_alpha: float | None
    cv_rmse: float
    cv_std: float
    cv_folds: int = CV_FOLDS
    convergence_warnings: list[str] = field(default_factory=list)

    @property
    def display_alpha(self) -> str:
        """Human-readable alpha label for reports."""
        return "N/A" if self.best_alpha is None else f"{self.best_alpha:g}"


def make_cv_splitter(
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> KFold:
    """Return the shared shuffled K-fold splitter used for all models."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validated_rmse(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv: KFold | None = None,
    n_jobs: int = -1,
) -> tuple[float, float]:
    """Return mean and standard deviation of fold RMSE on training data.

    Scikit-learn's scoring convention is negative RMSE; the sign is removed before
    reporting so that lower values are better.
    """
    splitter = cv if cv is not None else make_cv_splitter()
    try:
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
            n_jobs=n_jobs,
            error_score="raise",
        )
    except Exception as exc:
        raise RuntimeError(f"Cross-validation failed: {exc}") from exc
    rmse_scores = -np.asarray(scores, dtype=float)
    return float(rmse_scores.mean()), float(rmse_scores.std())


def build_estimator(name: str, alpha: float | None = None) -> BaseEstimator:
    """Create an estimator by name with the project's documented settings."""
    if name == "OLS":
        return LinearRegression()
    if name == "Ridge":
        from sklearn.linear_model import Ridge

        if alpha is None:
            raise ValueError("Ridge requires an alpha value.")
        return Ridge(alpha=float(alpha))
    if name == "Lasso":
        from sklearn.linear_model import Lasso

        if alpha is None:
            raise ValueError("Lasso requires an alpha value.")
        return Lasso(alpha=float(alpha), max_iter=LASSO_MAX_ITER, tol=1e-4)
    raise ValueError(f"Unknown model name '{name}'. Choose from {MODEL_ORDER}.")


def fit_ols(
    prepared: PreparedData,
    *,
    n_jobs: int = -1,
) -> ModelResult:
    """Fit an OLS baseline with cross-validated training RMSE.

    The test set is deliberately not used here.
    """
    pipeline = build_model_pipeline(
        build_estimator("OLS"),
        prepared.numeric_features,
        prepared.categorical_features,
    )
    cv_rmse, cv_std = cross_validated_rmse(
        pipeline,
        prepared.X_train,
        prepared.y_train,
        n_jobs=n_jobs,
    )
    pipeline.fit(prepared.X_train, prepared.y_train)
    return ModelResult(
        name="OLS",
        pipeline=pipeline,
        estimator_name="LinearRegression",
        best_alpha=None,
        cv_rmse=cv_rmse,
        cv_std=cv_std,
    )


def save_models(
    results: Mapping[str, ModelResult],
    output_dir: str | Path = DEFAULT_MODELS_DIR,
) -> dict[str, Path]:
    """Persist fitted pipelines with Joblib and return their paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, result in results.items():
        safe_name = name.lower().replace(" ", "_")
        path = directory / f"{safe_name}_pipeline.joblib"
        joblib.dump(result.pipeline, path)
        paths[name] = path
    return paths


def load_model(path: str | Path) -> Pipeline:
    """Load a Joblib-serialized fitted pipeline."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: '{model_path}'")
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError(f"Object at '{model_path}' is not a fitted estimator.")
    return model


def predict_house_value(
    model: Pipeline,
    features: Mapping[str, float] | pd.DataFrame,
) -> float:
    """Predict one median house value from raw feature inputs.

    Parameters
    ----------
    model:
        A fitted pipeline that includes preprocessing.
    features:
        Mapping or one-row DataFrame containing at least the eight raw California
        Housing features. Engineered features are added automatically, so callers
        may omit them.

    Returns
    -------
    float
        Predicted median house value in **units of $100,000**. Multiply by 100,000
        (or call :func:`predict_house_value_dollars`) for US dollars.

    Raises
    ------
    ValueError
        If required raw features are missing or the prediction is non-finite.
    TypeError
        If ``features`` is not a mapping or DataFrame.
    """
    if isinstance(features, Mapping):
        frame = pd.DataFrame([dict(features)])
    elif isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError("predict_house_value expects exactly one row.")
        frame = features.copy()
    else:
        raise TypeError("features must be a mapping or a one-row pandas DataFrame.")

    missing = [column for column in REQUIRED_RAW_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required raw features: {missing}")

    frame = add_engineered_features(frame)
    prediction = np.asarray(model.predict(frame), dtype=float)
    if prediction.size != 1:
        raise ValueError("Model returned more than one prediction for a single row.")
    value = float(prediction[0])
    if not np.isfinite(value):
        raise ValueError("Model returned a non-finite prediction.")
    return value


def predict_house_value_dollars(
    model: Pipeline,
    features: Mapping[str, float] | pd.DataFrame,
) -> float:
    """Return a prediction converted from $100,000 units to US dollars."""
    return predict_house_value(model, features) * 100_000.0


def main() -> None:
    """Fit the OLS baseline and print training diagnostics."""
    parser = argparse.ArgumentParser(description="Train the OLS housing baseline.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    arguments = parser.parse_args()

    frame = load_california_housing()
    prepared = prepare_datasets(frame)
    result = fit_ols(prepared, n_jobs=arguments.n_jobs)
    train_rmse = float(
        np.sqrt(np.mean((prepared.y_train - result.pipeline.predict(prepared.X_train)) ** 2))
    )
    print(f"Target: {TARGET_COLUMN} (units of $100,000)")
    print(f"Training rows: {len(prepared.X_train):,}")
    print(f"Test rows held out (not used): {len(prepared.X_test):,}")
    print(f"OLS CV RMSE: {result.cv_rmse:.4f} ± {result.cv_std:.4f}")
    print(f"OLS training RMSE: {train_rmse:.4f}")


if __name__ == "__main__":
    main()
