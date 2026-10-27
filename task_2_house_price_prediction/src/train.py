"""Model training utilities for OLS, Ridge, and Lasso regression.

The module builds complete preprocessing + estimator pipelines, tunes regularized
models on training data only, and provides a small prediction helper whose output
units are explicit.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
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


def tune_regularized_model(
    prepared: PreparedData,
    name: str,
    *,
    alpha_grid: Sequence[float] | None = None,
    cv: KFold | None = None,
    n_jobs: int = -1,
) -> ModelResult:
    """Tune Ridge or Lasso with ``GridSearchCV`` on training data only.

    The search uses shuffled five-fold CV and negative root mean squared error.
    Convergence warnings are captured and returned with the result so callers can
    report them rather than silently hiding numerical issues.

    Parameters
    ----------
    prepared:
        Leakage-safe training/test split.
    name:
        ``"Ridge"`` or ``"Lasso"``.
    alpha_grid:
        Candidate regularization strengths. Defaults to :data:`ALPHA_GRIDS`.
    cv:
        Optional splitter. Defaults to :func:`make_cv_splitter`.
    n_jobs:
        Number of parallel jobs passed to ``GridSearchCV``.
    """
    if name not in ("Ridge", "Lasso"):
        raise ValueError("name must be 'Ridge' or 'Lasso'.")
    alphas = tuple(alpha_grid if alpha_grid is not None else ALPHA_GRIDS[name])
    if not alphas:
        raise ValueError("alpha_grid must contain at least one value.")
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("All alpha values must be strictly positive.")

    splitter = cv if cv is not None else make_cv_splitter()
    pipeline = build_model_pipeline(
        build_estimator(name, alpha=float(alphas[0])),
        prepared.numeric_features,
        prepared.categorical_features,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"model__alpha": list(alphas)},
        scoring="neg_root_mean_squared_error",
        cv=splitter,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )

    convergence_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            search.fit(prepared.X_train, prepared.y_train)
        except Exception as exc:
            raise RuntimeError(f"{name} grid search failed: {exc}") from exc
        convergence_messages.extend(
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        )

    best_estimator = search.best_estimator_.named_steps["model"]
    best_iterations = getattr(best_estimator, "n_iter_", None)
    if (
        name == "Lasso"
        and best_iterations is not None
        and int(np.max(best_iterations)) >= LASSO_MAX_ITER
    ):
        convergence_messages.append(
            f"Lasso reached max_iter={LASSO_MAX_ITER}; results may be unstable."
        )

    best_alpha = float(search.best_params_["model__alpha"])
    cv_rmse = -float(search.best_score_)
    cv_std = float(search.cv_results_["std_test_score"][search.best_index_])

    return ModelResult(
        name=name,
        pipeline=search.best_estimator_,
        estimator_name=name,
        best_alpha=best_alpha,
        cv_rmse=cv_rmse,
        cv_std=cv_std,
        convergence_warnings=convergence_messages,
    )


def train_all_models(
    prepared: PreparedData,
    *,
    alpha_grids: Mapping[str, Sequence[float]] | None = None,
    n_jobs: int = -1,
) -> dict[str, ModelResult]:
    """Fit OLS, tune Ridge, and tune Lasso on the training split.

    The test rows in ``prepared`` are not passed to any estimator or search.
    """
    grids = dict(alpha_grids or ALPHA_GRIDS)
    results: dict[str, ModelResult] = {
        "OLS": fit_ols(prepared, n_jobs=n_jobs),
    }
    for name in ("Ridge", "Lasso"):
        results[name] = tune_regularized_model(
            prepared,
            name,
            alpha_grid=grids.get(name),
            n_jobs=n_jobs,
        )
    return results


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
    """Train all three models and print cross-validated training diagnostics."""
    parser = argparse.ArgumentParser(
        description="Tune OLS, Ridge, and Lasso on the training split."
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    arguments = parser.parse_args()

    frame = load_california_housing()
    prepared = prepare_datasets(frame)
    results = train_all_models(prepared, n_jobs=arguments.n_jobs)

    print(f"Target: {TARGET_COLUMN} (units of $100,000)")
    print(f"Training rows: {len(prepared.X_train):,}")
    print(f"Test rows held out (not used): {len(prepared.X_test):,}")
    print()
    print(f"{'Model':<7} {'Best alpha':>11} {'CV RMSE':>10} {'CV std':>8}")
    for name in MODEL_ORDER:
        result = results[name]
        print(
            f"{result.name:<7} {result.display_alpha:>11} "
            f"{result.cv_rmse:>10.4f} {result.cv_std:>8.4f}"
        )
        for message in result.convergence_warnings:
            print(f"  warning: {message}")


if __name__ == "__main__":
    main()
