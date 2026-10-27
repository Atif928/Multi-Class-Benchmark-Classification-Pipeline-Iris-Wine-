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
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from .data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_california_housing
from .features import add_engineered_features
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

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
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


def _json_default(value: object) -> object:
    """Convert NumPy and Path values for JSON serialization."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def run_pipeline(
    *,
    project_root: str | Path | None = None,
    cache_path: str | Path | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Run the full reproducible workflow and write all deliverables.

    Steps: load/validate → EDA figures → split → feature engineering → VIF →
    train/tune → test evaluation → residual/coefficient figures → reports →
    model artifacts → sample prediction → run metadata.
    """
    import json
    import platform
    import sys
    from datetime import datetime, timezone

    import joblib
    import matplotlib
    import seaborn
    import sklearn

    from .data_loader import summarize_dataset, summary_to_markdown, validate_dataset
    from .evaluate import evaluate_results, write_evaluation_reports
    from .features import variance_inflation_factors
    from .visualize import (
        generate_diagnostic_figures,
        generate_eda_figures,
        plot_vif_analysis,
        save_figure,
        set_plot_style,
    )

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    figure_dir = root / "outputs" / "figures"
    report_dir = root / "outputs" / "reports"
    model_dir = root / "models"
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    frame = load_california_housing(cache_path=cache_path)
    validation = validate_dataset(frame)
    summary = summarize_dataset(frame)

    (report_dir / "dataset_summary.json").write_text(
        json.dumps(
            {"validation": validation, "summary": summary},
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (report_dir / "dataset_summary.md").write_text(
        summary_to_markdown(validation, summary), encoding="utf-8"
    )

    figure_paths = generate_eda_figures(frame, figure_dir)

    prepared = prepare_datasets(frame)
    vif_frame = variance_inflation_factors(prepared.X_train)
    vif_path = report_dir / "vif_analysis.csv"
    vif_frame.to_csv(vif_path, index=False)
    set_plot_style()
    figure_paths["vif_analysis"] = save_figure(
        plot_vif_analysis(vif_frame), figure_dir / "07_vif_analysis.png"
    )

    results = train_all_models(prepared, n_jobs=n_jobs)
    evaluation = evaluate_results(results, prepared.X_test, prepared.y_test)
    report_paths = write_evaluation_reports(evaluation, report_dir)
    report_paths["vif_csv"] = vif_path
    report_paths["dataset_summary_json"] = report_dir / "dataset_summary.json"
    report_paths["dataset_summary_md"] = report_dir / "dataset_summary.md"

    figure_paths.update(generate_diagnostic_figures(evaluation, figure_dir))
    model_paths = save_models(results, model_dir)

    sample_row = prepared.X_test.iloc[0]
    sample_features = {
        column: float(sample_row[column]) for column in FEATURE_COLUMNS
    }
    sample_actual = float(prepared.y_test.iloc[0])
    sample_predictions: dict[str, Any] = {}
    for name, result in results.items():
        predicted = predict_house_value(result.pipeline, sample_features)
        sample_predictions[name] = {
            "predicted_med_house_val": predicted,
            "predicted_usd": predicted * 100_000.0,
            "absolute_error_100k": abs(sample_actual - predicted),
            "absolute_error_usd": abs(sample_actual - predicted) * 100_000.0,
        }
    sample_path = report_dir / "sample_prediction.json"
    sample_path.write_text(
        json.dumps(
            {
                "sample_index": int(prepared.X_test.index[0]),
                "split": "test",
                "input_features": sample_features,
                "actual_med_house_val": sample_actual,
                "actual_usd": sample_actual * 100_000.0,
                "predictions": sample_predictions,
                "unit_note": "MedHouseVal is in units of $100,000; usd fields multiply by 100,000.",
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    report_paths["sample_prediction"] = sample_path

    best_cv_model = min(results.values(), key=lambda result: result.cv_rmse)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "California Housing (sklearn.datasets.fetch_california_housing)",
        "target": TARGET_COLUMN,
        "target_unit": "$100,000",
        "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "cv_folds": CV_FOLDS,
        "cv_shuffle": True,
        "scoring": "neg_root_mean_squared_error",
        "alpha_grids": {name: list(values) for name, values in ALPHA_GRIDS.items()},
        "selected_alphas": {
            name: result.best_alpha for name, result in results.items()
        },
        "cv_rmse": {name: result.cv_rmse for name, result in results.items()},
        "convergence_warnings": {
            name: result.convergence_warnings
            for name, result in results.items()
            if result.convergence_warnings
        },
        "best_cv_model": best_cv_model.name,
        "lasso_max_iter": LASSO_MAX_ITER,
        "train_rows": int(len(prepared.X_train)),
        "test_rows": int(len(prepared.X_test)),
        "feature_columns": list(prepared.all_features),
        "engineered_features": [
            column
            for column in prepared.X_train.columns
            if column not in frame.columns
        ],
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__,
            "joblib": joblib.__version__,
        },
        "model_artifacts": {
            name: path.resolve().relative_to(root).as_posix()
            for name, path in model_paths.items()
        },
    }
    metadata_path = report_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    report_paths["run_metadata"] = metadata_path

    return {
        "frame": frame,
        "validation": validation,
        "summary": summary,
        "prepared": prepared,
        "results": results,
        "evaluation": evaluation,
        "figure_paths": figure_paths,
        "report_paths": report_paths,
        "model_paths": model_paths,
    }


def main() -> None:
    """Run the full training and evaluation pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Train, tune, evaluate, and report the California housing models."
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Optional CSV cache for the dataset (read if present, written if downloaded).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root for outputs; defaults to the task_2 directory.",
    )
    arguments = parser.parse_args()

    artifacts = run_pipeline(
        project_root=arguments.project_root,
        cache_path=arguments.cache_path,
        n_jobs=arguments.n_jobs,
    )
    prepared = artifacts["prepared"]
    results = artifacts["results"]
    evaluation = artifacts["evaluation"]

    print(f"Target: {TARGET_COLUMN} (units of $100,000)")
    print(f"Training rows: {len(prepared.X_train):,}")
    print(f"Test rows: {len(prepared.X_test):,}")
    print()
    print(evaluation.metrics_markdown())
    print()
    for name in MODEL_ORDER:
        result = results[name]
        print(
            f"{name}: selected alpha={result.display_alpha}, "
            f"CV RMSE={result.cv_rmse:.4f} ± {result.cv_std:.4f}"
        )
        for message in result.convergence_warnings:
            print(f"  warning: {message}")
    print()
    print(f"Figures written to: {artifacts['figure_paths']['target_distribution'].parent}")
    print(f"Reports written to: {artifacts['report_paths']['metrics_csv'].parent}")
    print(f"Models written to: {next(iter(artifacts['model_paths'].values())).parent}")


if __name__ == "__main__":
    main()
