"""Evaluation, residual summaries, and report writing for the housing models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:  # scikit-learn >= 1.4
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - compatibility path for older versions
    root_mean_squared_error = None  # type: ignore[assignment]

from .data_loader import TARGET_COLUMN, TARGET_UNIT
from .preprocessing import get_transformed_feature_names

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CORE_METRIC_COLUMNS: tuple[str, ...] = (
    "Model",
    "Best Alpha",
    "CV RMSE",
    "Test RMSE",
    "Test MAE",
    "Test R²",
)


@dataclass
class EvaluationResults:
    """Metrics, predictions, residuals, and coefficients for fitted models."""

    metrics: pd.DataFrame
    residual_summary: pd.DataFrame
    coefficients: pd.DataFrame
    predictions: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]
    intercepts: dict[str, float]

    def metrics_markdown(self) -> str:
        """Render the required comparison table as Markdown."""
        columns = [column for column in CORE_METRIC_COLUMNS if column in self.metrics]
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(
            "---:" if column not in {"Model", "Best Alpha"} else "---"
            for column in columns
        ) + " |"
        rows = []
        for _, row in self.metrics.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, (float, np.floating)):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *rows])


def _dataframe_to_markdown(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a small DataFrame as a Markdown table without extra dependencies."""
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows: list[str] = []
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE across scikit-learn versions."""
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _validate_vectors(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true_values = np.asarray(y_true, dtype=float).ravel()
    predicted_values = np.asarray(y_pred, dtype=float).ravel()
    if true_values.size == 0:
        raise ValueError("Cannot evaluate empty arrays.")
    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "y_true and y_pred must have the same length, got "
            f"{true_values.size} and {predicted_values.size}."
        )
    if not np.isfinite(true_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Evaluation arrays must contain only finite values.")
    return true_values, predicted_values


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for regression predictions.

    Returns
    -------
    dict
        Keys ``rmse``, ``mae``, ``r2``, and ``n_samples``. RMSE and MAE inherit the
        target unit ($100,000); R² is unitless.
    """
    true_values, predicted_values = _validate_vectors(y_true, y_pred)
    return {
        "rmse": _rmse(true_values, predicted_values),
        "mae": float(mean_absolute_error(true_values, predicted_values)),
        "r2": float(r2_score(true_values, predicted_values)),
        "n_samples": float(true_values.size),
    }


def _extract_coefficients(
    name: str,
    pipeline: Any,
) -> tuple[list[dict[str, Any]], float]:
    feature_names = get_transformed_feature_names(pipeline)
    estimator = pipeline.named_steps["model"]
    coefficients = np.asarray(estimator.coef_, dtype=float).ravel()
    if len(feature_names) != coefficients.size:
        raise ValueError(
            f"{name}: {len(feature_names)} transformed feature names but "
            f"{coefficients.size} coefficients."
        )
    records = [
        {
            "Model": name,
            "Feature": feature,
            "Coefficient": float(coefficient),
            "AbsCoefficient": float(abs(coefficient)),
        }
        for feature, coefficient in zip(feature_names, coefficients)
    ]
    intercept = float(np.asarray(estimator.intercept_).ravel()[0])
    return records, intercept


def evaluate_results(
    results: Mapping[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_order: Sequence[str] = ("OLS", "Ridge", "Lasso"),
) -> EvaluationResults:
    """Evaluate fitted models once on the untouched test set.

    ``results`` maps model names to objects exposing ``pipeline``, ``best_alpha``,
    ``cv_rmse``, and ``display_alpha`` (normally :class:`src.train.ModelResult`).
    """
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same number of rows.")
    present = [name for name in model_order if name in results]
    if not present:
        raise ValueError("No recognized models were supplied for evaluation.")

    metric_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    coefficient_records: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    intercepts: dict[str, float] = {}

    for name in present:
        result = results[name]
        prediction = np.asarray(result.pipeline.predict(X_test), dtype=float).ravel()
        true_values, predicted_values = _validate_vectors(y_test, prediction)
        residual = true_values - predicted_values
        metrics = regression_metrics(true_values, predicted_values)
        predictions[name] = predicted_values
        residuals[name] = residual

        metric_rows.append(
            {
                "Model": name,
                "Best Alpha": result.display_alpha,
                "CV RMSE": float(result.cv_rmse),
                "Test RMSE": metrics["rmse"],
                "Test MAE": metrics["mae"],
                "Test R²": metrics["r2"],
                "Test RMSE (USD)": metrics["rmse"] * 100_000.0,
                "Test MAE (USD)": metrics["mae"] * 100_000.0,
                "Test Samples": int(metrics["n_samples"]),
            }
        )

        residual_rows.append(
            {
                "Model": name,
                "Mean Residual": float(residual.mean()),
                "Median Residual": float(np.median(residual)),
                "Residual Std": float(residual.std(ddof=1)),
                "Min Residual": float(residual.min()),
                "Q25 Residual": float(np.quantile(residual, 0.25)),
                "Q75 Residual": float(np.quantile(residual, 0.75)),
                "Max Residual": float(residual.max()),
                "Max Abs Residual": float(np.max(np.abs(residual))),
                "Test MAE": metrics["mae"],
                "Test RMSE": metrics["rmse"],
                "Corr(Prediction, Residual)": float(
                    np.corrcoef(predicted_values, residual)[0, 1]
                ),
            }
        )

        records, intercept = _extract_coefficients(name, result.pipeline)
        coefficient_records.extend(records)
        intercepts[name] = intercept

    metrics_frame = pd.DataFrame(metric_rows)
    residual_frame = pd.DataFrame(residual_rows)
    coefficient_frame = pd.DataFrame(coefficient_records)
    return EvaluationResults(
        metrics=metrics_frame,
        residual_summary=residual_frame,
        coefficients=coefficient_frame,
        predictions=predictions,
        residuals=residuals,
        intercepts=intercepts,
    )


def comparison_markdown(
    evaluation: EvaluationResults,
    *,
    cv_folds: int = 5,
    random_state: int = 42,
) -> str:
    """Create a Markdown evaluation report from computed results."""
    best_by_test = evaluation.metrics.sort_values("Test RMSE").iloc[0]
    lines = [
        "# Model Evaluation — California Housing",
        "",
        f"Target: `{TARGET_COLUMN}` in units of **{TARGET_UNIT}**. "
        "RMSE and MAE are therefore expressed in the same unit; multiply by "
        "100,000 for US dollars.",
        "",
        f"Hyperparameters were selected with {cv_folds}-fold shuffled "
        f"cross-validation (random_state={random_state}) on the training set only. "
        "The test set was evaluated once.",
        "",
        "## Comparison",
        "",
        evaluation.metrics_markdown(),
        "",
        "## Metric interpretation",
        "",
        "- **RMSE** penalizes large errors quadratically. A value of 0.50 means the "
        "typical squared-error magnitude corresponds to about $50,000.",
        "- **MAE** is the average absolute error and is less sensitive to a few large "
        "misses.",
        "- **R²** is the fraction of test-set variance explained relative to always "
        "predicting the test mean. It is unitless and can be negative.",
        "",
        "## Observed test results",
        "",
        f"- Lowest test RMSE in this run: **{best_by_test['Model']}** "
        f"({best_by_test['Test RMSE']:.4f} target units, approximately "
        f"${best_by_test['Test RMSE'] * 100_000:,.0f}).",
        f"- That model's test MAE is {best_by_test['Test MAE']:.4f} target units "
        f"(approximately ${best_by_test['Test MAE'] * 100_000:,.0f}) and "
        f"R² is {best_by_test['Test R²']:.4f}.",
        "- This is a descriptive comparison on one test split, not proof that the "
        "model is universally best or correctly specified.",
        "",
        "## Residual summary",
        "",
        "Residuals are defined as **actual − predicted**. Positive values are "
        "under-predictions; negative values are over-predictions.",
        "",
        _dataframe_to_markdown(evaluation.residual_summary),
        "",
        "## Coefficient caveat",
        "",
        "Numeric features are standardized before fitting, so their coefficients are "
        "comparable within a model on a per-standard-deviation basis. Latitude and "
        "longitude remain correlated, engineered ratios share components with raw "
        "features, and one-hot coefficients are relative to the encoder's reference "
        "category. Coefficients therefore indicate conditional associations in this "
        "model, not causal effects.",
    ]
    return "\n".join(lines) + "\n"


def write_evaluation_reports(
    evaluation: EvaluationResults,
    report_dir: str | Path = DEFAULT_REPORTS_DIR,
    *,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Path]:
    """Write comparison, residual, and coefficient tables to disk."""
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    metrics_path = directory / "model_comparison.csv"
    evaluation.metrics.to_csv(metrics_path, index=False)
    paths["metrics_csv"] = metrics_path

    markdown_path = directory / "model_comparison.md"
    markdown_path.write_text(
        comparison_markdown(
            evaluation, cv_folds=cv_folds, random_state=random_state
        ),
        encoding="utf-8",
    )
    paths["comparison_markdown"] = markdown_path

    residual_path = directory / "residual_summary.csv"
    evaluation.residual_summary.to_csv(residual_path, index=False)
    paths["residual_csv"] = residual_path

    coefficient_path = directory / "coefficient_analysis.csv"
    evaluation.coefficients.to_csv(coefficient_path, index=False)
    paths["coefficient_csv"] = coefficient_path
    return paths
