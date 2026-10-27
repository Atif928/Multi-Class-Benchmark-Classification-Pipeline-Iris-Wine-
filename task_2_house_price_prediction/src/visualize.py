"""Plotting utilities for exploratory analysis and regression diagnostics.

Each plotting function returns a Matplotlib ``Figure`` so it can be displayed in a
notebook or saved by the orchestration helpers. Generated figures go to
``outputs/figures/`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import TARGET_COLUMN
from .data_loader import load_california_housing

FIGURE_DPI = 150
DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
SCATTER_FEATURES: tuple[str, ...] = ("MedInc", "AveRooms", "HouseAge", "AveOccup")
MODEL_COLORS: dict[str, str] = {
    "OLS": "#1f77b4",
    "Ridge": "#2ca02c",
    "Lasso": "#d62728",
}


def set_plot_style() -> None:
    """Apply a consistent, readable plotting theme."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": FIGURE_DPI,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "figure.titlesize": 14,
            "figure.titleweight": "bold",
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save_figure(figure: plt.Figure, output_path: str | Path) -> Path:
    """Save ``figure`` to ``output_path`` and close it.

    Parent directories are created automatically. Returns the resolved path.
    """
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_target_distribution(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
) -> plt.Figure:
    """Histogram of the target with a KDE curve, mean, and median markers."""
    values = frame[target].dropna().astype(float)
    mean_value = float(values.mean())
    median_value = float(values.median())
    cap_value = float(values.max())
    capped_count = int((values >= cap_value - 1e-6).sum())

    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    sns.histplot(values, bins=50, kde=True, color="#2a6f97", edgecolor="white", ax=axis)
    axis.axvline(
        mean_value,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_value:.3f} (~${mean_value * 100_000:,.0f})",
    )
    axis.axvline(
        median_value,
        color="#2ca02c",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_value:.3f} (~${median_value * 100_000:,.0f})",
    )
    axis.set_title("Distribution of Median House Values (Target)")
    axis.set_xlabel("Median house value (units of $100,000)")
    axis.set_ylabel("Number of census block groups")
    axis.legend(loc="upper right")
    axis.text(
        0.02,
        0.96,
        f"n = {len(values):,}\nMode near target cap: {capped_count:,} rows "
        f"({capped_count / len(values):.1%})\nUpper cap ≈ ${cap_value * 100_000:,.0f}",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    return figure


def plot_correlation_heatmap(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
) -> plt.Figure:
    """Pearson correlation heatmap for all numeric columns."""
    numeric = frame.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        raise ValueError(f"Target '{target}' is not numeric or not present.")
    correlation = numeric.corr()
    if correlation.isna().all().all():
        raise ValueError("Correlation matrix is empty; check numeric columns.")

    figure, axis = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation coefficient"},
        ax=axis,
    )
    axis.set_title("Correlation Heatmap of Numeric Features and Target")
    axis.set_xlabel("")
    axis.set_ylabel("")
    figure.text(
        0.5,
        0.01,
        "Values near ±1 indicate strong linear association; strong feature-feature "
        "correlations suggest multicollinearity.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    return figure


def plot_feature_distributions(
    frame: pd.DataFrame,
    features: Sequence[str] | None = None,
    target: str = TARGET_COLUMN,
) -> plt.Figure:
    """Grid of histograms with KDE curves for the input features."""
    columns = list(features) if features is not None else [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column != target
    ]
    if not columns:
        raise ValueError("No numeric feature columns are available to plot.")

    n_columns = 2
    n_rows = int(np.ceil(len(columns) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(12, 2.6 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, column in zip(axes_array, columns):
        sns.histplot(
            frame[column].dropna(),
            bins=40,
            kde=True,
            color="#4c72b0",
            edgecolor="white",
            ax=axis,
        )
        axis.set_title(column)
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")

    for axis in axes_array[len(columns) :]:
        axis.set_visible(False)

    figure.suptitle("Distributions of Input Features", y=1.01)
    return figure


def plot_feature_target_scatter(
    frame: pd.DataFrame,
    features: Sequence[str] = SCATTER_FEATURES,
    target: str = TARGET_COLUMN,
) -> plt.Figure:
    """Scatterplots of selected features against the target."""
    available = [feature for feature in features if feature in frame.columns]
    if not available:
        raise ValueError(
            "None of the requested scatter features are present. "
            f"Requested: {list(features)}"
        )

    n_columns = 2
    n_rows = int(np.ceil(len(available) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(12, 4.2 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, feature in zip(axes_array, available):
        sns.scatterplot(
            data=frame,
            x=feature,
            y=target,
            s=12,
            alpha=0.25,
            linewidth=0,
            color="#2a6f97",
            ax=axis,
            rasterized=True,
        )
        correlation = frame[[feature, target]].corr().iloc[0, 1]
        axis.set_title(f"{feature} vs target (r = {correlation:.2f})")
        axis.set_xlabel(feature)
        axis.set_ylabel("Median house value ($100,000)")

    for axis in axes_array[len(available) :]:
        axis.set_visible(False)

    figure.suptitle("Selected Features Against Median House Value", y=1.01)
    figure.text(
        0.5,
        0.005,
        "Points are partially transparent because 20,640 block groups overlap.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    return figure


def plot_geographic_target(
    frame: pd.DataFrame,
    target: str = TARGET_COLUMN,
) -> plt.Figure:
    """Map block groups by latitude/longitude, coloured by the target."""
    required = {"Latitude", "Longitude", target}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Geographic plot requires columns: {sorted(required)}")

    figure, axis = plt.subplots(figsize=(8, 9))
    scatter = axis.scatter(
        frame["Longitude"],
        frame["Latitude"],
        c=frame[target],
        cmap="viridis",
        s=8,
        alpha=0.7,
        linewidths=0,
        rasterized=True,
    )
    colorbar = figure.colorbar(scatter, ax=axis, shrink=0.8)
    colorbar.set_label("Median house value ($100,000)")
    axis.set_title("Geographic Distribution of Median House Values")
    axis.set_xlabel("Longitude (degrees)")
    axis.set_ylabel("Latitude (degrees)")
    axis.set_aspect("equal", adjustable="box")
    axis.text(
        0.02,
        0.98,
        "Higher values cluster near the coast,\nnotably around the Bay Area and Los Angeles.",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    return figure


def plot_feature_boxplots(
    frame: pd.DataFrame,
    features: Sequence[str] | None = None,
    target: str = TARGET_COLUMN,
) -> plt.Figure:
    """Univariate boxplots for outlier screening of input features."""
    columns = list(features) if features is not None else [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column != target
    ]
    if not columns:
        raise ValueError("No numeric feature columns are available to plot.")

    n_columns = 4
    n_rows = int(np.ceil(len(columns) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(14, 3.3 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis, column in zip(axes_array, columns):
        sns.boxplot(y=frame[column].dropna(), color="#8da0cb", width=0.45, ax=axis)
        axis.set_title(column)
        axis.set_ylabel(column)
        axis.set_xlabel("")
        axis.grid(axis="x", visible=False)

    for axis in axes_array[len(columns) :]:
        axis.set_visible(False)

    figure.suptitle(
        "Boxplots of Input Features (Univariate Outlier Screening)", y=1.01
    )
    figure.text(
        0.5,
        0.005,
        "Points beyond the whiskers are flagged as potential outliers. "
        "They are not automatically removed.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    return figure


def plot_vif_analysis(
    vif_frame: pd.DataFrame,
    *,
    low_threshold: float = 5.0,
    high_threshold: float = 10.0,
) -> plt.Figure:
    """Horizontal bar chart of variance inflation factors.

    ``vif_frame`` must contain ``feature`` and ``vif`` columns, typically produced
    by :func:`src.features.variance_inflation_factors`.
    """
    required = {"feature", "vif"}
    missing = required.difference(vif_frame.columns)
    if missing:
        raise ValueError(f"VIF frame is missing columns: {sorted(missing)}")

    data = vif_frame.sort_values("vif", ascending=True).copy()
    finite_values = data.loc[np.isfinite(data["vif"]), "vif"]
    finite_max = float(finite_values.max()) if not finite_values.empty else 10.0
    data["plot_vif"] = data["vif"].replace(np.inf, finite_max * 1.35)
    data["plot_vif"] = data["plot_vif"].clip(lower=1e-3)

    colors = [
        "#d62728" if value >= high_threshold
        else "#ff7f0e" if value >= low_threshold
        else "#2ca02c"
        for value in data["vif"]
    ]

    figure, axis = plt.subplots(figsize=(10, max(5, 0.45 * len(data) + 2)))
    bars = axis.barh(data["feature"], data["plot_vif"], color=colors, alpha=0.85)
    axis.set_xscale("log")
    axis.axvline(
        low_threshold,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"Moderate threshold ({low_threshold:g})",
    )
    axis.axvline(
        high_threshold,
        color="#111111",
        linestyle=":",
        linewidth=1.4,
        label=f"High threshold ({high_threshold:g})",
    )
    for bar, value in zip(bars, data["vif"]):
        label = "∞" if not np.isfinite(value) else f"{value:.1f}"
        axis.text(
            bar.get_width() * 1.05,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
        )
    axis.set_title("Variance Inflation Factors of Training Features")
    axis.set_xlabel("VIF (log scale; higher = more shared linear variance)")
    axis.set_ylabel("")
    axis.legend(loc="lower right")
    axis.grid(axis="y", visible=False)
    figure.text(
        0.5,
        0.005,
        "VIF is computed by regressing each feature on all other features. "
        "High values flag multicollinearity, not necessarily poor predictive power.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    return figure


def plot_actual_vs_predicted(
    predictions: Mapping[str, np.ndarray],
    y_test: pd.Series | np.ndarray,
    *,
    model_order: Sequence[str] = ("OLS", "Ridge", "Lasso"),
) -> plt.Figure:
    """Actual-versus-predicted scatterplots with a perfect-prediction line."""
    actual = np.asarray(y_test, dtype=float).ravel()
    names = [name for name in model_order if name in predictions]
    if not names:
        raise ValueError("No predictions matched the requested model names.")

    lower = float(min(actual.min(), *(np.min(predictions[name]) for name in names)))
    upper = float(max(actual.max(), *(np.max(predictions[name]) for name in names)))
    figure, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 5.2), squeeze=False)

    for axis, name in zip(axes[0], names):
        predicted = np.asarray(predictions[name], dtype=float).ravel()
        axis.scatter(
            actual,
            predicted,
            s=10,
            alpha=0.30,
            linewidths=0,
            color=MODEL_COLORS.get(name, "#1f77b4"),
            rasterized=True,
        )
        axis.plot(
            [lower, upper],
            [lower, upper],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="Perfect prediction (y = x)",
        )
        residual_sum = float(np.sum((actual - predicted) ** 2))
        rmse = float(np.sqrt(residual_sum / len(actual)))
        denom = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1.0 - residual_sum / denom if denom > 0 else float("nan")
        axis.set_title(name)
        axis.set_xlabel("Actual median house value ($100,000)")
        axis.set_ylabel("Predicted median house value ($100,000)")
        axis.legend(loc="upper left")
        axis.text(
            0.98,
            0.03,
            f"RMSE = {rmse:.4f}\nR² = {r2:.4f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    figure.suptitle("Actual vs Predicted Median House Values — Test Set", y=1.02)
    return figure


def _binned_residual_means(
    predicted: np.ndarray,
    residual: np.ndarray,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"predicted": predicted, "residual": residual})
    frame["bin"] = pd.qcut(frame["predicted"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        predicted=("predicted", "mean"),
        residual=("residual", "mean"),
    )
    return grouped["predicted"].to_numpy(), grouped["residual"].to_numpy()


def plot_residuals_vs_predicted(
    predictions: Mapping[str, np.ndarray],
    y_test: pd.Series | np.ndarray,
    *,
    model_order: Sequence[str] = ("OLS", "Ridge", "Lasso"),
) -> plt.Figure:
    """Residual scatterplots with a zero line and binned mean residual curve."""
    actual = np.asarray(y_test, dtype=float).ravel()
    names = [name for name in model_order if name in predictions]
    if not names:
        raise ValueError("No predictions matched the requested model names.")

    figure, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 5.2), squeeze=False)
    for axis, name in zip(axes[0], names):
        predicted = np.asarray(predictions[name], dtype=float).ravel()
        residual = actual - predicted
        axis.scatter(
            predicted,
            residual,
            s=10,
            alpha=0.30,
            linewidths=0,
            color=MODEL_COLORS.get(name, "#1f77b4"),
            rasterized=True,
        )
        axis.axhline(0, color="black", linestyle="--", linewidth=1.3)
        bin_centers, bin_means = _binned_residual_means(predicted, residual)
        axis.plot(
            bin_centers,
            bin_means,
            color="#d62728",
            marker="o",
            markersize=4,
            linewidth=1.8,
            label="Binned mean residual (deciles)",
        )
        axis.set_title(name)
        axis.set_xlabel("Predicted median house value ($100,000)")
        axis.set_ylabel("Residual: actual − predicted ($100,000)")
        axis.legend(loc="upper right")

    figure.suptitle("Residuals vs Predicted Values — Test Set", y=1.02)
    figure.text(
        0.5,
        0.01,
        "A flat zero-centred cloud is ideal. Curves or fan shapes indicate "
        "non-linearity or changing error variance.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    return figure


def plot_residual_distributions(
    residuals: Mapping[str, np.ndarray],
    *,
    model_order: Sequence[str] = ("OLS", "Ridge", "Lasso"),
) -> plt.Figure:
    """Histograms with KDE curves for test-set residuals."""
    names = [name for name in model_order if name in residuals]
    if not names:
        raise ValueError("No residuals matched the requested model names.")

    figure, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 4.8), squeeze=False)
    for axis, name in zip(axes[0], names):
        values = np.asarray(residuals[name], dtype=float).ravel()
        sns.histplot(
            values,
            bins=50,
            kde=True,
            color=MODEL_COLORS.get(name, "#1f77b4"),
            edgecolor="white",
            ax=axis,
        )
        axis.axvline(0, color="black", linestyle="--", linewidth=1.3)
        axis.axvline(
            float(values.mean()),
            color="#d62728",
            linestyle=":",
            linewidth=1.4,
            label=f"Mean = {values.mean():.4f}",
        )
        axis.set_title(name)
        axis.set_xlabel("Residual: actual − predicted ($100,000)")
        axis.set_ylabel("Frequency")
        axis.legend(loc="upper right")
        axis.text(
            0.02,
            0.96,
            f"Std = {values.std(ddof=1):.4f}\n"
            f"Median = {np.median(values):.4f}\n"
            f"Skew = {pd.Series(values).skew():.4f}",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    figure.suptitle("Residual Distributions — Test Set", y=1.02)
    return figure


def plot_metric_comparison(metrics: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart comparing RMSE, MAE, and R² across models."""
    required = {"Model", "Test RMSE", "Test MAE", "Test R²"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Metrics frame is missing columns: {sorted(missing)}")

    metric_columns = ["Test RMSE", "Test MAE", "Test R²"]
    metric_labels = ["RMSE", "MAE", "R²"]
    plot_data = metrics.set_index("Model")[metric_columns]
    positions = np.arange(len(plot_data))
    width = 0.25
    colors = ["#2a6f97", "#61a5c2", "#a9d6e5"]

    figure, axis = plt.subplots(figsize=(10, 5.8))
    for index, (column, label, color) in enumerate(
        zip(metric_columns, metric_labels, colors)
    ):
        offsets = (index - 1) * width
        bars = axis.bar(
            positions + offsets,
            plot_data[column],
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.008,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(positions)
    axis.set_xticklabels(plot_data.index)
    axis.set_ylabel("Metric value (RMSE/MAE in $100,000; R² unitless)")
    axis.set_title("Test-Set Performance Comparison")
    axis.legend(title="Metric", loc="upper right")
    axis.set_ylim(bottom=0)
    figure.text(
        0.5,
        0.01,
        "Lower RMSE/MAE is better; higher R² is better.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    return figure


def plot_coefficient_comparison(
    coefficients: pd.DataFrame,
    *,
    top_n: int = 12,
    model_order: Sequence[str] = ("OLS", "Ridge", "Lasso"),
) -> plt.Figure:
    """Compare standardized coefficients, highlighting Lasso sparsity.

    Numeric inputs are standardized by the preprocessing pipeline, so coefficients
    are on a per-standard-deviation scale. Engineered ratios and raw features share
    components, and encoded categorical coefficients are relative to a reference
    level; these are associations, not causal effects.
    """
    required = {"Model", "Feature", "Coefficient"}
    missing = required.difference(coefficients.columns)
    if missing:
        raise ValueError(f"Coefficient frame is missing columns: {sorted(missing)}")

    names = [name for name in model_order if name in set(coefficients["Model"])]
    if not names:
        raise ValueError("No recognized models were found in the coefficient frame.")

    regularized = coefficients[coefficients["Model"].isin(["Ridge", "Lasso"])]
    ranking_source = regularized if not regularized.empty else coefficients
    importance = (
        ranking_source.assign(Abs=ranking_source["Coefficient"].abs())
        .groupby("Feature", as_index=False)["Abs"]
        .max()
        .sort_values("Abs", ascending=False)
    )
    selected = importance.head(top_n)["Feature"].tolist()

    figure, axes = plt.subplots(1, len(names), figsize=(5.6 * len(names), 6.2), squeeze=False)
    for axis, name in zip(axes[0], names):
        subset = (
            coefficients[coefficients["Model"] == name]
            .set_index("Feature")
            .reindex(selected)
        )
        values = subset["Coefficient"].fillna(0.0).to_numpy(dtype=float)
        axis.barh(
            np.arange(len(selected)),
            values,
            color=MODEL_COLORS.get(name, "#1f77b4"),
            alpha=0.85,
        )
        axis.axvline(0, color="black", linewidth=1)
        axis.set_yticks(np.arange(len(selected)))
        axis.set_yticklabels(selected)
        axis.invert_yaxis()
        axis.set_title(name)
        axis.set_xlabel("Standardized coefficient (target units per 1 SD)")
        axis.grid(axis="y", visible=False)
        zero_count = int(np.isclose(values, 0.0, atol=1e-12).sum())
        axis.text(
            0.98,
            0.02,
            f"Zero coefficients shown: {zero_count}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    figure.suptitle("Coefficient Comparison — Top Features by Regularized Importance", y=1.01)
    figure.text(
        0.5,
        0.01,
        "Numeric predictors are standardized. Correlated and engineered features "
        "share information, so coefficients are not causal effects.",
        ha="center",
        fontsize=9,
        style="italic",
    )
    return figure


def generate_diagnostic_figures(
    evaluation: object,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Path]:
    """Generate and save test-set diagnostic and coefficient figures.

    ``evaluation`` is expected to expose ``predictions``, ``residuals``,
    ``metrics``, and ``coefficients`` attributes, normally an
    :class:`src.evaluate.EvaluationResults`.
    """
    set_plot_style()
    directory = Path(output_dir)
    # The bundle stores predictions and residuals, so recover actual values from
    # their identity (actual = predicted + residual) without retaining y_test.
    first_name = next(iter(evaluation.predictions))
    actual = evaluation.predictions[first_name] + evaluation.residuals[first_name]
    figures = {
        "actual_vs_predicted": plot_actual_vs_predicted(evaluation.predictions, actual),
        "residuals_vs_predicted": plot_residuals_vs_predicted(
            evaluation.predictions, actual
        ),
        "residual_distributions": plot_residual_distributions(evaluation.residuals),
        "metric_comparison": plot_metric_comparison(evaluation.metrics),
        "coefficient_comparison": plot_coefficient_comparison(evaluation.coefficients),
    }
    filenames = {
        "actual_vs_predicted": "08_actual_vs_predicted.png",
        "residuals_vs_predicted": "09_residuals_vs_predicted.png",
        "residual_distributions": "10_residual_distributions.png",
        "metric_comparison": "11_metric_comparison.png",
        "coefficient_comparison": "12_coefficient_comparison.png",
    }
    paths: dict[str, Path] = {}
    for key, figure in figures.items():
        paths[key] = save_figure(figure, directory / filenames[key])
    return paths


def generate_eda_figures(
    frame: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
    target: str = TARGET_COLUMN,
) -> dict[str, Path]:
    """Generate and save all exploratory figures.

    Returns a mapping from a short figure key to the saved path.
    """
    set_plot_style()
    directory = Path(output_dir)
    figures: dict[str, plt.Figure] = {
        "target_distribution": plot_target_distribution(frame, target),
        "correlation_heatmap": plot_correlation_heatmap(frame, target),
        "feature_distributions": plot_feature_distributions(frame, target=target),
        "feature_target_scatter": plot_feature_target_scatter(frame, target=target),
        "geographic_target": plot_geographic_target(frame, target),
        "feature_boxplots": plot_feature_boxplots(frame, target=target),
    }
    filenames = {
        "target_distribution": "01_target_distribution.png",
        "correlation_heatmap": "02_correlation_heatmap.png",
        "feature_distributions": "03_feature_distributions.png",
        "feature_target_scatter": "04_feature_target_scatter.png",
        "geographic_target": "05_geographic_target.png",
        "feature_boxplots": "06_feature_boxplots.png",
    }
    paths: dict[str, Path] = {}
    for key, figure in figures.items():
        paths[key] = save_figure(figure, directory / filenames[key])
    return paths


def main() -> None:
    """Generate EDA figures from the California Housing dataset."""
    parser = argparse.ArgumentParser(description="Generate California Housing EDA plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for saved figures.",
    )
    arguments = parser.parse_args()
    frame = load_california_housing()
    paths = generate_eda_figures(frame, arguments.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
