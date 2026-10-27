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
