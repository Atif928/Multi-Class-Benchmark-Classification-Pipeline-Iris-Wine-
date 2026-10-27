# Task 2 — Real Estate Housing Price Prediction

## Project overview

This project builds a reproducible regression workflow for predicting the median
house value of California census block groups. It compares three linear models:

- **Ordinary Least Squares (OLS)** — an unregularized baseline.
- **Ridge regression (L2)** — shrinks coefficients and stabilizes estimates when
  predictors are correlated.
- **Lasso regression (L1)** — shrinks coefficients and can set some exactly to zero,
  producing a sparse model.

The project covers validation, exploratory data analysis (EDA), safe feature
engineering, leakage-free preprocessing pipelines, multicollinearity analysis,
five-fold cross-validated hyperparameter tuning, held-out test evaluation, residual
diagnostics, coefficient analysis, and a reusable prediction helper.

## Problem statement

> Given demographic and geographic features for a California census block group,
> estimate the median house value for that block group.

This is a **supervised regression** problem. The target is continuous and the model
is evaluated on data that was never used for fitting, feature selection, or
hyperparameter tuning.

## Dataset

| Item | Detail |
| ---- | ------ |
| Source | `sklearn.datasets.fetch_california_housing(as_frame=True)` |
| Observations | 20,640 census block groups |
| Input features | 8 numerical features |
| Target | `MedHouseVal` |
| Target unit | **$100,000** (e.g. `2.5` means **$250,000**) |
| Missing values | None in the original dataset (verified at runtime) |
| Categorical columns | None in the original dataset |

Features:

| Column | Meaning |
| ------ | ------- |
| `MedInc` | Median household income in the block group (tens of thousands of USD) |
| `HouseAge` | Median house age in years |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block-group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block-group latitude |
| `Longitude` | Block-group longitude |

The dataset is downloaded by scikit-learn on first use and cached by scikit-learn.
The data loader also supports an optional local CSV cache. Downloaded data and model
binaries are deliberately excluded from Git.

## Technology stack

- Python 3.10+
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- Joblib
- Jupyter Notebook
- pytest
- Git / GitHub

## Project structure

```text
task_2_house_price_prediction/
├── data/                       # Optional local data cache (not committed)
├── notebooks/
│   └── house_price_prediction.ipynb
├── outputs/
│   ├── figures/                # Saved PNG figures
│   └── reports/                # CSV / Markdown / JSON reports
├── models/                     # Fitted pipelines saved with Joblib (not committed)
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### Windows (PowerShell)

```powershell
cd task_2_house_price_prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Linux / macOS

```bash
cd task_2_house_price_prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

### Run the notebook

```bash
python -m jupyter notebook notebooks/house_price_prediction.ipynb
```

Run all cells from top to bottom. The notebook reuses the functions in `src/` and
displays the generated figures and real evaluation results.

### Run the full Python pipeline

From `task_2_house_price_prediction/`:

```bash
python -m src.train
```

This downloads/loads the dataset, validates it, writes EDA and diagnostic figures,
trains all three models, evaluates them on the held-out test set, and writes reports
to `outputs/reports/`.

### Run the tests

```bash
python -m pytest -q
```

Tests use small synthetic datasets and do not download the full dataset.

## Method summary

1. Load and validate the dataset.
2. Split rows into 80% training and 20% test **before** any learned preprocessing.
3. Add three stateless engineered features using safe division.
4. Infer numerical and categorical columns programmatically.
5. Build a `ColumnTransformer`:
   - numerical: median imputation → standard scaling;
   - categorical (if present): most-frequent imputation → one-hot encoding with
     `handle_unknown="ignore"`.
6. Fit every preprocessing step inside a scikit-learn `Pipeline`, only on training
   folds/data.
7. Train OLS, Ridge, and Lasso.
8. Tune Ridge and Lasso with `GridSearchCV` and shuffled five-fold
   cross-validation on the training set only, scoring with
   `neg_root_mean_squared_error`.
9. Evaluate all models once on the untouched test set.
10. Save comparison tables, residual diagnostics, coefficient analysis, and the
    fitted pipelines.

## Evaluation metrics

| Metric | Meaning | Unit |
| ------ | ------- | ---- |
| RMSE | Square root of mean squared error; penalizes large errors more heavily | $100,000 |
| MAE | Mean absolute error; average absolute miss | $100,000 |
| R² | Share of target variance explained relative to predicting the mean | unitless |

Because the target is in units of **$100,000**, an RMSE of `0.50` corresponds to
approximately **$50,000**. R² can be negative on a test set if a model performs worse
than predicting the training mean.

## Outputs

```text
outputs/figures/
├── 01_target_distribution.png
├── 02_correlation_heatmap.png
├── 03_feature_distributions.png
├── 04_feature_target_scatter.png
├── 05_feature_boxplots.png
├── 06_vif_analysis.png
├── 07_actual_vs_predicted.png
├── 08_residuals_vs_predicted.png
├── 09_residual_distributions.png
├── 10_metric_comparison.png
└── 11_coefficient_comparison.png

outputs/reports/
├── dataset_summary.json / dataset_summary.md
├── vif_analysis.csv
├── model_comparison.csv / model_comparison.md
├── residual_summary.csv
├── coefficient_analysis.csv
├── sample_prediction.json
└── run_metadata.json
```

Fitted pipelines are written to `models/` and are intentionally ignored by Git.

## Reproducibility

- Random state: `42` for the train/test split and cross-validation shuffling.
- Tuning uses the same shuffled five-fold splitter for Ridge and Lasso.
- The test set is created once and is never passed to `GridSearchCV`.
- All metrics are computed from the executed pipeline; none are hardcoded.
- `outputs/reports/run_metadata.json` records library versions and the seed.
- Install the versions from `requirements.txt` for the closest reproduction.

## Limitations and future improvements

- The data are aggregated at the census block-group level, not individual homes.
- `MedHouseVal` is clipped at `5.00001` (about $500,001), so very expensive areas
  are censored; this limits predictions at the top of the price range.
- Linear models cannot capture complex non-linear spatial effects unless they are
  explicitly engineered.
- Latitude and longitude are used as raw numeric inputs; tree-based or spatial
  models may capture location patterns better.
- Residuals are likely heteroscedastic and spatially clustered; the diagnostic plots
  should be read carefully rather than assuming ideal errors.
- Future work: quantile or robust regression, geographical feature engineering,
  gradient boosting, spatial cross-validation, and calibrated prediction intervals.

## Author

Atif — Datheon Machine Learning & Data Science Internship, Task 2.
