# Task 2 — Real Estate Housing Price Prediction

## Project overview

This project builds a complete, reproducible regression workflow for predicting the
median house value of California census block groups. It compares three linear
models:

- **Ordinary Least Squares (OLS)** — an unregularized baseline.
- **Ridge regression (L2)** — shrinks coefficients and stabilizes estimates when
  predictors are correlated.
- **Lasso regression (L1)** — shrinks coefficients and can set some exactly to zero,
  producing a sparse model.

The project covers validation, exploratory data analysis (EDA), safe feature
engineering, leakage-free preprocessing pipelines, multicollinearity analysis,
five-fold cross-validated hyperparameter tuning, held-out test evaluation, residual
diagnostics, coefficient analysis, a reusable prediction helper, and an executed
Jupyter notebook.

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
| Duplicate rows | 0 |

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
The data loader also supports an optional local CSV cache:

```bash
python -m src.train --cache-path data/california_housing.csv
```

Downloaded data and model binaries are deliberately excluded from Git.

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
│   ├── figures/                # Saved PNG figures (committed deliverables)
│   └── reports/                # CSV / Markdown / JSON reports (committed)
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

Run all cells from top to bottom. The notebook imports the reusable functions from
`src/` and displays the actual evaluation results and generated figures.

### Run the full Python pipeline

From `task_2_house_price_prediction/`:

```bash
python -m src.train
```

This loads and validates the dataset, writes the EDA figures, splits the data,
engineers features, computes VIF, tunes all three models, evaluates them on the
held-out test set, and writes figures, reports, and Joblib pipelines.

Optional arguments:

```bash
python -m src.train --n-jobs 1
python -m src.train --cache-path data/california_housing.csv
```

### Run the tests

```bash
python -m pytest -q
```

The 34 tests use small synthetic datasets and do not download the full dataset.

## Method summary

1. Load and validate the dataset, checking target presence, types, missing values,
   infinite values, duplicates, and minimum sample size.
2. Split rows into 80% training and 20% test **before** any learned preprocessing.
3. Add four stateless engineered features to the training and test frames separately.
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

### Feature engineering

| Feature | Definition | Rationale |
| ------- | ---------- | --------- |
| `RoomsPerPerson` | `AveRooms / AveOccup` | Separates spacious households from crowded ones; the raw averages mix both. |
| `BedroomRatio` | `AveBedrms / AveRooms` | Bedroom share acts as a proxy for unit size and layout. |
| `IncomePerPerson` | `MedInc / AveOccup` | Scales neighbourhood income by household size. |
| `LogPopulation` | `log1p(Population)` | Compresses the long right tail of block-group population. |

Division uses `safe_divide`, which returns `NaN` for zero, near-zero, or missing
denominators instead of raising or creating infinities. The numeric imputer then
handles those values inside the pipeline. The target is never used to construct a
feature, and the engineering function is idempotent.

### Multicollinearity

VIF is computed on the training features only, using the identity
`VIF = 1 / (1 - R²)` from regressing each feature on all other features:

| Feature | VIF | Flag |
| ------- | --: | ---- |
| `AveRooms` | 28.05 | high |
| `AveBedrms` | 18.12 | high |
| `MedInc` | 16.34 | high |
| `IncomePerPerson` | 15.45 | high |
| `RoomsPerPerson` | 13.99 | high |
| `Latitude` | 9.34 | moderate |
| `Longitude` | 8.97 | moderate |
| `BedroomRatio` | 4.84 | low |
| `LogPopulation` | 2.88 | low |
| `Population` | 2.84 | low |
| `HouseAge` | 1.24 | low |
| `AveOccup` | 1.04 | low |

The engineered ratios share components with their source variables, and latitude
and longitude are strongly correlated (`r = 0.925`). This is expected and is not
"fixed" by dropping the features: Ridge and Lasso explicitly handle shared variance
through coefficient shrinkage. Cross-validation, not test-set inspection, chooses
the penalty strength.

## Results (executed run)

The values below were produced by the executed pipeline on 2026-10-27 with
`random_state=42`, an 80/20 split (16,512 train / 4,128 test rows), and five-fold
shuffled CV. They are descriptive results for this split, not universal claims.

| Model | Best Alpha | CV RMSE | Test RMSE | Test MAE | Test R² |
| ----- | ---------- | ------: | --------: | -------: | ------: |
| OLS | N/A | 0.6550 | 0.6621 | 0.4737 | 0.6655 |
| Ridge | 1 | 0.6550 | 0.6621 | 0.4737 | 0.6655 |
| Lasso | 0.001 | 0.6548 | 0.6626 | 0.4741 | 0.6649 |

Target units are **$100,000**, so:

- Test RMSE ≈ `0.6621 × $100,000` ≈ **$66,210**.
- Test MAE ≈ `0.4737 × $100,000` ≈ **$47,374**.
- Test R² ≈ **0.665**, meaning the models explain about 66.5% of the test-set
  variance relative to a mean-only baseline.

**Interpretation.** The three models are practically indistinguishable on this
split. OLS has the lowest test RMSE by roughly $5, Ridge's selected penalty is
effectively neutral, and Lasso's best CV score uses very weak regularization and
does not set coefficients exactly to zero. Regularization stabilizes correlated
coefficients but does not materially improve predictive accuracy here; the linear
functional form is the larger limitation. The lowest test RMSE is reported as an
observed result, not used to select hyperparameters.

## Residual analysis

Residuals are defined as **actual − predicted** in units of $100,000:

- Mean residual is about `0.0085` (≈ $851), so average under-prediction is small,
  but the median is about `-0.076`, and errors range from about `-3.61` to `+4.46`.
- Prediction–residual correlation is about `-0.02`, so there is no strong linear
  trend in error magnitude across predictions.
- The binned residual curve is U-shaped: low predictions tend to be too low
  (positive residuals), middle predictions slightly too high, and high predictions
  too low. This is consistent with the linear model compressing extremes and with
  the target being capped at `5.00001`.
- About 65.4% of test errors are within 0.5 target units (≈ $50,000) and 90.3% are
  within 1.0 unit (≈ $100,000). The largest errors occur at the high end of the
  market.
- Residual spread does not shrink at higher predictions; errors are
  heteroscedastic and spatially clustered. The model is not claimed to be unbiased
  or statistically ideal.

## Evaluation metrics

| Metric | Meaning | Unit |
| ------ | ------- | ---- |
| RMSE | Square root of mean squared error; penalizes large errors more heavily | $100,000 |
| MAE | Mean absolute error; average absolute miss | $100,000 |
| R² | Share of target variance explained relative to predicting the mean | unitless |

R² can be negative on a test set if a model performs worse than predicting the
training mean. RMSE and MAE must be multiplied by 100,000 to be read as dollars.

## Outputs

```text
outputs/figures/
├── 01_target_distribution.png
├── 02_correlation_heatmap.png
├── 03_feature_distributions.png
├── 04_feature_target_scatter.png
├── 05_geographic_target.png
├── 06_feature_boxplots.png
├── 07_vif_analysis.png
├── 08_actual_vs_predicted.png
├── 09_residuals_vs_predicted.png
├── 10_residual_distributions.png
├── 11_metric_comparison.png
└── 12_coefficient_comparison.png

outputs/reports/
├── dataset_summary.json / dataset_summary.md
├── vif_analysis.csv
├── model_comparison.csv / model_comparison.md
├── residual_summary.csv
├── coefficient_analysis.csv
├── sample_prediction.json
└── run_metadata.json
```

`sample_prediction.json` demonstrates the prediction helper on one held-out row,
including both target units and USD. Fitted pipelines are written to `models/` and
are intentionally ignored by Git.

### Using the prediction helper

```python
from src.train import load_model, predict_house_value, predict_house_value_dollars

pipeline = load_model("models/ridge_pipeline.joblib")
features = {
    "MedInc": 5.0,
    "HouseAge": 25.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1200.0,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25,
}
value_100k = predict_house_value(pipeline, features)   # e.g. 2.1
value_usd = predict_house_value_dollars(pipeline, features)  # e.g. 210000.0
```

The helper validates the eight raw inputs, adds engineered features, and returns
the model's prediction in the documented units. It does not accept the target.

## Reproducibility

- Random state: `42` for the train/test split and cross-validation shuffling.
- Tuning uses the same shuffled five-fold splitter for Ridge and Lasso.
- The test set is created once and is never passed to `GridSearchCV`.
- All metrics are computed from the executed pipeline; none are typed in manually.
- `outputs/reports/run_metadata.json` records library versions, the seed, selected
  alphas, convergence warnings, row counts, and artifact locations.
- Install the versions from `requirements.txt` for the closest reproduction.

## Limitations and future improvements

- The data are aggregated at the census block-group level, not individual homes.
- `MedHouseVal` is clipped at `5.00001` (about $500,001); 965 rows (4.68%) sit at
  the cap, so very expensive areas are censored.
- Linear models cannot capture complex non-linear spatial effects unless they are
  explicitly engineered. The U-shaped residual pattern is evidence of this.
- Latitude and longitude are used as raw numeric inputs; tree-based or spatial
  models may capture location patterns better.
- Residuals are heteroscedastic and spatially clustered, so constant-variance
  inference is not appropriate.
- Future work: quantile or robust regression, spatial features and spatial
  cross-validation, gradient boosting, and calibrated prediction intervals.

## Author

Atif — Datheon Machine Learning & Data Science Internship, Task 2.
