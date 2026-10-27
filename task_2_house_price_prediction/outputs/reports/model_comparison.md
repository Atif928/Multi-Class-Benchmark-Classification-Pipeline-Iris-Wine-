# Model Evaluation — California Housing

Target: `MedHouseVal` in units of **$100,000**. RMSE and MAE are therefore expressed in the same unit; multiply by 100,000 for US dollars.

Hyperparameters were selected with 5-fold shuffled cross-validation (random_state=42) on the training set only. The test set was evaluated once.

## Comparison

| Model | Best Alpha | CV RMSE | Test RMSE | Test MAE | Test R² |
| --- | --- | ---: | ---: | ---: | ---: |
| OLS | N/A | 0.6550 | 0.6621 | 0.4737 | 0.6655 |
| Ridge | 1 | 0.6550 | 0.6621 | 0.4737 | 0.6655 |
| Lasso | 0.001 | 0.6548 | 0.6626 | 0.4741 | 0.6649 |

## Metric interpretation

- **RMSE** penalizes large errors quadratically. A value of 0.50 means the typical squared-error magnitude corresponds to about $50,000.
- **MAE** is the average absolute error and is less sensitive to a few large misses.
- **R²** is the fraction of test-set variance explained relative to always predicting the test mean. It is unitless and can be negative.

## Observed test results

- Lowest test RMSE in this run: **OLS** (0.6621 target units, approximately $66,210).
- That model's test MAE is 0.4737 target units (approximately $47,374) and R² is 0.6655.
- This is a descriptive comparison on one test split, not proof that the model is universally best or correctly specified.

## Residual summary

Residuals are defined as **actual − predicted**. Positive values are under-predictions; negative values are over-predictions.

| Model | Mean Residual | Median Residual | Residual Std | Min Residual | Q25 Residual | Q75 Residual | Max Residual | Max Abs Residual | Test MAE | Test RMSE | Corr(Prediction, Residual) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OLS | 0.0085 | -0.0757 | 0.6621 | -3.6057 | -0.3810 | 0.3000 | 4.4573 | 4.4573 | 0.4737 | 0.6621 | -0.0197 |
| Ridge | 0.0085 | -0.0760 | 0.6621 | -3.6045 | -0.3809 | 0.3001 | 4.4581 | 4.4581 | 0.4737 | 0.6621 | -0.0195 |
| Lasso | 0.0083 | -0.0780 | 0.6627 | -3.5760 | -0.3816 | 0.2985 | 4.4413 | 4.4413 | 0.4741 | 0.6626 | -0.0152 |

## Coefficient caveat

Numeric features are standardized before fitting, so their coefficients are comparable within a model on a per-standard-deviation basis. Latitude and longitude remain correlated, engineered ratios share components with raw features, and one-hot coefficients are relative to the encoder's reference category. Coefficients therefore indicate conditional associations in this model, not causal effects.
