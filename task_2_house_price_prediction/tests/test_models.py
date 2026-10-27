"""Tests for training, evaluation, prediction helpers, and reproducibility."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.data_loader import FEATURE_COLUMNS
from src.evaluate import evaluate_results, regression_metrics, write_evaluation_reports
from src.preprocessing import prepare_datasets
from src.train import (
    MODEL_ORDER,
    load_model,
    predict_house_value,
    predict_house_value_dollars,
    save_models,
    train_all_models,
)

SMALL_ALPHA_GRIDS = {"Ridge": (0.1, 1.0), "Lasso": (0.001, 0.01)}


def test_training_returns_three_fitted_models(trained_models, prepared_data):
    assert set(trained_models) == set(MODEL_ORDER)
    for name, result in trained_models.items():
        assert result.name == name
        assert np.isfinite(result.cv_rmse)
        assert result.cv_rmse > 0
        assert np.isfinite(result.cv_std)
        predictions = result.pipeline.predict(prepared_data.X_test)
        assert len(predictions) == len(prepared_data.X_test)
        assert np.isfinite(predictions).all()


def test_regularized_models_report_alphas_from_the_grid(trained_models):
    assert trained_models["OLS"].best_alpha is None
    assert trained_models["OLS"].display_alpha == "N/A"
    assert trained_models["Ridge"].best_alpha in SMALL_ALPHA_GRIDS["Ridge"]
    assert trained_models["Lasso"].best_alpha in SMALL_ALPHA_GRIDS["Lasso"]


def test_metrics_are_finite_and_correct_for_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = regression_metrics(y_true, y_true)
    assert perfect["rmse"] == pytest.approx(0.0)
    assert perfect["mae"] == pytest.approx(0.0)
    assert perfect["r2"] == pytest.approx(1.0)

    imperfect = regression_metrics(y_true, y_true + 0.5)
    assert np.isfinite(imperfect["rmse"])
    assert imperfect["rmse"] == pytest.approx(0.5)
    assert imperfect["mae"] == pytest.approx(0.5)
    assert np.isfinite(imperfect["r2"])


def test_metrics_reject_invalid_inputs():
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        regression_metrics([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        regression_metrics([], [])


def test_evaluation_table_contains_required_columns(trained_models, prepared_data):
    evaluation = evaluate_results(
        trained_models, prepared_data.X_test, prepared_data.y_test
    )
    expected = {
        "Model",
        "Best Alpha",
        "CV RMSE",
        "Test RMSE",
        "Test MAE",
        "Test R²",
    }
    assert expected.issubset(evaluation.metrics.columns)
    assert evaluation.metrics["Test RMSE"].gt(0).all()
    assert np.isfinite(evaluation.metrics["Test R²"]).all()
    assert set(evaluation.predictions) == set(MODEL_ORDER)

    for name in MODEL_ORDER:
        expected_residual = (
            prepared_data.y_test.to_numpy() - evaluation.predictions[name]
        )
        np.testing.assert_allclose(evaluation.residuals[name], expected_residual)
        assert np.isfinite(evaluation.residuals[name]).all()

    assert not evaluation.coefficients.empty
    assert {"Model", "Feature", "Coefficient"}.issubset(evaluation.coefficients.columns)
    assert set(evaluation.intercepts) == set(MODEL_ORDER)
    markdown = evaluation.metrics_markdown()
    assert "| OLS | N/A |" in markdown


def test_test_set_is_not_used_for_training(synthetic_housing_frame):
    prepared = prepare_datasets(synthetic_housing_frame, random_state=7)
    altered = dataclasses.replace(
        prepared, y_test=prepared.y_test * 100.0, X_test=prepared.X_test.copy()
    )

    original_results = train_all_models(
        prepared, alpha_grids=SMALL_ALPHA_GRIDS, n_jobs=1
    )
    altered_results = train_all_models(
        altered, alpha_grids=SMALL_ALPHA_GRIDS, n_jobs=1
    )

    for name in MODEL_ORDER:
        assert original_results[name].best_alpha == altered_results[name].best_alpha
        assert original_results[name].cv_rmse == pytest.approx(
            altered_results[name].cv_rmse
        )
        np.testing.assert_allclose(
            original_results[name].pipeline.predict(prepared.X_test),
            altered_results[name].pipeline.predict(prepared.X_test),
        )


def test_training_is_reproducible(synthetic_housing_frame):
    prepared = prepare_datasets(synthetic_housing_frame, random_state=11)
    first = train_all_models(prepared, alpha_grids=SMALL_ALPHA_GRIDS, n_jobs=1)
    second = train_all_models(prepared, alpha_grids=SMALL_ALPHA_GRIDS, n_jobs=1)

    for name in MODEL_ORDER:
        assert first[name].cv_rmse == pytest.approx(second[name].cv_rmse)
        assert first[name].best_alpha == second[name].best_alpha
        np.testing.assert_allclose(
            first[name].pipeline.predict(prepared.X_test),
            second[name].pipeline.predict(prepared.X_test),
        )


def test_prediction_helper_uses_raw_features_and_returns_target_units(
    trained_models, prepared_data
):
    pipeline = trained_models["OLS"].pipeline
    raw_features = {
        column: float(prepared_data.X_test.iloc[0][column])
        for column in FEATURE_COLUMNS
    }
    prediction = predict_house_value(pipeline, raw_features)
    dollars = predict_house_value_dollars(pipeline, raw_features)

    assert np.isfinite(prediction)
    assert prediction > 0
    assert dollars == pytest.approx(prediction * 100_000.0)

    frame_prediction = predict_house_value(
        pipeline, pd.DataFrame([raw_features])
    )
    assert frame_prediction == pytest.approx(prediction)


def test_prediction_helper_validates_input(trained_models):
    pipeline = trained_models["OLS"].pipeline
    with pytest.raises(ValueError, match="Missing required raw features"):
        predict_house_value(pipeline, {"MedInc": 3.0})
    with pytest.raises(ValueError, match="exactly one row"):
        predict_house_value(
            pipeline,
            pd.DataFrame(
                [
                    {column: 1.0 for column in FEATURE_COLUMNS},
                    {column: 2.0 for column in FEATURE_COLUMNS},
                ]
            ),
        )
    with pytest.raises(TypeError):
        predict_house_value(pipeline, [1.0, 2.0])  # type: ignore[arg-type]


def test_save_and_load_model_round_trip(trained_models, prepared_data, tmp_path):
    paths = save_models(trained_models, tmp_path)
    assert set(paths) == set(MODEL_ORDER)
    for name, path in paths.items():
        assert path.is_file()
        loaded = load_model(path)
        np.testing.assert_allclose(
            loaded.predict(prepared_data.X_test),
            trained_models[name].pipeline.predict(prepared_data.X_test),
        )


def test_write_evaluation_reports_creates_files(
    trained_models, prepared_data, tmp_path
):
    evaluation = evaluate_results(
        trained_models, prepared_data.X_test, prepared_data.y_test
    )
    paths = write_evaluation_reports(evaluation, tmp_path)
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    assert "| Model | Best Alpha | CV RMSE |" in (
        tmp_path / "model_comparison.md"
    ).read_text(encoding="utf-8")
