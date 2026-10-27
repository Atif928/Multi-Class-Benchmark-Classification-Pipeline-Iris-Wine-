"""Shared pytest fixtures and import-path setup.

All fixtures use small synthetic data so the suite is fast and does not download the
California Housing dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import prepare_datasets  # noqa: E402
from src.train import train_all_models  # noqa: E402

SMALL_ALPHA_GRIDS = {
    "Ridge": (0.1, 1.0),
    "Lasso": (0.001, 0.01),
}


@pytest.fixture(scope="module")
def synthetic_housing_frame() -> pd.DataFrame:
    """A deterministic, California-Housing-shaped regression dataset."""
    rng = np.random.default_rng(42)
    n_rows = 260
    med_income = rng.lognormal(mean=1.2, sigma=0.35, size=n_rows)
    house_age = rng.integers(1, 53, size=n_rows).astype(float)
    ave_rooms = rng.normal(5.2, 1.1, size=n_rows).clip(2.0, 10.0)
    ave_bedrooms = (ave_rooms * rng.uniform(0.15, 0.30, size=n_rows)).clip(0.8, 2.6)
    population = rng.integers(150, 4000, size=n_rows).astype(float)
    ave_occup = rng.uniform(1.2, 6.0, size=n_rows)
    latitude = rng.uniform(32.5, 42.0, size=n_rows)
    longitude = rng.uniform(-124.0, -114.0, size=n_rows)

    target = (
        0.42 * med_income
        + 0.18 * (ave_rooms / ave_occup)
        + 0.10 * (ave_bedrooms / ave_rooms)
        + 0.05 * np.log1p(population)
        - 0.004 * house_age
        + 0.0005 * (latitude + longitude)
        + rng.normal(0, 0.12, size=n_rows)
    ).clip(0.15, 5.0)

    return pd.DataFrame(
        {
            "MedInc": med_income,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrooms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
            "MedHouseVal": target,
        }
    )


@pytest.fixture(scope="module")
def prepared_data(synthetic_housing_frame: pd.DataFrame):
    """Leakage-safe prepared train/test data built from the synthetic frame."""
    return prepare_datasets(synthetic_housing_frame, random_state=42)


@pytest.fixture(scope="module")
def trained_models(prepared_data):
    """Three fitted pipelines tuned with small grids for test speed."""
    return train_all_models(
        prepared_data,
        alpha_grids=SMALL_ALPHA_GRIDS,
        n_jobs=1,
    )
