"""Shared fixtures. The dataset is built once and reused across test modules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formalcrrc import dataset as ds  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def formal_dataset():
    """The full 2916-row Day-1 dataset, built from the seed."""
    return ds.build_dataset()


@pytest.fixture(scope="session")
def day2_calibration():
    """The frozen Day-2 CALIBRATION partition, rebuilt from seed 43."""
    from formalcrrc import day2_config as d2c

    day1_pairs = _family_c_pairs(ds.build_dataset())
    return ds.build_dataset(
        seed=d2c.SEED_CALIBRATION,
        partition=d2c.PARTITION_CALIBRATION,
        forbidden_pairs=day1_pairs,
        enforce_unique_targets=True,
    )


@pytest.fixture(scope="session")
def day2_test(day2_calibration):
    """The frozen Day-2 TEST partition, rebuilt from seed 44."""
    from formalcrrc import day2_config as d2c

    forbidden = _family_c_pairs(ds.build_dataset()) | _family_c_pairs(day2_calibration)
    return ds.build_dataset(
        seed=d2c.SEED_TEST,
        partition=d2c.PARTITION_TEST,
        forbidden_pairs=forbidden,
        enforce_unique_targets=True,
    )


def _family_c_pairs(frame):
    subset = frame[frame["family"] == "numeric_tolerance"].drop_duplicates(
        "artifact_id"
    )
    return frozenset(
        zip(subset["target_value"].astype(int), subset["reported_value"].astype(int))
    )


@pytest.fixture(scope="session")
def day3_test(day2_calibration, day2_test):
    """The frozen Day-3 evaluation partition, rebuilt from seed 45."""
    from formalcrrc import day3_config as d3c

    day1 = ds.build_dataset()
    forbidden = (
        _family_c_pairs(day1)
        | _family_c_pairs(day2_calibration)
        | _family_c_pairs(day2_test)
    )
    return ds.build_dataset(
        seed=d3c.SEED,
        partition=d3c.PARTITION_TAG,
        forbidden_pairs=forbidden,
        enforce_unique_targets=True,
    )
