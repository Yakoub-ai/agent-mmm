"""Shared pytest fixtures for agent_mmm tests."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def synthetic_weekly() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "synthetic_weekly.csv")


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "synthetic_panel.csv")
