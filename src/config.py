"""Shared paths and the project-wide random seed."""
import os
import random
from pathlib import Path

import numpy as np

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

TRAIN_CSV = DATA_RAW / "Train.csv"
TEST_CSV = DATA_RAW / "Test.csv"
ECON_CSV = DATA_RAW / "economic_indicators.csv"
FINDEX_CSV = DATA_RAW / "micro_gha.csv"


def set_seed(seed: int = SEED) -> None:
    """Fix every RNG we use. Call at the top of each notebook.

    Models still need random_state=SEED passed explicitly.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
