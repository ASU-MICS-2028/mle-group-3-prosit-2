"""Check everyone has the same raw data as the workplan (checked 28 Oct)."""
import pandas as pd
import pytest

from src import config

EXPECTED = {
    config.TRAIN_CSV: (68_654, 16),
    config.TEST_CSV: (18_594, 15),
    config.ECON_CSV: (27, 25),
    config.FINDEX_CSV: (1_000, 120),
}


@pytest.mark.parametrize("path,shape", EXPECTED.items(), ids=lambda p: getattr(p, "name", ""))
def test_shape(path, shape):
    assert path.exists(), f"Missing {path.name}; see data/README.md"
    df = pd.read_csv(path, encoding_errors="replace")
    assert df.shape == shape


def test_findex_has_weights():
    assert "wgt" in pd.read_csv(config.FINDEX_CSV, nrows=5).columns
