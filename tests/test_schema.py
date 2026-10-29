"""Guard the Part A audit: leaky or redundant columns must never become features."""
import pandas as pd

from src import config, schema


def test_leaky_columns_not_usable():
    assert not set(schema.LEAKY) & set(schema.USABLE)
    assert set(schema.LEAKY) <= set(schema.DROPPED)


def test_every_train_column_has_a_role():
    cols = set(pd.read_csv(config.TRAIN_CSV, nrows=1).columns)
    assigned = set(schema.IDS) | set(schema.USABLE) | set(schema.DROPPED) | set(schema.META) | {schema.TARGET}
    assert cols == assigned
