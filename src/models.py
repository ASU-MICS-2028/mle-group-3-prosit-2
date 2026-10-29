"""Model pipelines shared by Sections 4 to 12. Everything that learns from data
(category grouping, imputation, scaling) sits inside the pipeline, so it is
refitted on each training fold and never sees validation rows."""
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import features as F
from src.config import SEED

MAIN_FEATURES = F.LOAN + F.HISTORY + F.CALENDAR       # macro excluded by default (Section 3.4)
WITH_MACRO = MAIN_FEATURES + F.MACRO


def preprocessor(numeric, scale: bool) -> ColumnTransformer:
    num = [SimpleImputer(strategy="median", add_indicator=True)] + ([StandardScaler()] if scale else [])
    cat = make_pipeline(F.RareCategoryGrouper(min_count=30),
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    return ColumnTransformer([("num", make_pipeline(*num), list(numeric)), ("cat", cat, F.CATEGORICAL)],
                             verbose_feature_names_out=False)


def logreg(numeric=MAIN_FEATURES, class_weight="balanced", C=1.0) -> Pipeline:
    return Pipeline([("prep", preprocessor(numeric, scale=True)),
                     ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=SEED))])


def lgbm(numeric=MAIN_FEATURES, **params) -> Pipeline:
    base = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1,
                n_jobs=1)  # multithreaded fits were ~40x slower on macOS (OpenMP contention); data is small
    return Pipeline([("prep", preprocessor(numeric, scale=False)), ("model", LGBMClassifier(**{**base, **params}))])


def inputs(numeric=MAIN_FEATURES):
    return list(numeric) + F.CATEGORICAL
