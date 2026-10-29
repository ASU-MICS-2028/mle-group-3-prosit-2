"""Model pipelines shared by Sections 4 to 12. Everything that learns from data
(category grouping, imputation, scaling) sits inside the pipeline, so it is
refitted on each training fold and never sees validation rows."""
from lightgbm import LGBMClassifier, LGBMRegressor
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import features as F
from src.config import SEED

MAIN_FEATURES = F.LOAN + F.HISTORY + F.CALENDAR       # macro excluded by default (Section 3.4)
WITH_MACRO = MAIN_FEATURES + F.MACRO
# Deployed classifier: calendar dropped (no predictive value out of time, but it moved individual scores; Section 9)
DEPLOY_FEATURES = [c for c in MAIN_FEATURES if c not in F.CALENDAR]
# Loan-size regression: nothing computed from the amount being predicted (Section 1.4)
REG_FEATURES = [c for c in MAIN_FEATURES if c not in ("log_amount", "amount_per_day_log", "amount_vs_prev_mean")]


def preprocessor(numeric, scale: bool, log_cols=()) -> ColumnTransformer:
    num = [SimpleImputer(strategy="median", add_indicator=True)] + ([StandardScaler()] if scale else [])
    cat = make_pipeline(F.RareCategoryGrouper(min_count=30),
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    plain = [c for c in numeric if c not in log_cols]
    logged = [c for c in numeric if c in log_cols]
    parts = [("num", make_pipeline(*num), plain), ("cat", cat, F.CATEGORICAL)]
    if logged:
        parts.insert(1, ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                                              *[clone(step) for step in num]), logged))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def logreg(numeric=MAIN_FEATURES, class_weight="balanced", C=1.0) -> Pipeline:
    return Pipeline([("prep", preprocessor(numeric, scale=True)),
                     ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=SEED))])


def lgbm(numeric=MAIN_FEATURES, **params) -> Pipeline:
    base = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1,
                n_jobs=1)  # multithreaded fits were ~40x slower on macOS (OpenMP contention); data is small
    return Pipeline([("prep", preprocessor(numeric, scale=False)), ("model", LGBMClassifier(**{**base, **params}))])


def baseline(numeric=MAIN_FEATURES) -> Pipeline:
    """Predicts the training default rate for everyone, so it never flags a default."""
    return Pipeline([("prep", preprocessor(numeric, scale=False)), ("model", DummyClassifier(strategy="prior"))])


def tree(numeric=MAIN_FEATURES, class_weight=None, max_depth=6, min_samples_leaf=100) -> Pipeline:
    return Pipeline([("prep", preprocessor(numeric, scale=False)),
                     ("model", DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                      class_weight=class_weight, random_state=SEED))])


def forest(numeric=MAIN_FEATURES, class_weight=None, n_estimators=300, min_samples_leaf=20) -> Pipeline:
    return Pipeline([("prep", preprocessor(numeric, scale=False)),
                     ("model", RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                                      max_features="sqrt", class_weight=class_weight,
                                                      n_jobs=4, random_state=SEED))])


def undersampled(pipe: Pipeline, ratio: float = 0.1) -> ImbPipeline:
    """Random undersampling of repaid loans to `ratio` defaults per repaid loan.
    It sits inside the pipeline, so it only ever touches the training fold."""
    prep, model = pipe.steps
    return ImbPipeline([prep, ("under", RandomUnderSampler(sampling_strategy=ratio, random_state=SEED)), model])


def inputs(numeric=MAIN_FEATURES):
    return list(numeric) + F.CATEGORICAL


# ---- Regression (Section 6): target is log1p(Total_Amount) on repaid loans ----

# Terms run from 1 to 1,096 days; on a raw scale a linear model extrapolates a
# 3-year loan to e^57 KES. Linear models see these on a log scale.
LOG_TERMS = ("duration", "prev_duration_mean")


class ClippedRegressor(BaseEstimator, RegressorMixin):
    """Clip predictions to the range of targets seen in training: never suggest
    more than the largest (or less than the smallest) loan ever repaid."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.lo_, self.hi_ = float(np.min(y)), float(np.max(y))
        return self

    def predict(self, X):
        return np.clip(self.estimator_.predict(X), self.lo_, self.hi_)


def mean_reg(numeric=REG_FEATURES) -> Pipeline:
    return Pipeline([("prep", preprocessor(numeric, scale=False)), ("model", DummyRegressor(strategy="mean"))])


def linreg(numeric=REG_FEATURES, log_terms=LOG_TERMS) -> ClippedRegressor:
    return ClippedRegressor(Pipeline([("prep", preprocessor(numeric, scale=True, log_cols=log_terms)),
                                      ("model", LinearRegression())]))


def ridge(numeric=REG_FEATURES, alpha=10.0) -> ClippedRegressor:
    return ClippedRegressor(Pipeline([("prep", preprocessor(numeric, scale=True, log_cols=LOG_TERMS)),
                                      ("model", Ridge(alpha=alpha, random_state=SEED))]))


def lgbm_reg(numeric=REG_FEATURES, **params) -> ClippedRegressor:
    base = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=1)
    return ClippedRegressor(Pipeline([("prep", preprocessor(numeric, scale=False)),
                                      ("model", LGBMRegressor(**{**base, **params}))]))
