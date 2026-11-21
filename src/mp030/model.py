from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    if model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(random_state=random_state, n_estimators=200, max_depth=2, learning_rate=0.05)
    elif model_name == "random_forest":
        estimator = RandomForestRegressor(random_state=random_state, n_estimators=250, min_samples_leaf=2, n_jobs=-1)
    elif model_name == "random_forest_deep":
        estimator = RandomForestRegressor(random_state=random_state, n_estimators=350, max_depth=12, min_samples_leaf=1, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])
