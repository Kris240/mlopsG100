from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from .config import ROOT, load_params
from .data import FEATURES, NUMERIC_FEATURES, clean_data, sha256, split_xy
from .model import build_pipeline


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(expected, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    a = pd.cut(actual, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    e, a = e.clip(lower=1e-6), a.clip(lower=1e-6)
    return float(((a - e) * np.log(a / e)).sum())


def main() -> None:
    params = load_params()
    train_path = ROOT / params["data"]["train"]
    train = clean_data(pd.read_csv(train_path))
    later = clean_data(pd.read_csv(ROOT / params["data"]["later"]), drop_duplicates=False).sort_values(["date", "record_id"])
    v1 = joblib.load(ROOT / "models/model.joblib")
    later_predictions = v1.predict(later[FEATURES])
    group_mae = {name: float(mean_absolute_error(group["handle_min"], later_predictions[group.index])) for name, group in later.reset_index(drop=True).groupby("team")}
    cut = int(len(later) * params["monitoring"]["later_train_fraction"])
    build, check = later.iloc[:cut], later.iloc[cut:]
    combined = pd.concat([train, build], ignore_index=True)
    x_combined, y_combined = split_xy(combined)
    v2 = build_pipeline("gradient_boosting", params["model"]["random_state"])
    v2.fit(x_combined, y_combined)
    x_check, y_check = split_xy(check)
    mae_v1 = float(mean_absolute_error(y_check, v1.predict(x_check)))
    mae_v2 = float(mean_absolute_error(y_check, v2.predict(x_check)))
    promote = mae_v2 <= (1 - params["monitoring"]["promotion_improvement"]) * mae_v1
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(params["model"]["experiment_name"])
    with mlflow.start_run(run_name="v2-retrain") as run:
        mlflow.log_params({"model_name": "gradient_boosting", "data_sha256": sha256(train_path)})
        mlflow.log_metrics({"mae_v1_check": mae_v1, "mae_v2_check": mae_v2})
        mlflow.sklearn.log_model(v2, "model", serialization_format="cloudpickle")
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", params["model"]["registered_model_name"])
    if promote:
        MlflowClient().set_registered_model_alias(params["model"]["registered_model_name"], "champion", version.version)
        joblib.dump(v2, ROOT / "models/model.joblib")
    champion = joblib.load(ROOT / "models/model.joblib")
    holdout = pd.read_csv(ROOT / params["data"]["holdout"])
    predictions = pd.DataFrame({"record_id": holdout["record_id"], "prediction": champion.predict(clean_data(holdout, False)[FEATURES])})
    (ROOT / "predictions").mkdir(exist_ok=True)
    predictions.to_csv(ROOT / "predictions/holdout_predictions.csv", index=False)
    result = {"psi": {c: psi(train[c], later[c], params["monitoring"]["psi_bins"]) for c in NUMERIC_FEATURES}, "mae_by_group": group_mae, "mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2, "decision": "promote" if promote else "keep", "v2_version": version.version}
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
