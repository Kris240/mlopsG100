from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

from .config import ROOT, load_params
from .data import clean_data, sha256, split_xy
from .model import build_pipeline


def main() -> None:
    params = load_params()
    train_path = ROOT / params["data"]["train"]
    frame = clean_data(pd.read_csv(train_path)).sort_values(["date", "record_id"])
    split = int(len(frame) * (1 - params["model"]["validation_fraction"]))
    training, validation = frame.iloc[:split], frame.iloc[split:]
    x_train, y_train = split_xy(training)
    x_valid, y_valid = split_xy(validation)
    tracking_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["model"]["experiment_name"])
    digest = sha256(train_path)
    results = []
    for name in ["gradient_boosting", "random_forest", "random_forest_deep"]:
        model = build_pipeline(name, params["model"]["random_state"])
        with mlflow.start_run(run_name=name) as run:
            model.fit(x_train, y_train)
            mae = mean_absolute_error(y_valid, model.predict(x_valid))
            mlflow.log_params({"model_name": name, "data_sha256": digest})
            mlflow.log_metric("mae_validation", mae)
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
            results.append((mae, run.info.run_id, model))
    mae, run_id, champion = min(results, key=lambda item: item[0])
    model_name = params["model"]["registered_model_name"]
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, model_name)
    MlflowClient().set_registered_model_alias(model_name, "champion", registered.version)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(champion, ROOT / "models/model.joblib")
    print({"best_run_id": run_id, "mae": mae, "version": registered.version})


if __name__ == "__main__":
    main()
