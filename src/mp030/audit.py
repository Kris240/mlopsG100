from pathlib import Path
import yaml

from .config import ROOT, load_params
from .data import audit_data
import pandas as pd


def main() -> None:
    params = load_params()
    train = pd.read_csv(ROOT / params["data"]["train"])
    holdout = pd.read_csv(ROOT / params["data"]["holdout"])
    print(yaml.safe_dump(audit_data(train, holdout), sort_keys=False))


if __name__ == "__main__":
    main()
