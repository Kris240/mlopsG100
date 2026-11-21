from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_params() -> dict:
    with (ROOT / "params.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)
