import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_customer():
    example = json.loads((ROOT / "model/input_example.json").read_text(encoding="utf-8"))
    values = dict(zip(example["columns"], example["data"][0], strict=True))
    values["SeniorCitizen"] = str(values["SeniorCitizen"])
    return values