from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(path=None):
    with open(Path(path) if path else ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load()
DATA = CFG["data"]
MODEL = CFG["model"]
SPLIT = CFG["split"]
SHIFT = CFG["shift"]
SUBMISSION = CFG["submission"]

PROCESSED = ROOT / CFG["paths"]["processed"]
ARTIFACTS = ROOT / CFG["paths"]["artifacts"]
SUBMISSIONS = ROOT / CFG["paths"]["submissions"]
REPORTS = ROOT / CFG["paths"]["reports"]
