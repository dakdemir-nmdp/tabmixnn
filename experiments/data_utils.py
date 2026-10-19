"""Shared dataset loader for TabMixNN benchmark experiments (Tables 2-4, arXiv:2512.23787)."""

import io
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

_URLS = {
    # Regression
    "sleepstudy":   "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv",
    "chickweight":  "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ChickWeight.csv",
    "mathachieve":  "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/MathAchieve.csv",
    "spruce":       "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Spruce.csv",
    "radon":        "https://vincentarelbundock.github.io/Rdatasets/csv/HLMdiag/radon.csv",
    # Classification
    "contraception":"https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Contraception.csv",
    "toenail":      "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/toenail.csv",
    "verbagg":      "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/VerbAgg.csv",
    "train":        "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Train.csv",
    "exam":         "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Exam.csv",
}


def load(name: str) -> pd.DataFrame:
    """Load dataset by name, downloading and caching if needed."""
    key = name.lower()
    if key not in _URLS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_URLS)}")
    path = DATA_DIR / f"{key}.csv"
    if not path.exists():
        print(f"Downloading {key}...")
        r = requests.get(_URLS[key], timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_csv(path)
    drop = [c for c in df.columns if c in ("Unnamed: 0", "rownames")]
    return df.drop(columns=drop)
