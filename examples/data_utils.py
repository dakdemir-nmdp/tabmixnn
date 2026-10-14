"""
Utility functions for loading real datasets used in TabMixNN examples.
These datasets are standard benchmarks from R packages (lme4, nlme, mlmRev).
"""

import os
import pandas as pd
import requests
from pathlib import Path

# Create data directory in examples folder
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Standard datasets used in mixed effects modeling
DATASETS = {
    # Famous mixed effects regression datasets
    "sleepstudy": "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv",
    "chickweight": "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ChickWeight.csv",
    "co2": "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/CO2.csv",
    "machines": "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Machines.csv",
    "oxboys": "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Oxboys.csv",
    "rail": "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Rail.csv",

    # Binary classification with grouping
    "cbpp": "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cbpp.csv",
    "contraception": "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Contraception.csv",

    # Multi-level datasets
    "inst_eval": "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/InstEval.csv",
    "exam": "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Exam.csv",
}


def download_dataset(name):
    """Download a dataset from R datasets repository."""
    name_lower = name.lower()

    if name_lower not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    url = DATASETS[name_lower]
    filename = f"{name_lower}.csv"
    file_path = DATA_DIR / filename

    if file_path.exists():
        print(f"Dataset '{name}' already downloaded.")
    else:
        print(f"Downloading '{name}' from {url}...")
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            file_path.write_bytes(response.content)
            print(f"Downloaded to {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download {name}: {e}")

    return file_path


def load_dataset(name):
    """
    Load a dataset, downloading if necessary.

    Parameters
    ----------
    name : str
        Dataset name (case-insensitive). Options:
        - sleepstudy: Reaction times vs sleep deprivation (18 subjects)
        - chickweight: Chick growth over time (50 chicks, 4 diets)
        - co2: CO2 uptake in plants (12 plants, 2 types)
        - machines: Production scores on 3 machines (6 workers)
        - oxboys: Growth of Oxford boys (26 boys)
        - rail: Travel time on rails (6 rails)
        - cbpp: Cattle disease incidence (binary, 15 herds)
        - contraception: Contraceptive use (binary, districts)
        - inst_eval: University lecture evaluations
        - exam: Exam scores by school

    Returns
    -------
    pd.DataFrame
        The loaded dataset with row index column removed
    """
    file_path = download_dataset(name)

    df = pd.read_csv(file_path)

    # Remove row index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    elif df.columns[0] in ["rownames", "X"]:
        df = df.drop(columns=[df.columns[0]])

    return df


def get_dataset_info(name):
    """Get information about a dataset."""
    info = {
        "sleepstudy": {
            "description": "Reaction times (ms) vs days of sleep deprivation",
            "outcome": "Reaction",
            "grouping": "Subject",
            "n_obs": 180,
            "n_groups": 18,
            "type": "longitudinal regression"
        },
        "chickweight": {
            "description": "Chick weight over time for different diets",
            "outcome": "weight",
            "grouping": "Chick",
            "n_obs": 578,
            "n_groups": 50,
            "type": "growth curve"
        },
        "co2": {
            "description": "CO2 uptake in grass plants",
            "outcome": "uptake",
            "grouping": "Plant",
            "n_obs": 84,
            "n_groups": 12,
            "type": "repeated measures"
        },
        "machines": {
            "description": "Production scores on machines by workers",
            "outcome": "score",
            "grouping": "Worker",
            "n_obs": 54,
            "n_groups": 6,
            "type": "crossed effects"
        },
        "cbpp": {
            "description": "Cattle disease (CBPP) incidence by herd",
            "outcome": "incidence/size (binary)",
            "grouping": "herd",
            "n_obs": 56,
            "n_groups": 15,
            "type": "binomial GLMM"
        }
    }

    name_lower = name.lower()
    if name_lower in info:
        return info[name_lower]
    else:
        return {"description": "No detailed info available"}


if __name__ == "__main__":
    # Download all datasets
    print("Downloading all example datasets...")
    for name in DATASETS.keys():
        try:
            df = load_dataset(name)
            print(f"  {name}: {df.shape}")
        except Exception as e:
            print(f"  {name}: Failed - {e}")
