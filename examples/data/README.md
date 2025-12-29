# Data Directory

This directory contains datasets that are automatically downloaded when you run the examples.

## Auto-Download

Datasets are automatically downloaded from [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) on first use.

## Available Datasets

The following datasets are used in the examples:

- **sleepstudy** - Reaction times vs sleep deprivation (180 obs, 18 subjects)
- **cbpp** - Cattle disease incidence (56 obs, 15 herds)
- **chickweight** - Chick growth over time (578 obs, 50 chicks)
- **machines** - Production scores (54 obs, 6 workers, 3 machines)

And more available through `data_utils.py`.

## Usage

```python
from data_utils import load_dataset

# Load a dataset (auto-downloads if needed)
df = load_dataset('sleepstudy')
```

The downloaded CSV files are cached here for faster subsequent access.

## Note

Downloaded data files (*.csv) are excluded from Git via .gitignore.
