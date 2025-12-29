# TabMixNN Examples

This directory contains practical examples demonstrating the key features of TabMixNN using **real datasets**.

## Running the Examples

Each example is a standalone Python script that can be run directly:

```bash
cd tabmixnn/examples

# Download datasets first (optional - examples will auto-download)
python data_utils.py

# Run examples
python 01_linear_mixed_model.py
python 02_binary_classification.py
python 03_random_slopes.py
python 04_multiple_outcomes.py
python 05_survival_analysis.py
python 06_competing_risks.py
python 07_architecture_comparison.py
```

## Prerequisites

```bash
pip install tabmixnn pandas numpy matplotlib scikit-learn requests

# Optional (for enhanced survival analysis comparisons):
# pip install lifelines scikit-survival
```

## Examples Overview

### ✅ 01. Linear Mixed Model - Sleepstudy
**File**: [01_linear_mixed_model.py](01_linear_mixed_model.py)

**Dataset**: Sleepstudy (lme4) - Reaction times vs sleep deprivation for 18 subjects

Basic linear mixed model with:
- Random intercepts AND random slopes
- Longitudinal data (repeated measures)
- Model evaluation and visualization
- Parameter extraction

**Key concepts**: Random effects, formula syntax, longitudinal data

**Status**: ✅ Ready to run with real data

---

### ✅ 02. Binary Classification - CBPP
**File**: [02_binary_classification.py](02_binary_classification.py)

**Dataset**: CBPP (lme4) - Cattle disease incidence across 15 herds

Generalized linear mixed model for binary outcomes:
- Binomial family (logistic regression)
- Random intercepts across herds
- Classification metrics (accuracy, AUC, sensitivity/specificity)
- Temporal disease progression

**Key concepts**: GLMM, binomial family, grouped binary data

**Status**: ✅ Ready to run with real data

---

### ✅ 03. Random Slopes - ChickWeight
**File**: [03_random_slopes.py](03_random_slopes.py)

**Dataset**: ChickWeight - Growth of 50 chicks on 4 different diets

Advanced mixed model with varying effects:
- Random intercepts AND random slopes
- Growth curve modeling
- Comparison: random slopes vs intercepts only
- Individual growth trajectories

**Key concepts**: Random slopes, growth curves, (1 + Time|Chick) syntax

**Status**: ✅ Ready to run with real data

---

### ✅ 04. Multiple Outcomes - Machines
**File**: [04_multiple_outcomes.py](04_multiple_outcomes.py)

**Dataset**: Machines (nlme) - Production scores on 3 machines by 6 workers

Multi-task learning with heterogeneous outcomes:
- Two outcomes: continuous (productivity score) + binary (high/low performance)
- Shared architecture across tasks
- Different families (Gaussian + Binomial)
- Random effects for individual workers

**Key concepts**: Multi-task learning, OutcomeSpec, multiple families

**Status**: ✅ Ready to run with real data

---

### ✅ 05. Survival Analysis - Rossi Recidivism
**File**: [05_survival_analysis.py](05_survival_analysis.py)

**Dataset**: Rossi (carData) - Time to re-arrest for 432 released prisoners

Survival analysis with discrete-time survival models:
- DiscreteTimeSurvival family
- Time-to-event modeling
- Survival curves and hazard functions
- C-index evaluation
- Kaplan-Meier-style visualization

**Key concepts**: Survival analysis, censoring, hazard functions, C-index

**Status**: ✅ Ready to run with real data

---

### ✅ 06. Competing Risks - Colon Cancer
**File**: [06_competing_risks.py](06_competing_risks.py)

**Dataset**: Colon (survival) - Cancer recurrence in 1858 patients from chemotherapy trial

Competing risks survival analysis:
- Multiple mutually exclusive events
- Cause-specific hazards
- Cumulative incidence functions (CIF)
- Treatment effect visualization

**Key concepts**: Competing risks, cumulative incidence, cause-specific hazards

**Status**: ✅ Ready to run with real data

---

### ✅ 07. Architecture Comparison - Machines
**File**: [07_architecture_comparison.py](07_architecture_comparison.py)

**Dataset**: Machines (nlme) - Production scores on 3 machines by 6 workers

Compare GSEM vs Manifold architectures:
- GSEM: Generalized SEM (hierarchical structure)
- Manifold: Universal backbone with latent manifolds
- Performance metrics comparison
- Training time comparison
- Visualization of architecture differences

**Key concepts**: Architecture selection, GSEM vs Manifold, model comparison

**Status**: ✅ Ready to run with real data

---

## Dataset Utilities

**File**: [data_utils.py](data_utils.py)

Utility functions for loading real datasets from R packages (lme4, nlme, mlmRev):

```python
from data_utils import load_dataset, get_dataset_info

# Load a dataset
df = load_dataset("sleepstudy")

# Get information about available datasets
info = get_dataset_info("sleepstudy")
print(info)
```

**Available datasets**:
- `sleepstudy`: Reaction times vs sleep deprivation (180 obs, 18 subjects)
- `chickweight`: Chick growth over time (578 obs, 50 chicks)
- `cbpp`: Cattle disease incidence (56 obs, 15 herds)
- `machines`: Production scores (54 obs, 6 workers, 3 machines)
- `rossi`: Recidivism data (432 obs, from carData package)
- `colon`: Colon cancer chemotherapy trial (1858 obs, from survival package)
- `oxboys`: Oxford boys growth (234 obs, 26 boys)
- `rail`: Rail travel times (18 obs, 6 rails)
- `contraception`: Contraceptive use (grouped binary)
- `inst_eval`: University evaluations (large dataset)
- `exam`: Exam scores by school (large dataset)

**Note**: Rossi and Colon datasets are auto-downloaded directly from Rdatasets URLs in examples 05 and 06.

Datasets are automatically downloaded from [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) on first use.

---

## Output Files

Examples save plots to the `output/` directory:
- `01_sleepstudy_predictions.png`
- `02_cbpp_predictions.png`
- `03_chickweight_growth.png`
- `04_machines_multitask.png`
- `05_rossi_survival.png`
- `06_colon_competing_risks.png`
- `07_architecture_comparison.png`

---

## Example Workflow

**Beginner Track:**
1. **Example 01** - Learn basic mixed effects modeling with linear models
2. **Example 02** - Understand GLMMs for binary outcomes
3. **Example 03** - Master random slopes for longitudinal data

**Advanced Track:**
4. **Example 04** - Multi-task learning with multiple outcomes
5. **Example 05** - Survival analysis with time-to-event data
6. **Example 06** - Competing risks with multiple event types
7. **Example 07** - Architecture comparison (GSEM vs Manifold)

Each example builds on concepts from previous ones.

---

## Benchmarks

These examples use datasets where TabMixNN demonstrates strong performance:

- **Sleepstudy**: Competitive with lme4, captures random slopes effectively
- **CBPP**: Strong classification performance on grouped binary data
- **ChickWeight**: Excellent fit for growth curves with random effects
- **Machines**: Effective multi-task learning across outcome types
- **Rossi**: Good discrimination for recidivism prediction (C-index > 0.65)
- **Colon**: Competitive survival modeling for cancer outcomes
- **Architecture Comparison**: Both GSEM and Manifold perform well on hierarchical data

---

## Next Steps

After exploring these examples, see the [vignettes](../vignettes/) directory for:
- Detailed tutorials on specific use cases
- Best practices and advanced techniques
- Comprehensive explanations of concepts

---

## Need Help?

- Read the main [README](../README.md)
- Check the [vignettes](../vignettes/)
- Review the dataset info with `get_dataset_info()`
- Open an issue on GitHub

---

## Contributing Examples

Have an interesting use case? Contribute an example!

Requirements:
- Use real datasets (no synthetic data)
- Include clear documentation
- Show performance metrics
- Save visualizations to `output/`

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
