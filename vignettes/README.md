# TabMixNN Vignettes

Comprehensive tutorials for mastering TabMixNN.

## Overview

Vignettes provide in-depth coverage of specific topics with detailed explanations, code examples, and best practices.

## Available Vignettes

### 1. Getting Started
**File**: [01_getting_started.md](01_getting_started.md)

Complete introduction to TabMixNN covering:
- Installation and setup
- Core concepts (mixed effects + deep learning)
- Formula syntax
- Different outcome families
- Your first model
- Model fitting and prediction
- Model inspection

**Start here if you're new to TabMixNN!**

---

### 2. Mixed Effects Modeling in Depth
**File**: [02_mixed_effects_modeling.md](02_mixed_effects_modeling.md)

Comprehensive guide to mixed effects:
- What are mixed effects models?
- Random intercepts vs random slopes
- Multiple grouping factors
- Nested vs crossed effects
- Interpreting random effects
- When to use random effects
- Best practices

**Read this to master hierarchical modeling.**

---

### 3. Multi-task Learning
**File**: [03_multitask_learning.md](03_multitask_learning.md)

Modeling multiple outcomes simultaneously:
- What is multi-task learning?
- Benefits of multi-task models
- Heterogeneous outcome types
- Structural Equation Models (SEM)
- Specifying causal relationships
- SEM modes: independent, lavaan, learned

**Essential for modeling related outcomes.**

---

### 4. Survival Analysis
**File**: [04_survival_analysis.md](04_survival_analysis.md)

Complete guide to time-to-event modeling:
- Survival analysis basics
- DiscreteTimeSurvival for hazard-based models
- Competing risks analysis
- ContinuousSurvival parametric models
- Prediction modes (survival curves, hazards, CIF)
- Multitask survival models
- Best practices and common pitfalls

**Essential for survival analysis and competing risks.**

---

## How to Use These Vignettes

1. **Sequential Learning**: Read vignettes in order (1 → 2 → 3 → 4) for a complete understanding

2. **Topic-Specific**: Jump to specific vignettes based on your needs:
   - New user? Start with #1
   - Working with grouped data? See #2
   - Multiple outcomes? Check #3
   - Survival/time-to-event data? See #4

3. **Hands-On Practice**: Each vignette includes runnable code examples. Try them out!

## Relationship to Examples

| Examples | Vignettes |
|----------|-----------|
| Quick, focused code | Detailed explanations |
| Minimal commentary | Conceptual depth |
| Specific use cases | Comprehensive coverage |
| Run and go | Read and understand |

**Workflow**: Read vignettes → Run examples → Build your models

## Additional Resources

- **Examples**: [../examples/](../examples/) - Practical, runnable code
- **Main README**: [../README.md](../README.md) - Quick start guide
- **API Documentation**: [../API.md](../API.md) - Comprehensive API reference

## Getting Help

If something is unclear:
1. Re-read the relevant vignette section
2. Check the examples for similar use cases
3. Review the main README
4. Open an issue on GitHub with your question

## Contributing

Found an error or have suggestions for improving these vignettes? Please open an issue or pull request on GitHub!
