"""
TabMixNN: Mixed Effects Neural Networks for Tabular Data

A PyTorch-based framework for mixed-effects modeling with neural networks,
supporting multiple outcome types including survival analysis, competing risks,
and multitask learning.

Example Usage:
    >>> from tabmixnn import TabMixNNRegressor
    >>> model = TabMixNNRegressor(formula="y ~ x1 + x2 + (1|group)")
    >>> model.fit(data)
    >>> predictions = model.predict(new_data)

For more information, see: https://github.com/dakdemir-nmdp/tabmixnn
"""

__version__ = "0.4.1"
__author__ = "Deniz Akdemir"
__email__ = "deniz.akdemir.work@gmail.com"

from .core.model import TabMixNN, TabMixNNRegressor, TabMixNNClassifier, GMSEMEstimator
from .parsing.formula import OutcomeSpec

from .core.families import (
    Gaussian, Binomial, Poisson, NegBinomial, 
    DiscreteTimeSurvival, DiscreteTimeCompetingRisks,
    MultivariateGaussian, MultiLabelBinomial, ContinuousSurvival, Multinomial
)
from .utils.logging_config import set_log_level, disable_logging, enable_logging
from .utils.summary import summary, parameter_estimates

# Add summary method to TabMixNN
TabMixNN.summary = lambda self, **kwargs: summary(self, **kwargs)
TabMixNN.extract_parameters = lambda self, data=None, **kwargs: parameter_estimates(self, data=data, **kwargs)

# Legacy aliases - Pointing to Unified
GeneralizedTabMixNN = TabMixNN 
# GMSEMEstimator is now a class for backward compatibility
__all__ = [
    # Version
    "__version__",
    
    # Main Unified Class
    "TabMixNN",
    "TabMixNNRegressor",
    "TabMixNNClassifier",
    "GeneralizedTabMixNN",
    "GMSEMEstimator",
    
    # Outcome Specification
    "OutcomeSpec",
    
    # Families
    "Gaussian",
    "Binomial",
    "Poisson",
    "NegBinomial",
    "DiscreteTimeSurvival",
    "DiscreteTimeCompetingRisks",
    "MultivariateGaussian",
    "MultiLabelBinomial",
    "ContinuousSurvival",
    "Multinomial",
    
    # Utilities
    "set_log_level",
    "disable_logging",
    "enable_logging",
]
