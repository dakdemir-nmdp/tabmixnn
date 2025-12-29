from .core.model import TabMixNN, TabMixNNRegressor, TabMixNNClassifier, GMSEMEstimator
from .parsing.formula import OutcomeSpec

from .core.families import (
    Gaussian, Binomial, Poisson, NegBinomial, 
    DiscreteTimeSurvival, DiscreteTimeCompetingRisks,
    MultivariateGaussian, MultiLabelBinomial, ContinuousSurvival
)
from .utils.logging_config import set_log_level, disable_logging, enable_logging
from .utils.summary import summary, parameter_estimates

# Add summary method to TabMixNN
TabMixNN.summary = lambda self, **kwargs: summary(self, **kwargs)
TabMixNN.extract_parameters = lambda self, **kwargs: parameter_estimates(self, **kwargs)

# Legacy aliases - Pointing to Unified
GeneralizedTabMixNN = TabMixNN 
# GMSEMEstimator is now a class for backward compatibility
__all__ = [
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
    
    # Utilities
    "set_log_level",
    "disable_logging",
    "enable_logging",
]
