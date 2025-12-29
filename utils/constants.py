"""Constants used throughout TabMixNN for numerical stability and defaults."""

# Numerical stability constants
EPSILON = 1e-8
MIN_LOG_SIGMA = -5.0
MAX_LOG_SIGMA = 5.0
MIN_LOG_PARAM = -15.0  # For NegBin and other families
MAX_LOG_PARAM = 15.0
HAZARD_CLIP_MIN = 1e-7
HAZARD_CLIP_MAX = 1.0 - 1e-7

# SAM layer defaults
DEFAULT_SAM_ALPHA_INIT = 0.0
DEFAULT_SAM_MAX_OFFDIAG = 0.9
DEFAULT_SAM_RESIDUAL_SCALE = 1.0

# Training hyperparameter defaults
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_HIDDEN_DIMS = [64]
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 20
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LAMBDA_SPARSE = 1e-3
DEFAULT_LAMBDA_KL = 1e-2

# Validation and early stopping defaults
DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 0.0

# SAM location options
VALID_SAM_LOCATIONS = ["none", "first", "output", "both", "all"]

# Device options
VALID_DEVICE_PREFIXES = ["cpu", "cuda", "mps"]

# Activation function options
VALID_ACTIVATIONS = ["identity", "none", "relu", "gelu", "tanh", "silu", "elu"]

# Unknown level handling strategies
VALID_UNKNOWN_STRATEGIES = ["learned", "zero"]

# Large penalty for numerical recovery
LARGE_PENALTY = 1e6
