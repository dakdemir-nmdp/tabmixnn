import torch.nn as nn
from abc import ABC, abstractmethod

class TabMixNNBackbone(nn.Module, ABC):
    """
    Abstract base class for TabMixNN backbones.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim # Default, subclasses should update

    @abstractmethod
    def forward(self, x, context=None):
        """
        Args:
            x: Input tensor [batch, input_dim]
            context: Optional dict containing 'Z_indices', 'Z_slopes', 'random_config' etc.
        Returns:
            latent: Output tensor [batch, output_dim]
            aux_loss: Auxiliary loss (e.g. sparsity penalty) calculated per forward pass if needed
                      (though ideally penalties are parameter-based and retrieved via .penalty())
        """
        pass

    def penalty(self):
        """
        Calculate regularization penalty (e.g. SAM sparsity, SPDE priors).
        Should not depend on batch data if possible.
        """
        return 0.0
