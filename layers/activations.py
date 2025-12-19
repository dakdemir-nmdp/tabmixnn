import torch
import torch.nn as nn
import torch.nn.functional as F

class GEGLULayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

def get_activation(name):
    """Factory for activation functions."""
    if name is None or name == "none" or name.lower() == "identity":
        return nn.Identity()
    
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    
    # Check legacy support for functional
    if name in ["sigmoid"]:
        return nn.Sigmoid()
        
    # Return Identity as safe fallback or raise?
    # Existing code raised ValueError, let's stick to that but expanded.
    raise ValueError(f"Unknown activation '{name}'. Supported: identity, relu, gelu, tanh, silu.")

# Alias for backward compatibility if needed, but get_activation returns Modules not functions now
resolve_activation = get_activation 
