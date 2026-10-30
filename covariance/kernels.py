import torch
import torch.nn as nn
import numpy as np

def rbf_kernel(x1, x2, lengthscale, jitter=1e-6):
    """
    RBF (Squared Exponential) kernel.
    K(x, y) = exp(-0.5 * ||x-y||^2 / l^2)
    
    Args:
        x1: [N, D]
        x2: [M, D]
        lengthscale: scalar or [D]
    """
    self_kernel = x1 is x2
    if x1.dim() == 1: x1 = x1.unsqueeze(1)
    if x2.dim() == 1: x2 = x2.unsqueeze(1)
    
    # Scale inputs
    x1 = x1 / lengthscale
    x2 = x2 / lengthscale
    
    # Compute distances
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2x'y
    x1_sq = (x1**2).sum(dim=-1, keepdim=True)
    x2_sq = (x2**2).sum(dim=-1, keepdim=True)
    dist_sq = x1_sq + x2_sq.transpose(-2, -1) - 2 * torch.matmul(x1, x2.transpose(-2, -1))
    
    K = torch.exp(-0.5 * dist_sq.clamp(min=0))
    
    if self_kernel:
        K = K + jitter * torch.eye(x1.shape[0], device=K.device, dtype=K.dtype)
        
    return K

def matern_kernel(x1, x2, lengthscale, nu=1.5, jitter=1e-6):
    """
    Matérn kernel.
    nu=0.5: Exponential
    nu=1.5: Once differentiable
    nu=2.5: Twice differentiable
    """
    self_kernel = x1 is x2
    if x1.dim() == 1: x1 = x1.unsqueeze(1)
    if x2.dim() == 1: x2 = x2.unsqueeze(1)
    
    x1 = x1 / lengthscale
    x2 = x2 / lengthscale
    
    x1_sq = (x1**2).sum(dim=-1, keepdim=True)
    x2_sq = (x2**2).sum(dim=-1, keepdim=True)
    dist_sq = x1_sq + x2_sq.transpose(-2, -1) - 2 * torch.matmul(x1, x2.transpose(-2, -1))
    dist = torch.sqrt(dist_sq.clamp(min=1e-12))
    
    if nu == 0.5:
        K = torch.exp(-dist)
    elif nu == 1.5:
        sqrt3 = np.sqrt(3.0)
        K = (1.0 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)
    elif nu == 2.5:
        sqrt5 = np.sqrt(5.0)
        K = (1.0 + sqrt5 * dist + (5.0/3.0) * dist_sq) * torch.exp(-sqrt5 * dist)
    else:
        # Fallback or general case (requires Bessel functions, hard in pure torch)
        K = torch.exp(-0.5 * dist_sq) # default to RBF
        
    if self_kernel:
        K = K + jitter * torch.eye(x1.shape[0], device=K.device, dtype=K.dtype)
        
    return K
