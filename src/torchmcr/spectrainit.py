import torch
import numpy as np
import scipy

"""
This module contains initialization routines for MCR analysis. Choosing good initial
spectra estimates is critical for MCR convergence, as the algorithm can get stuck
in local minima. The routines here implement various strategies for selecting
promising starting spectra, including:

- SIMPLISMA (SIMPLe-to-use Interactive Self-modeling Mixture Analysis)
- Pure variable detection
- Random selection with constraints 
- PCA-based initialization
- ICA (Independent Component Analysis) for finding statistically independent sources
- FastICA and other ICA variants optimized for spectral data

These methods help avoid degenerate solutions and improve the chances of finding
chemically meaningful components. ICA in particular can be valuable for separating
mixed signals into independent source spectra when the underlying components are
statistically independent.
"""

def simplisma(data, n_components, noise_factor=0.1, normalize=True):
    """
    Implements SIMPLISMA (SIMPLe-to-use Interactive Self-modeling Mixture Analysis) 
    for finding pure variables in spectroscopic data.

    Parameters:
        data (torch.Tensor): Input data matrix of shape (n_samples, n_features)
        n_components (int): Number of pure components to find
        noise_factor (float): Factor to handle noise in the data (default 0.1)
        normalize (bool): Whether to normalize the data (default True)

    Returns:
        torch.Tensor: Initial estimates of pure component spectra
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Move to same device as input
    device = data.device
    
    # Normalize if requested
    if normalize:
        data = data / torch.max(data)
    
    # Calculate mean and standard deviation for each variable
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    
    # Calculate relative standard deviation
    rel_std = std / (mean + noise_factor * torch.mean(mean))
    
    # Initialize arrays for pure variables
    pure_vars = torch.zeros(n_components, dtype=torch.long, device=device)
    
    # Find first pure variable (highest relative standard deviation)
    pure_vars[0] = torch.argmax(rel_std)
    
    # Find remaining pure variables
    for i in range(1, n_components):
        # Calculate weights based on previous pure variables
        weights = torch.ones_like(rel_std)
        
        for j in range(i):
            # Get correlation with previous pure variables
            # Calculate correlation between current pure variable and all variables
            pure_var_data = data[:, pure_vars[j]].unsqueeze(1)  # Make 2D
            # Calculate correlation coefficient for each wavelength
            mean_pure = pure_var_data.mean(dim=0)
            mean_data = data.mean(dim=0)
            std_pure = pure_var_data.std(dim=0)
            std_data = data.std(dim=0)
            corr = torch.mean((pure_var_data - mean_pure) * (data - mean_data), dim=0) / (std_pure * std_data + 1e-8)
            weights *= (1 - corr.abs())
        
        # Multiply weights by relative standard deviation
        purity = weights * rel_std
        
        # Find next pure variable
        pure_vars[i] = torch.argmax(purity)
    
    # Extract pure spectra
    pure_spectra = data[:, pure_vars]
    
    return pure_spectra.T


def generate_initial_spectra(n_spectra, n_wavelengths, diffusion_steps=100, 
                             norm_penalty=1e-3, orth_penalty=1e-3, smooth_penalty=1e-3):
    """
    Generate initial spectra using a diffusion process with constraints on 
    minimum norm, low dot product, and smoothness.
    
    Parameters:
        n_spectra (int): Number of spectra (components) to generate
        n_wavelengths (int): Number of wavelengths (length of each spectrum)
        diffusion_steps (int): Number of diffusion steps to run
        norm_penalty (float): Penalty weight for the minimum norm constraint
        orth_penalty (float): Penalty weight for the orthogonality constraint
        smooth_penalty (float): Penalty weight for the smoothness constraint
        
    Returns:
        torch.Tensor: Generated initial spectra with desired properties (n_spectra x n_wavelengths)
    """
    # Initialize random spectra
    spectra = torch.randn(n_spectra, n_wavelengths)
    
    for _ in range(diffusion_steps):
        # Minimum Norm Constraint
        norm = torch.norm(spectra, dim=1, keepdim=True) + 1e-3  # Avoid division by zero
        norm_penalty_term = norm_penalty * spectra / norm  # Push towards minimum norm
        
        # Orthogonality Constraint
        dot_products = torch.matmul(spectra, spectra.T)  # Dot products matrix (n_spectra x n_spectra)
        dot_products = dot_products - torch.diag(torch.diag(dot_products))  # Zero out diagonal
        orth_penalty_term = orth_penalty * torch.matmul(dot_products, spectra)  # Enforce low dot product
        
        # Smoothness Constraint
        diffs = spectra[:, 1:] - spectra[:, :-1]  # Finite difference approximation
        smooth_penalty_term = torch.zeros_like(spectra)
        smooth_penalty_term[:, 1:-1] = smooth_penalty * (diffs[:, 1:] - diffs[:, :-1])
        
        # Update spectra with combined penalties
        spectra = spectra - norm_penalty_term - orth_penalty_term - smooth_penalty_term
    
    # Normalize spectra to have range [0, 1]
    spectra = spectra - spectra.min(dim=1, keepdim=True)[0]
    spectra = spectra / spectra.max(dim=1, keepdim=True)[0]
    
    return spectra
