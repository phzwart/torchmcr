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


import torch

def simplisma(data, n_components, noise_factor=0.1, normalize=True):
    """
    Implements SIMPLISMA (SIMPLe-to-use Interactive Self-modeling Mixture Analysis) 
    for finding pure spectra in spectroscopic data.

    Parameters:
        data (torch.Tensor): Input data matrix of shape (n_samples, n_wavenumbers)
        n_components (int): Number of pure components to find
        noise_factor (float): Factor to handle noise in the data (default 0.1)
        normalize (bool): Whether to normalize the data (default True)

    Returns:
        torch.Tensor: Initial estimates of pure component spectra (n_components, n_wavenumbers)
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Move to same device as input
    device = data.device
    
    # Normalize if requested
    if normalize:
        data = data / (torch.max(data) + 1e-8)
    
    n_samples, n_wavenumbers = data.shape
    
    # Calculate mean and standard deviation for each sample (spectrum)
    mean = torch.mean(data, dim=1)  # Shape: (n_samples,)
    std = torch.std(data, dim=1)    # Shape: (n_samples,)
    
    # Calculate relative standard deviation for each sample
    rel_std = std / (mean + noise_factor * torch.mean(mean))
    
    # Initialize array for pure spectra indices
    pure_spectra_indices = torch.zeros(n_components, dtype=torch.long, device=device)
    
    # Find first pure spectrum (highest relative standard deviation)
    pure_spectra_indices[0] = torch.argmax(rel_std)
    
    # Find remaining pure spectra
    for i in range(1, n_components):
        # Calculate weights based on previous pure spectra
        weights = torch.ones_like(rel_std)
        
        for j in range(i):
            # Get data for the selected pure spectrum
            pure_spectrum = data[pure_spectra_indices[j], :].unsqueeze(0)  # Shape: (1, n_wavenumbers)
            
            # Compute mean and standard deviation across wavelengths
            mean_pure = pure_spectrum.mean(dim=1)      # Shape: (1,)
            mean_data = data.mean(dim=1)               # Shape: (n_samples,)
            std_pure = pure_spectrum.std(dim=1)        # Shape: (1,)
            std_data = data.std(dim=1)                 # Shape: (n_samples,)
            
            # Compute covariance between pure spectrum and all spectra
            covariance = ((data - mean_data.unsqueeze(1)) * (pure_spectrum - mean_pure.unsqueeze(1))).sum(dim=1) / (n_wavenumbers - 1)
            
            # Compute correlation coefficient
            corr = covariance / (std_data * std_pure.squeeze() + 1e-8)
            
            # Update weights
            weights *= (1 - corr.abs())
        
        # Multiply weights by relative standard deviation
        purity = weights * rel_std
        
        # Find next pure spectrum
        pure_spectra_indices[i] = torch.argmax(purity)
    
    # Extract pure spectra
    pure_spectra = data[pure_spectra_indices, :]  # Shape: (n_components, n_wavenumbers)
    
    return pure_spectra
