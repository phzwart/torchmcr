import torch
import numpy as np
from typing import List, Tuple, Optional
import torch.nn.functional as F
from .randomfield import generate_coefficient_maps
from .randomgaussians import generate_spectra

def generate_spatial_spectra(
    n_base_spectra: int,
    lengthscales: List[float],
    size: Tuple[int, int],
    temperature: float = 1.0,
    includes_empty: bool = False,
    N_waves: int = 1000,
    N_peaks: int = 3,
    x_range: Tuple[float, float] = (0, 1000),
    center_limits: Tuple[float, float] = (100, 900),
    variance_limits: Tuple[float, float] = (1, 50),
    amplitude_limits: Tuple[float, float] = (0.5, 1.5),
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate spatially varying spectra based on random fields and base spectra.
    
    Args:
        n_base_spectra: Number of base spectra to generate
        lengthscales: List of lengthscales for coefficient maps
        size: Tuple of (height, width) for the spatial maps
        temperature: Controls the sparsity of the maps. Lower values (< 1.0) 
                    make maps more sparse, higher values make them more uniform
        includes_empty: If True, one base spectrum will be all zeros
        N_waves: Number of wavelength points
        N_peaks: Number of Gaussian peaks per spectrum
        x_range: Range of x values for spectra
        center_limits: Limits for Gaussian peak centers
        variance_limits: Limits for Gaussian peak variances
        amplitude_limits: Limits for Gaussian peak amplitudes
        device: torch device to use
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of:
        - x values (torch.Tensor of shape (N_waves,))
        - spatial spectra (torch.Tensor of shape (N_waves, height, width))
        - base spectra (torch.Tensor of shape (n_base_spectra, N_waves))
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate coefficient maps
    coeff_maps = generate_coefficient_maps(
        size=size,
        n_coefficients=n_base_spectra,
        lengthscales=lengthscales,
        temperature=temperature,
        device=device,
        seed=seed
    )
    
    # Generate base spectra
    actual_spectra_to_generate = n_base_spectra - 1 if includes_empty else n_base_spectra
    x, base_spectra = generate_spectra(
        M_spectra=actual_spectra_to_generate,
        N_peaks=N_peaks,
        N_waves=N_waves,
        x_range=x_range,
        center_limits=center_limits,
        variance_limits=variance_limits,
        amplitude_limits=amplitude_limits
    )
    
    # Convert to torch tensors
    x = torch.from_numpy(x).to(device)
    base_spectra = torch.from_numpy(base_spectra).to(device)
    
    # Add empty spectrum if required
    if includes_empty:
        zero_spectrum = torch.zeros((1, N_waves), device=device)
        base_spectra = torch.cat([base_spectra, zero_spectrum], dim=0)
    
    # Generate spatial spectra through linear combination
    base_spectra_expanded = base_spectra.T.unsqueeze(-1).unsqueeze(-1)
    coeff_maps_expanded = coeff_maps.unsqueeze(0)
    
    # Compute linear combination
    spatial_spectra = (base_spectra_expanded * coeff_maps_expanded).sum(dim=1)
    
    return x, spatial_spectra, base_spectra
