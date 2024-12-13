import torch
import numpy as np
from typing import List, Tuple, Optional
import torch.nn.functional as F

def generate_coefficient_maps(
    size: Tuple[int, int],
    n_coefficients: int,
    lengthscales: List[float],
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate random coefficient maps that sum to 1 and are positive.
    
    Args:
        size: Tuple of (height, width) for the output maps
        n_coefficients: Number of coefficient maps to generate
        lengthscales: List of lengthscales for each coefficient map
        temperature: Controls the sparsity of the maps. Lower values (< 1.0) 
                    make maps more sparse, higher values make them more uniform
        device: torch device to use
        seed: Random seed for reproducibility
    
    Returns:
        torch.Tensor of shape (n_coefficients, height, width) containing the coefficient maps
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if seed is not None:
        torch.manual_seed(seed)
        
    if len(lengthscales) != n_coefficients:
        raise ValueError("Number of lengthscales must match number of coefficients")

    # Generate frequency grids
    freq_y = torch.fft.fftfreq(size[0], d=1.0).to(device)
    freq_x = torch.fft.fftfreq(size[1], d=1.0).to(device)
    freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
    freq_grid = torch.sqrt(freq_grid_y**2 + freq_grid_x**2)

    maps = []
    for ls in lengthscales:
        # Generate power spectrum
        power_spectrum = torch.exp(-2 * (np.pi * freq_grid * ls)**2)
        
        # Generate random complex numbers
        real_part = torch.randn(size, device=device)
        imag_part = torch.randn(size, device=device)
        fourier_space = (real_part + 1j * imag_part) * torch.sqrt(power_spectrum)
        
        # Inverse FFT and take real part
        field = torch.fft.ifft2(fourier_space).real
        
        # Normalize to zero mean and unit variance
        field = (field - field.mean()) / field.std()
        
        maps.append(field)

    # Stack maps
    maps = torch.stack(maps)
    
    # Apply temperature-controlled softmax to ensure positive values that sum to 1
    maps = F.softmax(maps / temperature, dim=0)
    
    return maps
