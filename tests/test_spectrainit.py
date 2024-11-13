import pytest
import torch
import numpy as np
from torchmcr.spectrainit import simplisma

@pytest.fixture
def sample_data():
    # Create synthetic data with known pure components
    wavelengths = 100
    samples = 50
    
    # Create three distinct gaussian peaks as pure components
    x = np.linspace(0, wavelengths, wavelengths)
    component1 = np.exp(-(x - 20)**2 / 50)
    component2 = np.exp(-(x - 50)**2 / 50)
    component3 = np.exp(-(x - 80)**2 / 50)
    
    # Create mixture data
    pure_components = np.vstack([component1, component2, component3])
    concentrations = np.random.rand(samples, 3)
    data = np.dot(concentrations, pure_components)
    
    return torch.tensor(data, dtype=torch.float32)

def test_simplisma_basic():
    # Test with simple synthetic data
    data = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
    ], dtype=torch.float32)
    
    result = simplisma(data, n_components=3)
    
    assert result.shape == (3, 3)
    assert torch.allclose(torch.sum(result, dim=1), torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)

def test_simplisma_with_sample_data(sample_data):
    n_components = 3
    result = simplisma(sample_data, n_components=n_components)
    
    # Check output shape
    assert result.shape == (n_components, sample_data.shape[1])
    
    # Check if results are normalized
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)

def test_simplisma_input_validation():
    # Test with numpy array input
    data = np.random.rand(10, 5)
    result = simplisma(data, n_components=2)
    assert isinstance(result, torch.Tensor)
    
    # Test with invalid n_components
    with pytest.raises(Exception):
        simplisma(data, n_components=0)
