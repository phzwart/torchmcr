import pytest
import torch
import torch.nn as nn
from torchmcr.basemodel import MCR

class DummyModule(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = nn.Parameter(matrix)
    
    def forward(self, **kwargs):
        return self.matrix

@pytest.fixture
def mcr_model():
    # Create dummy weight and spectra matrices
    weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]], requires_grad=True)
    spectra = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    
    weights_module = DummyModule(weights)
    spectra_module = DummyModule(spectra)
    
    return MCR(weights_module, spectra_module)

def test_mcr_initialization(mcr_model):
    assert isinstance(mcr_model, nn.Module)
    assert isinstance(mcr_model.weights, nn.Module)
    assert isinstance(mcr_model.spectra, nn.Module)

def test_mcr_forward(mcr_model):
    result = mcr_model.forward()
    expected = torch.tensor([
        [2.5, 3.5, 4.5],
        [3.1, 4.1, 5.1]
    ])
    assert torch.allclose(result, expected, rtol=1e-4)

def test_freeze_all_weights(mcr_model):
    mcr_model.freeze_weights()  # Freeze all weights
    weight_matrix = next(mcr_model.weights.parameters())
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    if weight_matrix.grad is not None:
        assert torch.all(weight_matrix.grad == torch.zeros_like(weight_matrix.grad))  # All weights should have zero gradient
    else:
        assert True  # If grad is None, it means no gradient was computed, which is expected for frozen weights

def test_unfreeze_all_weights(mcr_model):
    mcr_model.freeze_weights()  # Freeze all weights first
    mcr_model.unfreeze_weights()  # Then unfreeze all weights
    weight_matrix = next(mcr_model.weights.parameters())
    
    # Reset gradients
    weight_matrix.grad = None
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert torch.any(weight_matrix.grad != 0)  # All weights should have non-zero gradient

def test_freeze_all_spectra(mcr_model):
    mcr_model.freeze_spectra()  # Freeze all spectra
    spectra_matrix = next(mcr_model.spectra.parameters())
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    if spectra_matrix.grad is not None:
        assert torch.all(spectra_matrix.grad == torch.zeros_like(spectra_matrix.grad))  # All spectra should have zero gradient
    else:
        assert True  # If grad is None, it means no gradient was computed, which is expected for frozen spectra

def test_unfreeze_all_spectra(mcr_model):
    mcr_model.freeze_spectra()  # Freeze all spectra first
    mcr_model.unfreeze_spectra()  # Then unfreeze all spectra
    spectra_matrix = next(mcr_model.spectra.parameters())
    
    # Reset gradients
    spectra_matrix.grad = None
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert torch.any(spectra_matrix.grad != 0)  # All spectra should have non-zero gradient
