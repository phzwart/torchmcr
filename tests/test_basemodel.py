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

def test_freeze_weights_rows(mcr_model):
    mcr_model.freeze_weights(row_indices=[0])
    weight_matrix = next(mcr_model.weights.parameters())
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert torch.all(weight_matrix.grad[0, :] == 0)  # Frozen row should have zero gradient
    assert torch.any(weight_matrix.grad[1, :] != 0)  # Non-frozen row should have non-zero gradient

def test_freeze_weights_cols(mcr_model):
    mcr_model.freeze_weights(col_indices=[0])
    weight_matrix = next(mcr_model.weights.parameters())
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert torch.all(weight_matrix.grad[:, 0] == 0)  # Frozen column should have zero gradient
    assert torch.any(weight_matrix.grad[:, 1] != 0)  # Non-frozen column should have non-zero gradient

def test_freeze_weights_coords(mcr_model):
    mcr_model.freeze_weights(coords=[(0, 0), (1, 1)])
    weight_matrix = next(mcr_model.weights.parameters())
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert weight_matrix.grad[0, 0].item() == 0  # Specific coordinate should have zero gradient
    assert weight_matrix.grad[1, 1].item() == 0  # Specific coordinate should have zero gradient
    assert weight_matrix.grad[0, 1].item() != 0  # Non-frozen element should have non-zero gradient
    assert weight_matrix.grad[1, 0].item() != 0  # Non-frozen element should have non-zero gradient

def test_unfreeze_weights(mcr_model):
    mcr_model.freeze_weights(row_indices=[0], col_indices=[1])
    mcr_model.unfreeze_weights(row_indices=[0], col_indices=[1])
    weight_matrix = next(mcr_model.weights.parameters())
    
    # Ensure gradients are recalculated properly
    weight_matrix.requires_grad_(True)
    
    # Reset gradients
    weight_matrix.grad = None
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert torch.any(weight_matrix.grad[0, :] != 0)  # Unfrozen row should have non-zero gradient
    assert torch.any(weight_matrix.grad[:, 1] != 0)  # Unfrozen column should have non-zero gradient

def test_freeze_spectra_rows(mcr_model):
    mcr_model.freeze_spectra(row_indices=[0])
    spectra_matrix = next(mcr_model.spectra.parameters())
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert torch.all(spectra_matrix.grad[0, :] == 0)  # Frozen row should have zero gradient
    assert torch.any(spectra_matrix.grad[1, :] != 0)  # Non-frozen row should have non-zero gradient

def test_freeze_spectra_cols(mcr_model):
    mcr_model.freeze_spectra(col_indices=[0])
    spectra_matrix = next(mcr_model.spectra.parameters())
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert torch.all(spectra_matrix.grad[:, 0] == 0)  # Frozen column should have zero gradient
    assert torch.any(spectra_matrix.grad[:, 1] != 0)  # Non-frozen column should have non-zero gradient

def test_freeze_spectra_coords(mcr_model):
    mcr_model.freeze_spectra(coords=[(0, 0), (1, 1)])
    spectra_matrix = next(mcr_model.spectra.parameters())
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert spectra_matrix.grad[0, 0].item() == 0  # Specific coordinate should have zero gradient
    assert spectra_matrix.grad[1, 1].item() == 0  # Specific coordinate should have zero gradient
    assert spectra_matrix.grad[0, 1].item() != 0  # Non-frozen element should have non-zero gradient
    assert spectra_matrix.grad[1, 0].item() != 0  # Non-frozen element should have non-zero gradient

def test_unfreeze_spectra(mcr_model):
    mcr_model.freeze_spectra(row_indices=[0], col_indices=[1])
    mcr_model.unfreeze_spectra(row_indices=[0], col_indices=[1])
    spectra_matrix = next(mcr_model.spectra.parameters())
    
    # Ensure gradients are recalculated properly
    spectra_matrix.requires_grad_(True)
    
    # Reset gradients
    spectra_matrix.grad = None
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert torch.any(spectra_matrix.grad[0, :] != 0)  # Unfrozen row should have non-zero gradient
    assert torch.any(spectra_matrix.grad[:, 1] != 0)  # Unfrozen column should have non-zero gradient

def test_freeze_all_weights(mcr_model):
    mcr_model.freeze_weights()  # Freeze all weights
    weight_matrix = next(mcr_model.weights.parameters())
    
    # Perform a forward and backward pass
    output = mcr_model.forward()
    output.sum().backward()
    
    # Check gradients
    assert torch.all(weight_matrix.grad == 0)  # All weights should have zero gradient

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
    assert torch.all(spectra_matrix.grad == 0)  # All spectra should have zero gradient

def test_unfreeze_all_spectra(mcr_model):
    mcr_model.freeze_spectra()  # Freeze all spectra first
    mcr_model.unfreeze_spectra()  # Then unfreeze all spectra
    spectra_matrix = next(mcr_model.spectra.parameters())

def test_freeze_unfreeze_with_coords():
    """Test freezing and unfreezing specific coordinates in weights and spectra"""
    # Create dummy weights and spectra modules
    class DummyModule(nn.Module):
        def __init__(self, matrix):
            super().__init__()
            self.matrix = nn.Parameter(matrix)
        def forward(self):
            return self.matrix

    weights = DummyModule(torch.randn(4, 3))
    spectra = DummyModule(torch.randn(3, 5))
    
    # Initialize MCR model
    model = MCR(weights, spectra)
    
    # Test coordinates
    coords = [(0, 1), (2, 0)]
    
    # Test freezing specific coordinates
    model.freeze_weights(coords=coords)
    assert model.weights_grad_mask[0, 1] == 0
    assert model.weights_grad_mask[2, 0] == 0
    
    # Test unfreezing specific coordinates
    model.unfreeze_weights(coords=coords)
    assert model.weights_grad_mask[0, 1] == 1
    assert model.weights_grad_mask[2, 0] == 1
    
    # Test with spectra
    model.freeze_spectra(coords=coords)
    assert model.spectra_grad_mask[0, 1] == 0
    assert model.spectra_grad_mask[2, 0] == 0
    
    model.unfreeze_spectra(coords=coords)
    assert model.spectra_grad_mask[0, 1] == 1
    assert model.spectra_grad_mask[2, 0] == 1



