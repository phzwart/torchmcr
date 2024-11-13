import pytest
import torch
import torch.nn as nn
from torchmcr.train import train_mcr_model

@pytest.fixture(params=['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def device(request):
    return request.param

class SimpleMCRModel(nn.Module):
    def __init__(self, n_components=2, n_wavelengths=10, n_timepoints=5):
        super().__init__()
        # Initialize with known values for testing
        self.spectra = nn.Parameter(torch.rand(n_components, n_wavelengths))
        self.weights = nn.Parameter(torch.rand(n_timepoints, n_components))
        
    def forward(self):
        return torch.matmul(self.weights, self.spectra)

def test_train_mcr_model(device):
    print(device)
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create synthetic data
    n_components = 2
    n_wavelengths = 10
    n_timepoints = 5
    
    # Create synthetic ground truth
    true_spectra = torch.rand(n_components, n_wavelengths).to(device)
    true_weights = torch.rand(n_timepoints, n_components).to(device)
    observed_data = torch.matmul(true_weights, true_spectra)
    
    # Create model and move it to the appropriate device
    model = SimpleMCRModel(n_components, n_wavelengths, n_timepoints).to(device)  # Move model to GPU
    
    # Create synthetic ground truth and move it to the same device as the model
    observed_data = observed_data.to(device)  # Move observed data to GPU
    
    # Calculate initial loss
    initial_loss = torch.nn.functional.l1_loss(model(), observed_data)
    
    # Train model with minimal epochs for fast testing
    train_mcr_model(
        model=model,
        observed_data=observed_data,
        num_epochs=10,
        mini_epochs=2,
        lr=0.1,
        show_every=100,
        device=device  # Suppress printing
    )
    
    # Calculate final loss
    final_loss = torch.nn.functional.l1_loss(model(), observed_data)
    
    # Assert that training improved the loss
    assert final_loss < initial_loss
    assert not torch.isnan(final_loss) 
    # Test for convergence
    assert final_loss < initial_loss

    train_mcr_model(
        model=model,
        observed_data=observed_data,
        num_epochs=10,
        mini_epochs=2,
        lr=0.1,
        show_every=100,
        device=None  # Suppress printing
    ) # Calculate final loss
    final_loss = torch.nn.functional.l1_loss(model(), observed_data)
    
    # Assert that training improved the loss
    assert final_loss < initial_loss
    assert not torch.isnan(final_loss) 
    # Test for convergence
    assert final_loss < initial_loss