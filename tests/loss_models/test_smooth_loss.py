import pytest
import torch
import torch.nn.functional as F
from torchmcr.loss_models.smooth_loss import BaseMSELoss, create_smooth_loss

class TestBaseMSELoss:
    def test_init(self):
        loss = BaseMSELoss()
        assert loss.name == "base_mse"
        assert loss.loss_fn == F.mse_loss

    def test_call(self):
        loss = BaseMSELoss()
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = loss(None, predicted, target)
        assert torch.isclose(result, torch.tensor(0.0))

        # Test with different values
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        result = loss(None, predicted, target)
        assert result > 0.0

class TestCreateSmoothLoss:
    def test_default_parameters(self):
        registry = create_smooth_loss()
        
        # Check if all components are registered
        assert "base" in registry.losses
        assert "spectra_smoothness" in registry.losses
        assert "weight_smoothness" in registry.losses
        assert "weight_products" in registry.losses
        
        # Check default weights
        assert registry.weights["base"] == 1.0
        assert registry.weights["spectra_smoothness"] == 0.1
        assert registry.weights["weight_smoothness"] == 0.1
        assert registry.weights["weight_products"] == 0.0

    def test_custom_parameters(self):
        registry = create_smooth_loss(
            base_loss_fn=F.l1_loss,
            smooth_spectra_weight=0.2,
            smooth_weight_weight=0.3,
            weight_cross_product_weight=0.4,
            smoothness_power=3.0
        )
        
        # Check if weights are properly set
        assert registry.weights["spectra_smoothness"] == 0.2
        assert registry.weights["weight_smoothness"] == 0.3
        assert registry.weights["weight_products"] == 0.4

    def test_loss_calculation(self):
        registry = create_smooth_loss()
        
        # Create dummy data
        batch_size = 2
        n_components = 3
        n_wavelengths = 4
        n_timepoints = 5
        
        # Create mock MCR model with spectra as a property/method
        class MockMCRModel:
            def __init__(self):
                self._spectra = torch.rand(n_components, n_wavelengths, requires_grad=True)
                self._weights = torch.rand(batch_size, n_timepoints, n_components, requires_grad=True)
            
            def spectra(self, **kwargs):
                return self._spectra
            def weights(self, **kwargs):
                return self._weights
                
        mcr_model = MockMCRModel()
        predicted = torch.rand(batch_size, n_timepoints, n_wavelengths, requires_grad=True)
        target = torch.rand(batch_size, n_timepoints, n_wavelengths)
        
        # Calculate total loss
        total_loss, _ = registry.compute_total_loss(
            mcr_model, 
            predicted=predicted, 
            target=target
        )
        
        # Check if loss is a scalar and has gradient
        assert isinstance(total_loss.item(), float)
        assert total_loss.requires_grad
