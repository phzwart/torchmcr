import torch
import torch.nn.functional as F
from .loss_components import (
    MCRLossComponent,
    DataLoss,
    SmoothnessPriorWeights,
    SmoothnessPriorSpectra,
    TopKWeightProductSumPrior
)
from .loss_registry import MCRLossRegistry

class BaseMSELoss(MCRLossComponent):
    """Basic MSE loss component for MCR."""
    def __init__(self, loss_fn=F.mse_loss):
        super().__init__("base_mse")
        self.loss_fn = loss_fn
    
    def __call__(self, mcr_model, predicted, target, **kwargs):
        return self.loss_fn(predicted, target)

def create_smooth_loss(
    base_loss_fn=F.l1_loss,
    smooth_spectra_weight=0.1,
    smooth_weight_weight=0.1,
    weight_cross_product_weight=0.0,
    smoothness_power=2.0
):
    """Factory function to create a configured loss registry with smoothness components.
    
    Parameters
    ----------
    base_loss_fn : callable
        Base loss function (default: F.l1_loss)
    smooth_spectra_weight : float
        Weight for spectral smoothness penalty
    smooth_weight_weight : float
        Weight for concentration profile smoothness penalty
    weight_cross_product_weight : float
        Weight for weight product penalty
    smoothness_power : float
        Power for smoothness penalties (default: 2.0 for squared differences)
    
    Returns
    -------
    MCRLossRegistry
        Configured loss registry with all components
    """
    registry = MCRLossRegistry()
    
    # Register all components with their weights
    registry.register(
        "base",
        BaseMSELoss(base_loss_fn),
        weight=1.0
    )
    
    registry.register(
        "spectra_smoothness",
        SmoothnessPriorSpectra(
            smoothness_factor=1.0,
            power=smoothness_power
        ),
        weight=smooth_spectra_weight
    )
    
    registry.register(
        "weight_smoothness",
        SmoothnessPriorWeights(
            smoothness_factor=1.0,
            power=smoothness_power
        ),
        weight=smooth_weight_weight
    )
    
    registry.register(
        "weight_products",
        TopKWeightProductSumPrior(factor=1.0, k=2),
        weight=weight_cross_product_weight
    )
    
    return registry

