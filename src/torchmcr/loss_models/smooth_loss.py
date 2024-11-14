import torch
import torch.nn.functional as F
from torch import nn

class SmoothLoss(torch.nn.Module):
    def __init__(self, 
                 base_loss_fn=F.l1_loss, 
                 smooth_spectra_weight=0.1,
                 smooth_weight_weight=0.1,
                 weight_cross_product_weight=0.0                 
                ):
        """
        A custom loss function that combines a base loss with smoothness penalties for both spectra and weights.

        Parameters
        ----------
        base_loss_fn : callable, optional
            The base loss function to use (e.g. F.l1_loss, F.mse_loss), by default F.l1_loss
        smooth_spectra_weight : float, optional
            Weight factor for the spectra smoothness penalty term, by default 0.1
        smooth_weight_weight : float, optional
            Weight factor for the weights smoothness penalty term, by default 0.1
        weight_cross_product_weight : float, optional
            Weight factor for the weights cross-product penalty term, by default 0.0

        Notes
        -----
        The total loss is computed as:
        total_loss = base_loss + smooth_spectra_weight * spectra_penalty + smooth_weight_weight * weights_penalty
        
        where the penalties are calculated using finite differences between adjacent elements.


        Examples
        --------
        To use this loss function in a training script:

        >>> # Create loss function with default parameters
        >>> loss_fn = CustomLoss()
        >>> 
        >>> # Or customize with lambda for specific parameters
        >>> loss_fn = lambda pred, target, spectra: CustomLoss(
        ...     base_loss_fn=F.mse_loss,
        ...     smooth_spectra_weight=0.2,
        ...     smooth_weight_weight=0.3
        ... )(pred, target, spectra)
        """
        super(SmoothLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.smooth_weight_weight = smooth_weight_weight
        self.smooth_spectra_weight = smooth_spectra_weight 
        self.weight_cross_product_weight = weight_cross_product_weight

    def forward(self, predicted, target, spectra, weights):
        """
        Compute the total loss combining base loss and smoothness penalties.

        Parameters
        ----------
        predicted : torch.Tensor
            The predicted output from the model
        target : torch.Tensor
            The ground truth target data
        spectra : torch.Tensor
            The spectra matrix to compute smoothness penalties on
        weights: torch.Tensor
            The weights matrix to compute smoothness penalties on

        Returns
        -------
        torch.Tensor
            The computed total loss value

        Notes
        -----
        The smoothness penalties are normalized by the mean absolute value of the input
        to make them scale-invariant.
        """
        # Compute the base reconstruction loss
        base_loss = self.base_loss_fn(predicted, target)

        # Normalize by mean absolute value to make scale-invariant
        spectra_scale = torch.mean(torch.abs(spectra))
        weights_scale = torch.mean(torch.abs(weights))
        
        # Compute smoothness penalties using finite differences
        diff_spectra = (spectra[:, 1:] - spectra[:, :-1]) / spectra_scale
        smoothness_penalty_spectra = torch.mean(diff_spectra ** 2)
        
        diff_weights = (weights[1:, :] - weights[:-1, :]) / weights_scale  
        smoothness_penalty_weights = torch.mean(diff_weights ** 2)


        # Compute weight cross-product penalty by multiplying weights along N dimension for each K
        weight_products = torch.prod(weights, dim=1)  # Shape: (K,)
        weight_cross_product_penalty = torch.mean(weight_products)
        
        # Combine losses with weights
        total_loss = base_loss
        total_loss = total_loss + self.smooth_weight_weight * smoothness_penalty_weights
        total_loss = total_loss + self.smooth_spectra_weight * smoothness_penalty_spectra
        total_loss = total_loss + self.weight_cross_product_weight * weight_cross_product_penalty
        return total_loss

