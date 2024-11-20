import torch
import torch.nn.functional as F

class MCRLossComponent:
    """Base class for MCR loss components.
    
    This class serves as an abstract base class for implementing various loss components
    that can be used with MCR models. Each component should implement the __call__ method.
    
    Parameters
    ----------
    name : str
        Unique identifier for the loss component
    """
    def __init__(self, name: str):
        self.name = name
    
    def __call__(self, mcr_model, **kwargs):
        """Compute the loss value.
        
        Parameters
        ----------
        mcr_model : MCR
            The MCR model instance to compute loss for
        **kwargs : dict
            Additional arguments needed for loss computation
            
        Returns
        -------
        torch.Tensor
            The computed loss value
            
        Raises
        ------
        NotImplementedError
            If the child class does not implement __call__
        """
        raise NotImplementedError("Loss components must implement __call__")

class DataLoss(MCRLossComponent):
    """Loss component for measuring reconstruction error between predicted and target data.
    
    Computes MSE loss between the MCR model output and target data.
    """
    def __init__(self):
        super().__init__("data_loss")
    
    def __call__(self, mcr_model, target, **kwargs):
        """Compute MSE loss between model prediction and target.
        
        Parameters
        ----------
        mcr_model : MCR
            The MCR model instance
        target : torch.Tensor
            Ground truth target data
        **kwargs : dict
            Additional arguments passed to mcr_model.forward()
            
        Returns
        -------
        torch.Tensor
            MSE loss value
        """
        pred = mcr_model(**kwargs)
        return torch.nn.functional.mse_loss(pred, target)

class SmoothnessPriorWeights(MCRLossComponent):
    """Loss component that penalizes non-smooth weights.
    
    Adds regularization to encourage smoothness in the weight components by
    penalizing large differences between adjacent weights.
    
    Parameters
    ----------
    smoothness_factor : float, optional
        Weight factor for the smoothness penalty (default is 1.0)
    power : float, optional
        Power to raise the absolute differences to (default is 1.0)
    """
    def __init__(self, 
                 smoothness_factor: float = 1.0, 
                 power: float = 1.0):
        super().__init__("smoothness")
        self.factor = smoothness_factor
        self.power = power
    
    def __call__(self, mcr_model, **kwargs):
        """Compute smoothness penalty on weights.
        
        Parameters
        ----------
        mcr_model : MCR
            The MCR model instance
        **kwargs : dict
            Additional arguments, may include 'weights_kwargs' dict
            
        Returns
        -------
        torch.Tensor
            Weighted mean squared difference between adjacent weight points
        """
        weights = mcr_model.weights(**kwargs.get('weights_kwargs', {}))
        weights_scale = torch.mean(torch.abs(weights))

        # Calculate smoothness penalty
        diff = torch.abs(weights[1:, :] - weights[:-1, :]) / weights_scale
        return self.factor * torch.mean(diff ** self.power)
    
class SmoothnessPriorSpectra(MCRLossComponent):
    """Loss component that penalizes non-smooth spectra.
    
    Adds regularization to encourage smoothness in the spectral components by
    penalizing large differences between adjacent wavelengths.
    
    Parameters
    ----------
    smoothness_factor : float, optional
        Weight factor for the smoothness penalty (default is 1.0)
    power : float, optional
        Power to raise the absolute differences to (default is 1.0)
    """
    def __init__(self, 
                 smoothness_factor: float = 1.0, 
                 power: float = 1.0):
        super().__init__("smoothness")
        self.factor = smoothness_factor
        self.power = power
    
    def __call__(self, mcr_model, **kwargs):
        """Compute smoothness penalty on spectra."""
        spectra = mcr_model.spectra(**kwargs.get('spectra_kwargs', {}))
        spectra_scale = torch.mean(torch.abs(spectra))

        # Calculate smoothness penalty
        diff = torch.abs(spectra[:, 1:] - spectra[:, :-1]) / spectra_scale
        return self.factor * torch.mean(diff ** self.power)
    

class WeightProductSumPrior(MCRLossComponent):
    """Loss component that penalizes the sum of weight products across time.
    
    For each time point, multiplies all weights together and then sums across
    all time points to create a regularization term.
    
    Parameters
    ----------
    factor : float, optional
        Weight factor for the penalty (default is 1.0)
    """
    def __init__(self, factor: float = 1.0):
        super().__init__("weight_product_sum")
        self.factor = factor
    
    def __call__(self, mcr_model, **kwargs):
        """Compute weight product sum penalty.
        
        Parameters
        ----------
        mcr_model : MCR
            The MCR model instance
        **kwargs : dict
            Additional arguments, may include 'weights_kwargs' dict
            
        Returns
        -------
        torch.Tensor
            Sum of weight products across time points
        """
        weights = mcr_model.weights(**kwargs.get('weights_kwargs', {}))
        
        # Multiply weights along component dimension for each time point
        weight_products = torch.prod(weights, dim=-1)  # Shape: (N,)
        
        # Sum across time points
        product_sum = torch.sum(weight_products)
        
        return self.factor * product_sum

class TopKWeightProductSumPrior(MCRLossComponent):
    """Loss component that penalizes the sum of products of top k weights across time.
    
    For each time point, multiplies the k largest weights together and then sums across
    all time points to create a regularization term.
    
    Parameters
    ----------
    factor : float, optional
        Weight factor for the penalty (default is 1.0)
    k : int, optional
        Number of largest weights to use in product (default is 2)
    """
    def __init__(self, factor: float = 1.0, k: int = 2):
        super().__init__("top_k_weight_product_sum")
        self.factor = factor
        self.k = k
    
    def __call__(self, mcr_model, **kwargs):
        """Compute product sum penalty using k largest weights.
        
        Parameters
        ----------
        mcr_model : MCR
            The MCR model instance
        **kwargs : dict
            Additional arguments, may include 'weights_kwargs' dict
            
        Returns
        -------
        torch.Tensor
            Sum of products of k largest weights across time points
        """
        weights = mcr_model.weights(**kwargs.get('weights_kwargs', {}))
        
        # Get top k values along component dimension for each time point
        top_k_weights, _ = torch.topk(weights, k=min(self.k, weights.shape[-1]), dim=-1)
        
        # Multiply top k weights along component dimension for each time point
        weight_products = torch.prod(top_k_weights, dim=-1)  # Shape: (N,)
        
        # Sum across time points
        product_sum = torch.mean(weight_products)
        
        return self.factor * product_sum


class ThresholdedWeightProductSumPrior(MCRLossComponent):
    """Loss component that penalizes the sum of weight products across time,
    but only when the product exceeds a threshold value.
    
    For each time point, multiplies all weights together and applies the penalty
    only if the product is above the threshold value.
    
    Parameters
    ----------
    factor : float, optional
        Weight factor for the penalty (default is 1.0)
    threshold : float, optional
        Threshold value below which no penalty is applied (default is 0.1)
    """
    def __init__(self, factor: float = 1.0, threshold: float = 0.1):
        super().__init__("thresholded_weight_product_sum")
        self.factor = factor
        self.threshold = threshold
    
    def __call__(self, mcr_model, **kwargs):
        """Compute thresholded weight product sum penalty.
        
        Parameters
        ----------
        mcr_model : MCR
            The MCR model instance
        **kwargs : dict
            Additional arguments, may include 'weights_kwargs' dict
            
        Returns
        -------
        torch.Tensor
            Sum of weight products across time points where product > threshold
        """
        weights = mcr_model.weights(**kwargs.get('weights_kwargs', {}))
        
        # Multiply weights along component dimension for each time point
        weight_products = torch.prod(weights, dim=-1)  # Shape: (N,)
        
        # Zero out products below threshold using mask
        mask = (weight_products > self.threshold)
        masked_products = weight_products * mask
        
        # Sum across time points
        product_sum = torch.sum(masked_products)
        
        return self.factor * product_sum
