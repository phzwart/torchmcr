import torch

class MCRLossRegistry(object):
    """A registry for managing and combining multiple loss functions for MCR models.
    
    This class allows registration of multiple loss functions with associated weights,
    and computes their weighted sum during training.
    """
    
    def __init__(self):
        """Initialize empty dictionaries for storing loss functions and their weights."""
        self.losses = {}
        self.weights = {}
    
    def register(self, name: str, loss_fn, weight: float = 1.0):
        """Register a new loss component with an optional weight.
        
        Parameters
        ----------
        name : str
            Unique identifier for the loss function
        loss_fn : callable
            The loss function to register
        weight : float, optional
            Weight factor for this loss component (default is 1.0)
        """
        self.losses[name] = loss_fn
        self.weights[name] = weight
    
    def unregister(self, name: str):
        """Remove a loss component from the registry.
        
        Parameters
        ----------
        name : str
            Identifier of the loss function to remove
        """
        if name in self.losses:
            del self.losses[name]
            del self.weights[name]
    
    def compute_total_loss(self, mcr_model, **kwargs):
        """Compute weighted sum of all registered losses.
        
        Parameters
        ----------
        mcr_model : MCR
            The MCR model instance to compute losses for
        **kwargs : dict
            Additional arguments passed to loss functions
            
        Returns
        -------
        tuple
            (total_loss, loss_components) where total_loss is the weighted sum
            and loss_components is a dict of individual loss values
        """
        total_loss = 0.0
        loss_components = {}
        
        for name, loss_fn in self.losses.items():
            component_loss = loss_fn(mcr_model, **kwargs)
            weighted_loss = component_loss * self.weights[name]
            total_loss += weighted_loss
            loss_components[name] = component_loss.item()
            
        return total_loss, loss_components