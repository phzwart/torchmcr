import torch
import torch.nn as nn


class MCR(nn.Module):
    def __init__(self, weights, spectra):
        """
        Initialize the MCR object with weights and spectra objects.

        Parameters:
            weights (nn.Module): An object representing weights, must inherit from nn.Module.
            spectra (nn.Module): An object representing spectra, must inherit from nn.Module.
        """
        super(MCR, self).__init__()
        self.weights = weights
        self.spectra = spectra

    def forward(self, **kwargs):
        """
        Compute the forward pass for MCR, allowing dynamic injection of arguments.

        Parameters:
            **kwargs: Arbitrary keyword arguments that will be passed to the weights and spectra objects.

        Returns:
            torch.Tensor: The result of the MCR forward calculation, a matrix product of weights and spectra.
        """
        # Extract or set default values from kwargs
        weights_kwargs = kwargs.get('weights_kwargs', {})
        spectra_kwargs = kwargs.get('spectra_kwargs', {})

        # Forward pass through weights and spectra
        weights_result = self.weights(**weights_kwargs)
        spectra_result = self.spectra(**spectra_kwargs)

        # Perform matrix multiplication between weights and spectra
        result = torch.matmul(weights_result, spectra_result)
        return result
