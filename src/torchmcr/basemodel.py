import torch
import torch.nn as nn

#TODO: add a method to freeze and unfreeze weights and spectra by setting requires_grad to False or True

class MCR(nn.Module):
    def __init__(self, weights, spectra):
        """
        Initialize the MCR object with weights and spectra objects.

        Parameters:
            weights (nn.Module): An object representing weights, must inherit from nn.Module.
            spectra (nn.Module): An object representing spectra, must inherit from nn.Module.

        Details:
        weights: a N_observations x N_components matrix
        spectra: a N_components x N_wavelengths matrix
        output spectra from self.forward(): a N_observations x N_wavelengths matrix
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

    def freeze_weights(self, row_indices=None, col_indices=None, coords=None):
        """
        Freeze specific weight parameters by setting requires_grad to False.

        Parameters:
            row_indices (list, optional): List of row indices to freeze entirely.
            col_indices (list, optional): List of column indices to freeze entirely.
            coords (list of tuples, optional): List of (row, col) coordinates to freeze specific values.
            If no parameters are provided, freeze all weights.
        """
        for param in self.weights.parameters():
            param.requires_grad = False

    def unfreeze_weights(self, row_indices=None, col_indices=None, coords=None):
        """
        Unfreeze specific weight parameters by setting requires_grad to True.

        Parameters:
            row_indices (list, optional): List of row indices to unfreeze entirely.
            col_indices (list, optional): List of column indices to unfreeze entirely.
            coords (list of tuples, optional): List of (row, col) coordinates to unfreeze specific values.
            If no parameters are provided, unfreeze all weights.
        """
        for param in self.weights.parameters():
            param.requires_grad = True

    def freeze_spectra(self, row_indices=None, col_indices=None, coords=None):
        """
        Freeze specific spectra parameters by setting requires_grad to False.

        Parameters:
            row_indices (list, optional): List of row indices to freeze entirely.
            col_indices (list, optional): List of column indices to freeze entirely.
            coords (list of tuples, optional): List of (row, col) coordinates to freeze specific values.
            If no parameters are provided, freeze all spectra.
        """
        for param in self.spectra.parameters():
            param.requires_grad = False

    def unfreeze_spectra(self, row_indices=None, col_indices=None, coords=None):
        """
        Unfreeze specific spectra parameters by setting requires_grad to True.

        Parameters:
            row_indices (list, optional): List of row indices to unfreeze entirely.
            col_indices (list, optional): List of column indices to unfreeze entirely.
            coords (list of tuples, optional): List of (row, col) coordinates to unfreeze specific values.
            If no parameters are provided, unfreeze all spectra.
        """
        for param in self.spectra.parameters():
            param.requires_grad = True