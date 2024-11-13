import torch
import torch.nn as nn

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

        # Initialize gradient masks with all ones (meaning everything is trainable)
        # Get initial shapes from a forward pass
        with torch.no_grad():
            weights_shape = self.weights().shape
            spectra_shape = self.spectra().shape
        
        self.weights_grad_mask = torch.ones(weights_shape, dtype=torch.float32)
        self.spectra_grad_mask = torch.ones(spectra_shape, dtype=torch.float32)

        # Register hooks on all parameters
        for param in self.weights.parameters():
            param.register_hook(self._apply_weights_grad_mask)
        for param in self.spectra.parameters():
            param.register_hook(self._apply_spectra_grad_mask)

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

    def _apply_weights_grad_mask(self, grad):
        """
        Apply the weights gradient mask to the computed gradient during backpropagation.
        """
        return grad * self.weights_grad_mask

    def _apply_spectra_grad_mask(self, grad):
        """
        Apply the spectra gradient mask to the computed gradient during backpropagation.
        """
        return grad * self.spectra_grad_mask

    def freeze_weights(self, row_indices=None, col_indices=None, coords=None):
        """
        Freeze specific weight parameters by updating the gradient mask.

        Parameters:
            row_indices (list, optional): List of row indices to freeze entirely.
            col_indices (list, optional): List of column indices to freeze entirely.
            coords (list of tuples, optional): List of (row, col) coordinates to freeze specific values.
            If no parameters are provided, freeze all weights.
        """
        if row_indices is None and col_indices is None and coords is None:
            # Freeze all weights if no arguments are provided
            self.weights_grad_mask[:] = 0
        else:
            if row_indices is not None:
                self.weights_grad_mask[row_indices, :] = 0
            
            if col_indices is not None:
                self.weights_grad_mask[:, col_indices] = 0

            if coords is not None:
                rows, cols = zip(*coords)
                self.weights_grad_mask[rows, cols] = 0

    def unfreeze_weights(self, row_indices=None, col_indices=None, coords=None):
        """
        Unfreeze specific weight parameters by updating the gradient mask.

        Parameters:
            row_indices (list, optional): List of row indices to unfreeze entirely.
            col_indices (list, optional): List of column indices to unfreeze entirely.
            coords (list of tuples, optional): List of (row, col) coordinates to unfreeze specific values.
            If no parameters are provided, unfreeze all weights.
        """
        if row_indices is None and col_indices is None and coords is None:
            # Unfreeze all weights if no arguments are provided
            self.weights_grad_mask[:] = 1
        else:
            if row_indices is not None:
                self.weights_grad_mask[row_indices, :] = 1
            
            if col_indices is not None:
                self.weights_grad_mask[:, col_indices] = 1

            if coords is not None:
                rows, cols = zip(*coords)
                self.weights_grad_mask[rows, cols] = 1

    def freeze_spectra(self, row_indices=None, col_indices=None, coords=None):
        """
        Freeze specific spectra parameters by updating the gradient mask.

        Parameters:
            row_indices (list, optional): List of row indices to freeze entirely.
            col_indices (list, optional): List of column indices to freeze entirely.
            coords (list of tuples, optional): List of (row, col) coordinates to freeze specific values.
            If no parameters are provided, freeze all spectra.
        """
        if row_indices is None and col_indices is None and coords is None:
            # Freeze all spectra if no arguments are provided
            self.spectra_grad_mask[:] = 0
        else:
            if row_indices is not None:
                self.spectra_grad_mask[row_indices, :] = 0
            
            if col_indices is not None:
                self.spectra_grad_mask[:, col_indices] = 0

            if coords is not None:
                rows, cols = zip(*coords)
                self.spectra_grad_mask[rows, cols] = 0

    def unfreeze_spectra(self, row_indices=None, col_indices=None, coords=None):
        """
        Unfreeze specific spectra parameters by updating the gradient mask.

        Parameters:
            row_indices (list, optional): List of row indices to unfreeze entirely.
            col_indices (list, optional): List of column indices to unfreeze entirely.
            coords (list of tuples, optional): List of (row, col) coordinates to unfreeze specific values.
            If no parameters are provided, unfreeze all spectra.
        """
        if row_indices is None and col_indices is None and coords is None:
            # Unfreeze all spectra if no arguments are provided
            self.spectra_grad_mask[:] = 1
        else:
            if row_indices is not None:
                self.spectra_grad_mask[row_indices, :] = 1
            
            if col_indices is not None:
                self.spectra_grad_mask[:, col_indices] = 1

            if coords is not None:
                rows, cols = zip(*coords)
                self.spectra_grad_mask[rows, cols] = 1
