import torch

def inverse_softplus(y):
    """
    Computes the inverse of the softplus function.
    
    The softplus function is defined as: f(x) = ln(1 + exp(x))
    This computes its inverse: f^(-1)(y) = ln(exp(y) - 1)
    
    For numerical stability with small values, returns y when y <= 1e-4
    since softplus(x) ≈ x for small x.
    
    Args:
        y (torch.Tensor or array-like): Input values to compute inverse softplus
            
    Returns:
        torch.Tensor: The inverse softplus of the input
    """
    # Ensure input is a tensor
    y = torch.as_tensor(y)
    
    # Inverse softplus: ln(exp(y) - 1)
    # Use torch.where for numerical stability with small values
    return torch.where(y > 1e-4, torch.log(torch.exp(y) - 1), y)


def normalized_softmax(x, scale_factor=None, dim=-1):
    """
    Applies softmax but scales the output to have a target sum.
    If scale_factor is None, uses length of input as scale factor.
    
    Args:
        x: Input tensor
        scale_factor: Target sum for the output. If None, uses len(x)
        dim: Dimension along which to apply softmax
    
    Returns:
        Scaled softmax output tensor with desired sum
    """
    if scale_factor is None:
        scale_factor = x.shape[dim]
    
    # Regular softmax first
    softmaxed = torch.softmax(x, dim=dim)
    
    # Since softmax sums to 1, multiply by scale_factor to get desired sum
    return softmaxed * scale_factor

def inverse_normalized_softmax(y, scale_factor=None, dim=-1):
    """
    Inverse of normalized_softmax function. First un-scales the input,
    then computes the inverse softmax (logits).
    
    Args:
        y: Input tensor that was output from normalized_softmax
        scale_factor: The scale factor used in normalized_softmax. 
                     If None, uses length of input
    
    Returns:
        Approximate logits that would produce the input when passed through normalized_softmax
    """
    if scale_factor is None:
        scale_factor = y.shape[dim]
        
    # First un-scale
    y_unscaled = y / scale_factor
    
    # Inverse softmax (log of probabilities)
    # Add small epsilon to avoid log(0)
    return torch.log(y_unscaled + 1e-10)