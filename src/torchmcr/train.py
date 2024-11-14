import torch
import torch.optim as optim
import torch.nn.functional as F

def train_mcr_model(model, 
                    observed_data, 
                    num_epochs=1000, 
                    mini_epochs=5, 
                    lr=0.01, 
                    tolerance=1e-16,
                    optimizer_class=None, 
                    loss_fn=None, 
                    device='cpu', 
                    show_every=10):
    """
    Train the MCR model with alternating updates for spectra and weights, using custom optimizer and loss function.

    Parameters:
        model (nn.Module): The MCR model to be trained.
        observed_data (torch.Tensor): The target data to fit the model against.
        num_epochs (int): The number of epochs for training.
        mini_epochs (int): The number of mini-epochs to alternate between spectra and weights updates.
        lr (float): Learning rate for the optimizer.
        tolerance (float): Tolerance for early stopping based on loss change.
        optimizer_class (type, optional): Optimizer class to use, e.g., torch.optim.SGD or torch.optim.Adam.
                                         If None, defaults to Adam.
        loss_fn (callable, optional): Loss function to use. If None, defaults to L1 loss (Mean Absolute Error).
        device (torch.device, optional): Device to run the model on (CPU or GPU).
        show_every (int): Frequency of loss printing.

    Returns:
        None
    """
    if device is None:
        device = observed_data.device

    # Move model and data to the specified device
    model = model.to(device)
    observed_data = observed_data.to(device)

    # Default optimizer and loss function
    if optimizer_class is None:
        optimizer_class = optim.Adam
    if loss_fn is None:
        loss_fn = F.l1_loss  # L1 loss (Mean Absolute Error)

    # Conditionally create optimizers if there are parameters that require gradients
    # Check if any parameters in spectra require gradients
    spectra_requires_grad = model.spectra.requires_grad if hasattr(model.spectra, 'requires_grad') else any(p.requires_grad for p in model.spectra.parameters())
    if spectra_requires_grad:
        spectra_optimizer = optimizer_class(model.spectra.parameters(), lr=lr)
    else:
        spectra_optimizer = None  # TODO: Check if this is correct, we use gradient weight matrix to allow for mixed / targeted updates

    # Check if any parameters in weights require gradients
    weights_requires_grad = model.weights.requires_grad if hasattr(model.weights, 'requires_grad') else any(p.requires_grad for p in model.weights.parameters())
    if weights_requires_grad:
        weights_optimizer = optimizer_class(model.weights.parameters(), lr=lr)
    else:
        weights_optimizer = None  # TODO: Check if this is correct, we use gradient weight matrix to allow for mixed / targeted updates

    # Use closure for LBFGS and set retain_graph=True for multiple backward passes
    use_closure = optimizer_class == optim.LBFGS


    # Track loss for early stopping
    prev_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Initialize epoch loss

        # Run mini-epochs for spectra and weights separately
        if spectra_optimizer is not None:
            spectra_optimizer.zero_grad()
        if weights_optimizer is not None:
            weights_optimizer.zero_grad()

        if spectra_optimizer is not None:
            for mini_epoch in range(mini_epochs):
                if use_closure:
                    # Define a closure function for LBFGS and run optimizer
                    def spectra_closure():
                        spectra_optimizer.zero_grad()
                        predicted_data = model()
                        loss = loss_fn(predicted_data, observed_data)
                        loss.backward(retain_graph=True)
                        return loss
                    loss = spectra_optimizer.step(spectra_closure)  # Update spectra with LBFGS
                else:
                    spectra_optimizer.zero_grad()
                    predicted_data = model()  # Forward pass with current weights and spectra
                    loss = loss_fn(predicted_data, observed_data)
                    loss.backward(retain_graph=True)  # Backward pass for spectra only
                    spectra_optimizer.step()  # Update spectra
        else:
            # No update for spectra; just compute the loss
            predicted_data = model()
            loss = loss_fn(predicted_data, observed_data)

        # Update weights while keeping spectra fixed
        if weights_optimizer is not None:
            for mini_epoch in range(mini_epochs):
                if use_closure:
                    # Define a closure function for LBFGS and run optimizer
                    def weights_closure():
                        weights_optimizer.zero_grad()
                        predicted_data = model()
                        loss = loss_fn(predicted_data, observed_data)
                        loss.backward(retain_graph=True)
                        return loss
                    loss = weights_optimizer.step(weights_closure)  # Update weights with LBFGS
                else:
                    weights_optimizer.zero_grad()
                    predicted_data = model()  # Forward pass with updated spectra
                    loss = loss_fn(predicted_data, observed_data)
                    loss.backward(retain_graph=True)  # Backward pass for weights only
                    weights_optimizer.step()  # Update weights

        epoch_loss += loss.item()

        # Compute average epoch loss
        epoch_loss /= mini_epochs

        # Print loss and check for early stopping condition
        if epoch % show_every == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        # Early stopping condition if the change in loss is small
        if abs(prev_loss - epoch_loss) < tolerance:
            print("Early stopping: Loss change below tolerance.")
            break

        # Update the previous loss
        prev_loss = epoch_loss

    assert "Training complete."