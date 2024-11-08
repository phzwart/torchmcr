import torch
import torch.optim as optim
import torch.nn.functional as F

def train_mcr_model(model, observed_data, num_epochs=1000, mini_epochs=5, lr=0.01, tolerance=1e-16,
                    optimizer_class=None, loss_fn=None, device=None):
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

    Returns:
        None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model and data to the specified device
    model = model.to(device)
    observed_data = observed_data.to(device)

    # Default optimizer and loss function
    if optimizer_class is None:
        optimizer_class = optim.Adam
    if loss_fn is None:
        loss_fn = F.l1_loss  # L1 loss (Mean Absolute Error)

    # Conditionally create optimizers if there are parameters that require gradients
    if any(p.requires_grad for p in model.spectra.parameters()):
        spectra_optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.spectra.parameters()), lr=lr)
    else:
        spectra_optimizer = None

    if any(p.requires_grad for p in model.weights.parameters()):
        weights_optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.weights.parameters()), lr=lr)
    else:
        weights_optimizer = None

    # Use closure for LBFGS
    use_closure = optimizer_class == optim.LBFGS

    # Track loss for early stopping
    prev_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Initialize epoch loss

        # Run mini-epochs for spectra and weights separately
        for mini_epoch in range(mini_epochs):
            # Update spectra while keeping weights fixed
            if spectra_optimizer:
                if use_closure:
                    # Define a closure function for LBFGS and run optimizer
                    def spectra_closure():
                        spectra_optimizer.zero_grad()
                        predicted_data = model()
                        loss = loss_fn(predicted_data, observed_data)
                        loss.backward()
                        return loss

                    loss = spectra_optimizer.step(spectra_closure)  # Update spectra with LBFGS
                else:
                    spectra_optimizer.zero_grad()
                    predicted_data = model()  # Forward pass with current weights and spectra
                    loss = loss_fn(predicted_data, observed_data)
                    loss.backward()  # Backward pass for spectra only
                    spectra_optimizer.step()  # Update spectra
            else:
                # No update for spectra; just compute the loss
                predicted_data = model()
                loss = loss_fn(predicted_data, observed_data)

            # Update weights while keeping spectra fixed
            if weights_optimizer:
                if use_closure:
                    # Define a closure function for LBFGS and run optimizer
                    def weights_closure():
                        weights_optimizer.zero_grad()
                        predicted_data = model()
                        loss = loss_fn(predicted_data, observed_data)
                        loss.backward()
                        return loss

                    loss = weights_optimizer.step(weights_closure)  # Update weights with LBFGS
                else:
                    weights_optimizer.zero_grad()
                    predicted_data = model()  # Forward pass with updated spectra
                    loss = loss_fn(predicted_data, observed_data)
                    loss.backward()  # Backward pass for weights only
                    weights_optimizer.step()  # Update weights

            # Accumulate loss for the epoch (use latest mini-batch loss)
            epoch_loss += loss.item()

        # Compute average epoch loss
        epoch_loss /= mini_epochs

        # Print loss and check for early stopping condition
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        # Early stopping condition if the change in loss is small
        if abs(prev_loss - epoch_loss) < tolerance:
            print("Early stopping: Loss change below tolerance.")
            break

        # Update the previous loss
        prev_loss = epoch_loss

    print("Training complete.")
