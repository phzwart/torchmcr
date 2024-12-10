import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianMCRLoss(nn.Module):
    def __init__(self, alpha, initial_sigma=1.0):
        super(BayesianMCRLoss, self).__init__()
        # Alpha parameter for Dirichlet prior
        self.alpha = torch.tensor(alpha, requires_grad=False)
        # Variance (sigma^2) parameter - learnable
        self.log_sigma = torch.nn.Parameter(torch.log(torch.tensor(initial_sigma, dtype=torch.float32)))

    def forward(self, predicted, observed, weights):
        # Ensure weights are normalized (i.e., sum to 1)
        weights = F.softmax(weights, dim=-1)

        # Likelihood loss (fit term)
        sigma = torch.exp(self.log_sigma)
        fit_loss = torch.sum((observed - predicted) ** 2) / (2 * sigma ** 2) + 0.5 * torch.log(sigma ** 2)

        # Dirichlet prior (regularization term)
        prior_loss = - torch.sum((self.alpha - 1) * torch.log(weights))

        # Total loss
        total_loss = fit_loss + prior_loss

        return total_loss
