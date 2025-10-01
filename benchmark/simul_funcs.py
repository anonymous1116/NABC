import torch
import math
from torch.distributions import MultivariateNormal

# Dimensionality
def get_bernoulli_prior():
    dim = 10

    # Mean vector (zero mean)
    loc = torch.zeros(dim)

    # Diagonal precision matrix (inverse of covariance matrix)
    precision_diag = 0.5 * torch.ones(dim)
    precision_matrix = torch.diag(precision_diag)

    # Define the prior
    prior = MultivariateNormal(loc=loc, precision_matrix=precision_matrix)
    return prior

def bernoulli_GLM(thetas):
    design_matrix = torch.load("/home/hyun18/NDP/benchmark/design_matrix.pt")
    DM_t = torch.transpose(design_matrix, 0, 1)
    Psi = torch.matmul(thetas, DM_t) 
    Z = 1 / (1 + torch.exp(-Psi))
    Y = (torch.rand(Z.shape) < Z).float()
    return torch.matmul(Y,design_matrix).to("cpu")

def batched_bernoulli_GLM(thetas, batch_size=10_000_000):
    results = []
    for i in range(0, thetas.shape[0], batch_size):
        th = thetas[i:i+batch_size]
        out = bernoulli_GLM(th)
        results.append(out)
    return torch.cat(results, dim=0)



def truncated_normal(shape, mean=0.0, std=1.0, lower=-0.5, upper=0.5):
    """
    Generates samples from a truncated normal distribution in O(1) time using inverse CDF method.

    Returns:
    - Tensor of shape `shape` with samples from the truncated normal distribution.
    """
    # Convert lower and upper bounds to standard normal space
    lower_cdf = 0.5 * (1 + math.erf((lower - mean) / (std * math.sqrt(2))))
    upper_cdf = 0.5 * (1 + math.erf((upper - mean) / (std * math.sqrt(2))))

    # Sample uniformly in the truncated CDF range
    uniform_samples = torch.rand(shape, dtype=torch.float32) * (upper_cdf - lower_cdf) + lower_cdf

    # Apply inverse CDF (probit function) using erfinv
    truncated_samples = mean + std * torch.erfinv(2 * uniform_samples - 1) * math.sqrt(2)

    return truncated_samples

def truncated_mvn_sample(L, mean, std, lower, upper):
    """
    L: size of priors
    mean, std, lower, upper: torch.tensor with size [d]
    """
    d = mean.size(0)
    samples = []
    for j in range(d):
        tmp = truncated_normal((L,), mean[j], std[j], lower[j], upper[j])
        samples.append(tmp)
    return torch.column_stack(samples)