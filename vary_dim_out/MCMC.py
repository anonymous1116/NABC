import torch
import numpy as np
from torch.distributions.normal import Normal
from sbibm.metrics.c2st import c2st
import copy
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import argparse

def metropolis_hastings(log_prob_fn, initial_theta, num_samples, proposal_std=0.05, burn_in=0.1):
    theta = initial_theta.clone()
    samples = []
    log_p = log_prob_fn(theta)

    total_iters = int(num_samples / (1 - burn_in))

    for _ in range(total_iters):
        # Propose from Gaussian
        theta_prop = theta + proposal_std * torch.randn_like(theta)

        # Reject if out of prior support [-5, 5]""""""
        if torch.any(theta_prop < -5) or torch.any(theta_prop > 5):
            samples.append(theta.clone())  # don't move
            continue

        log_p_prop = log_prob_fn(theta_prop)
        accept_ratio = torch.exp(log_p_prop - log_p)

        if torch.rand(1).item() < accept_ratio.item():
            theta = theta_prop
            log_p = log_p_prop

        samples.append(theta.clone())

    samples = torch.stack(samples)
    burn = int(burn_in * len(samples))
    return samples[burn:]

def log_posterior_closed_form(mu, s_dp, sigma0, b):
    """
    Exact log p(s_dp | mu) using convolution of N(μ, σ₀²) and Laplace(0, b).

    Args:
        mu: tensor (scalar or batch)
        s_dp: scalar or tensor, same shape as mu or broadcastable
        sigma0: float
        b: float

    Returns:
        log probability: tensor same shape as mu
    """
    mu = mu.clone().detach().float()
    s_dp = torch.tensor(s_dp, dtype=torch.float32)
    sigma = torch.tensor(sigma0, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    z = s_dp - mu
    norm = Normal(0.0, 1.0)

    # Exponential prefactor
    exp_term = torch.exp(sigma**2 / (2 * b**2))
    
    u = z / sigma
    t = sigma / b

    # Two terms
    term1 = torch.exp(-z / b) * norm.cdf(u - t)
    term2 = torch.exp(z / b) * (1 - norm.cdf(u + t))

    # Final likelihood
    result = (1 / (2 * b)) * exp_term * (term1 + term2)

    # Log-likelihood with numerical stability
    log_result = torch.log(result + 1e-12)

    return log_result


def main(args):
    b = 0.2
    sigma0 = 0.2

    s_dp_list = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5]
    s_dp = s_dp_list[args.s_dp_ind]
    torch.manual_seed(args.s_dp_ind)
    np.random.seed(args.s_dp_ind)

    # Define scalar-valued log-probability for MCMC
    log_prob = lambda mu: log_posterior_closed_form(mu, s_dp, sigma0, b)

    # Initial guess for mu
    initial_mu = torch.tensor([0.0])

    samples = metropolis_hastings(log_prob, initial_mu, num_samples=500000, proposal_std=1, burn_in = 0.5)
    
    post_nums = 10_000
    sam_ind_post = np.random.choice(np.arange(0, samples.size(0)), post_nums, replace = False)
    post_samples_tensor = torch.tensor(samples[sam_ind_post], dtype = torch.float32)
    torch.save(post_samples_tensor, f"../depot_hyun/NeuralABC_R/NDP/posterior_samples_sdp_{args.s_dp_ind}.pt")


def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--s_dp_ind", type = int, default = 1,
                        help = "See number (default: 1)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    #main_cond(args)
    
    # Use the parsed arguments
    print(f"s_dp_ind: {args.s_dp_ind}")




