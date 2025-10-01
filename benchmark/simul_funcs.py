import torch
import math
import numpy as np
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



class UnifSample:
    def __init__(self, bins = 10):
        self.bins = bins
        
    def box(self, sample ,num):
        heights, intervals = np.histogram(sample, self.bins, density = True)
        a, b = UnifSample.support_index(heights)
        samples = UnifSample.support_sample(intervals, a, b, num)
        return torch.tensor(samples, dtype = torch.float32)
    
    @staticmethod
    def support_index(heights):
        temp = ( (heights /np.sum(heights) ) != 0.0)
        return intervals_connect(heights,temp)
        
    @staticmethod
    def support_sample(intervals, a, b, num):
        interval_diffs = intervals[b] - intervals[a]
        prop = interval_diffs / np.sum(interval_diffs)
        size_num = np.random.multinomial(num, prop)
        
        # Preallocate an array for the results
        ran = np.empty(num)
        
        cum_sum = 0
        for i, size in enumerate(size_num):
            if size > 0:
                # Generate samples in the current interval
                tmp = np.random.uniform(0, 1, size) * interval_diffs[i] + intervals[a][i]
                ran[cum_sum:cum_sum + size] = tmp
                cum_sum += size
        
        np.random.shuffle(ran)
        return ran[:num]

def param_box(unifsam, sample, num):
    """
    unifsam: UnifSample object with determined seeds
    sample : n*p tensor
    num : the number of samples
    """
    theta_new = []
    for j in range(sample.size()[1]):
        sam = sample[:,j]
        theta_new.append(torch.reshape(unifsam.box(sam, num), (num, 1)))
        del sam
    return torch.cat(theta_new, 1)

def intervals_connect(heights, indices):
    a = list()
    b = list()
    for i in range(len(heights)):
        if indices[i] == True:
            if i == 0:
                a.append(i)
            elif indices[i-1] == False:
                a.append(i)
            if i == len(heights)-1:
                b.append(i+1)
            elif indices[i+1] == False:
                b.append(i+1)
    return [a,b]
