import sbibm
import torch
import numpy as np
from pypolyagamma import PyPolyaGamma
from tqdm import tqdm


def main():
    sbibm.get_available_tasks()
    task = sbibm.get_task("bernoulli_glm")  # See sbibm.get_available_tasks() for all tasks
    
    dim_data = 10

    Binv2 = torch.diag(torch.ones(dim_data)* 0.5)
    Binv2 = Binv2.numpy()
    design_matrix = torch.load("/home/hyun18/NDP/benchmark/design_matrix.pt")
    
    mcmc_num_samples_warmup = 25000
    mcmc_thinning = 25
    num_samples = 10_000
    mcmc_num_samples = mcmc_num_samples_warmup + mcmc_thinning * num_samples

    pg = PyPolyaGamma()
    X = design_matrix.numpy()

    for x0_ind in range(1, 11):
        task.raw = True
        obs = task.get_observation(num_observation=x0_ind).numpy()  # 10 per task

        # Init at zero (or random), since true parameters aren't known
        true_parameters = task.get_true_parameters(num_observation=x0_ind)
        sample = true_parameters.numpy().reshape(-1)  # Init at true parameters
        samples = []
        for j in tqdm(range(mcmc_num_samples)):
            psi = np.dot(X, sample)
            w = np.array([pg.pgdraw(1, b) for b in psi])
            O = np.diag(w)
            V = np.linalg.inv(np.dot(np.dot(X.T, O), X) + Binv2)
            m = np.dot(V, np.dot(X.T, obs.reshape(-1) - 1 * 0.5))
            sample = np.random.multivariate_normal(np.ravel(m), V)
            samples.append(sample)

        samples = np.asarray(samples).astype(np.float32)
        samples_subset = samples[mcmc_num_samples_warmup::mcmc_thinning, :]

        reference_posterior_samples = torch.from_numpy(samples_subset)
        torch.save(reference_posterior_samples, f"/home/hyun18/depot_hyun/NeuralABC_R/bernoulli_glm/post_{x0_ind}.pt")
        
if __name__ == "__main__":
    main()
