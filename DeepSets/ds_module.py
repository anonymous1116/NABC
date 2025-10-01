import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSets2D(nn.Module):
    def __init__(self, input_dim, hidden, out_dim,):
        super().__init__()
        self.phi = nn.Sequential(          # per-row map: R^2 -> R^h
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.rho = nn.Sequential(          # pooled map: R^h -> output
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):                  # x: (B,4,2)
        if x.ndim == 2:
            x = x.reshape(-1,4,2)
        
        h = self.phi(x)                    # (B,4,h)
        h_pool = h.mean(dim=1)             # (B,h)  (sum/mean = invariant)
        return self.rho(h_pool)

    
class DeepSets2D_bounded(nn.Module):
    def __init__(self, input_dim, hidden, out_dim, bounds):
        super(DeepSets2D_bounded, self).__init__()
        self.bounds = bounds
        self.phi = nn.Sequential(          # per-row map: R^2 -> R^h
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.rho = nn.Sequential(          # pooled map: R^h -> output
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        

    def transform_output(self, raw_output):
        # Bound each output using sigmoid scaling to (a, b)
        scaled = []
        for i, (a, b) in enumerate(self.bounds):
            scaled.append(a + (b - a) * torch.sigmoid(raw_output[:, i]))
        return torch.stack(scaled, dim=1)
    
    def forward(self, x):
        if x.ndim == 2:
            x = x.reshape(-1,4,2)
        h = self.phi(x)                    # (B,4,h)
        h_pool = h.mean(dim=1)             # (B,h)  (sum/mean = invariant)
        return self.transform_output(self.rho(h_pool))
    


class DeepSets_CovarianceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,  output_dim=5, jitter=1e-6):
        super().__init__()
        self.d = output_dim
        self.jitter = jitter
        self.phi = nn.Sequential(          # per-row map: R^2 -> R^h
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        
        tril_size = output_dim * (output_dim + 1) // 2
        
        self.fc1  = nn.Linear(hidden_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, tril_size)


        # Precompute indices once (no forward-time allocations)
        tril_idx = torch.tril_indices(output_dim, output_dim, offset=0)
        diag_idx = torch.arange(output_dim, dtype=torch.long)
        self.register_buffer("tril_idx", tril_idx, persistent=False)
        self.register_buffer("diag_idx", diag_idx, persistent=False)

    def forward(self, x, return_sigma=False):
        # Allow [B,T] as shorthand for [B,T,1]
        if x.ndim == 2:
            x = x.reshape(-1,4,2)
        
        
        h = self.phi(x)                    # (B,4,h)
        h_pool = h.mean(dim=1)             # (B,h)  (sum/mean = invariant)
        
        chol_vec = self.fc2(self.fc1(h_pool))             # [B, tril_size]


        # Fill lower triangle
        B = chol_vec.size(0)
        L = chol_vec.new_zeros(B, self.d, self.d)
        L[:, self.tril_idx[0], self.tril_idx[1]] = chol_vec

        # Stable positive diagonal: softplus + tiny epsilon
        diag = L[:, self.diag_idx, self.diag_idx]
        L[:, self.diag_idx, self.diag_idx] = F.softplus(diag) + self.jitter

        if return_sigma:
            Sigma = L @ L.transpose(1, 2)
            # Add jitter for extra safety
            Sigma = Sigma + self.jitter * torch.eye(self.d, device=Sigma.device).unsqueeze(0)
            return Sigma
        return L
    