import torch
import torch.nn as nn
import torch.nn.functional as F

class FL_Net(nn.Module):
    def __init__(self, D_in, D_out,H = 128, H2 = 128, H3 = 128, p1=0.1, p2=0.1, p3=0.1, device="cuda"):
        super().__init__()
        self.device = device
        
        self.fc1 = nn.Linear(D_in, H)
        self.bn1 = nn.BatchNorm1d(num_features=H)
        self.dn1 = nn.Dropout(p1)

        self.fc2 = nn.Linear(H, H2)
        self.bn2 = nn.BatchNorm1d(num_features=H2)
        self.dn2 = nn.Dropout(p2)

        self.fc3 = nn.Linear(H2, H3)
        self.bn3 = nn.BatchNorm1d(num_features=H3)
        self.dn3 = nn.Dropout(p3)

        self.fc4 = nn.Linear(H3, D_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
    
class FL_Net_bounded(nn.Module):
    def __init__(self, D_in, D_out, H, p, bounds):
        super(FL_Net_bounded, self).__init__()
        self.bounds = bounds
        self.mlp = nn.Sequential(
            nn.Linear(D_in, H),
            nn.BatchNorm1d(H),
            nn.Dropout(p),
            nn.ReLU(),

            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.Dropout(p),
            nn.ReLU(),

            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.Dropout(p),
        
            nn.Linear(H, D_out)
        )

    def transform_output(self, raw_output):
        # Bound each output using sigmoid scaling to (a, b)
        scaled = []
        for i, (a, b) in enumerate(self.bounds):
            scaled.append(a + (b - a) * torch.sigmoid(raw_output[:, i]))
        return torch.stack(scaled, dim=1)
    
    def forward(self, x):
        output = self.mlp(x)
        return self.transform_output(output)
    

class CovarianceNet(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, p = 0.1):
        super().__init__()
        self.d = output_dim
        tril_size = output_dim * (output_dim + 1) // 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, tril_size)
        )

    def forward(self, x):
        chol_vec = self.net(x)
        L = self.fill_lower_triangle(chol_vec, self.d)

        # Enforce positive diagonals
        diag_idx = torch.arange(self.d, device=x.device)
        L[:, diag_idx, diag_idx] = torch.exp(L[:, diag_idx, diag_idx])
        return L

    @staticmethod
    def fill_lower_triangle(vec, d):
        batch_size = vec.size(0)
        L = torch.zeros((batch_size, d, d), device=vec.device)
        tril_indices = torch.tril_indices(row=d, col=d, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = vec
        return L
    


class CovarianceNet2(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=128, p = 0.1):
        super().__init__()
        self.d = output_dim
        tril_size = output_dim * (output_dim + 1) // 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Dropout(p),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, tril_size)
        )

    def forward(self, x):
        chol_vec = self.net(x)
        L = self.fill_lower_triangle(chol_vec, self.d)

        # Enforce positive diagonals
        diag_idx = torch.arange(self.d, device=x.device)
        L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-6
        return L

    @staticmethod
    def fill_lower_triangle(vec, d):
        batch_size = vec.size(0)
        L = torch.zeros((batch_size, d, d), device=vec.device)
        tril_indices = torch.tril_indices(row=d, col=d, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = vec
        return L
    


class GRU_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        GRU-based model for reducing a sequence to a fixed-size continuous output.
        Args:
            input_dim: Dimension of the input features at each time step.
            hidden_dim: Dimension of the GRU hidden state.
            output_dim: Dimension of the output (e.g., 3 for continuous variables).
        """
        super(GRU_net, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Map hidden state to final output78

    def forward(self, x):
        """
        Forward pass for the GRU model.
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
        Returns:
            Tensor of shape [batch_size, output_dim].
        """
        # Pass the sequence through the GRU
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # Now: (batch_size, 2000, 1)
        
        _, hidden_state = self.gru(x)  # hidden_state: [1, batch_size, hidden_dim]
        
        # Use the last hidden state
        hidden_state = hidden_state.squeeze(0)  # [batch_size, hidden_dim]
        
        # Map to final output
        output = self.fc(hidden_state)  # [batch_size, output_dim]
        return output
    


class GRU_net_bounded(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bounds):
        """
        GRU-based model for reducing a sequence to a fixed-size continuous output.
        Args:
            input_dim: Dimension of the input features at each time step.
            hidden_dim: Dimension of the GRU hidden state.
            output_dim: Dimension of the output (e.g., 3 for continuous variables).
        """
        super(GRU_net_bounded, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Map hidden state to final output
        self.bounds = bounds  # List of (a, b) tuples for each dimension

    def transform_output(self, raw_output):
        # Bound each output using sigmoid scaling to (a, b)
        scaled = []
        for i, (a, b) in enumerate(self.bounds):
            scaled.append(a + (b - a) * torch.sigmoid(raw_output[:, i]))
        return torch.stack(scaled, dim=1)
    
    def forward(self, x):
        """
        Forward pass for the GRU model.
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
        Returns:
            Tensor of shape [batch_size, output_dim].
        """
        # Pass the sequence through the GRU
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # Now: (batch_size, 250, 1)
        
        _, hidden_state = self.gru(x)  # hidden_state: [1, batch_size, hidden_dim]
        
        # Use the last hidden state
        hidden_state = hidden_state.squeeze(0)  # [batch_size, hidden_dim]
        
        # Map to final output
        output = self.fc(hidden_state)  # [batch_size, output_dim]
        return self.transform_output(output)
    

class GRU_CovarianceNet(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=64):
        super(GRU_CovarianceNet, self).__init__()
        self.d = output_dim
        tril_size = output_dim * (output_dim + 1) // 2
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tril_size)  # Map hidden state to lower-tri vector

    def forward(self, x, return_sigma=False):
        # Pass the sequence through the GRU
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # Now: (batch_size, 250, 1)
        
        _, hidden_state = self.gru(x)  # hidden_state: [1, batch_size, hidden_dim]
        
        # Use the last hidden state
        hidden_state = hidden_state.squeeze(0)  # [batch_size, hidden_dim]
    
        chol_vec = self.fc(hidden_state)        # [B, tril_size]

        L = self.fill_lower_triangle(chol_vec, self.d)  # [B, d, d]

        # Enforce positive diagonals
        diag_idx = torch.arange(self.d, device=x.device)
        L[:, diag_idx, diag_idx] = torch.exp(L[:, diag_idx, diag_idx])

        if return_sigma:
            Sigma = L @ L.transpose(1, 2)
            return Sigma
        return L

    @staticmethod
    def fill_lower_triangle(vec, d):
        B = vec.size(0)
        L = vec.new_zeros(B, d, d)  # preserves dtype & device
        tril_idx = torch.tril_indices(d, d, offset=0, device=vec.device)
        L[:, tril_idx[0], tril_idx[1]] = vec
        return L

class GRU_CovarianceNet2(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=64, jitter=1e-6):
        super().__init__()
        self.d = output_dim
        self.jitter = jitter

        tril_size = output_dim * (output_dim + 1) // 2
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
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
            # If you ever pass [B,T] but input_dim != 1, this is wrong; assert:
            # assert self.gru.input_size == 1, "2D input implies input_dim=1"
            x = x.unsqueeze(-1)

        # (Perf) helps cuDNN pick fast kernels
        self.gru.flatten_parameters()

        _, h_n = self.gru(x)              # h_n: [1, B, hidden_dim]
        h = h_n[-1]                       # [B, hidden_dim]

        chol_vec = self.fc2(self.fc1(h))             # [B, tril_size]


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
