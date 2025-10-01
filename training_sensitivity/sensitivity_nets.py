import torch
import torch.nn as nn

class FL_Net_sen(nn.Module):
    def __init__(self, D_in, D_out, H, p, num=3):
        """
        Parameters
        ----------
        D_in : int
            Input dimension
        D_out : int
            Output dimension
        H : int
            Hidden layer width
        p : float
            Dropout probability
        bounds : list of tuples
            Each (a, b) defines bounds for one output dimension
        num : int
            Number of hidden layers
        """
        super(FL_Net_sen, self).__init__()
        
        layers = []
        in_dim = D_in

        # Add `num` hidden layers
        for _ in range(num):
            layers.append(nn.Linear(in_dim, H))
            layers.append(nn.BatchNorm1d(H))
            layers.append(nn.Dropout(p))
            layers.append(nn.ReLU())
            in_dim = H

        # Final output layer
        layers.append(nn.Linear(H, D_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        output = self.mlp(x)
        return output



class FL_Net_sen2(nn.Module):
    def __init__(self, D_in, D_out, H, p, num=3):
        """
        Parameters
        ----------
        D_in : int
            Input dimension
        D_out : int
            Output dimension
        H : int
            Hidden layer width
        p : float
            Dropout probability
        bounds : list of tuples
            Each (a, b) defines bounds for one output dimension
        num : int
            Number of hidden layers
        """
        super(FL_Net_sen2, self).__init__()
        
        layers = []
        in_dim = D_in

        # Add `num` hidden layers
        for _ in range(num):
            layers.append(nn.Linear(in_dim, H))
            layers.append(nn.BatchNorm1d(H))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p))
            in_dim = H

        # Final output layer
        layers.append(nn.Linear(H, D_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        output = self.mlp(x)
        return output


class FL_Net_bounded_sen(nn.Module):
    def __init__(self, D_in, D_out, H, p, bounds, num=3):
        """
        Parameters
        ----------
        D_in : int
            Input dimension
        D_out : int
            Output dimension
        H : int
            Hidden layer width
        p : float
            Dropout probability
        bounds : list of tuples
            Each (a, b) defines bounds for one output dimension
        num : int
            Number of hidden layers
        """
        super(FL_Net_bounded_sen, self).__init__()
        self.bounds = bounds

        layers = []
        in_dim = D_in

        # Add `num` hidden layers
        for _ in range(num):
            layers.append(nn.Linear(in_dim, H))
            layers.append(nn.BatchNorm1d(H))
            layers.append(nn.Dropout(p))
            layers.append(nn.ReLU())
            in_dim = H

        # Final output layer
        layers.append(nn.Linear(H, D_out))

        self.mlp = nn.Sequential(*layers)

    def transform_output(self, raw_output):
        # Bound each output using sigmoid scaling to (a, b)
        scaled = []
        for i, (a, b) in enumerate(self.bounds):
            scaled.append(a + (b - a) * torch.sigmoid(raw_output[:, i]))
        return torch.stack(scaled, dim=1)

    def forward(self, x):
        output = self.mlp(x)
        return self.transform_output(output)
