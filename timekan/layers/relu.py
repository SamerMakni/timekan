import numpy as np
import torch
import torch.nn as nn


class ReLU(nn.Module):
    """
    A ReLU-based Kolmogorov-Arnold Network (KAN) layer for nonlinear transformations.

    This layer uses trainable ReLU activations and a convolutional operation to transform
    input features into a higher-dimensional output, suitable for enhancing recurrent models.

    Args:
        inputdim (int): Number of input features.
        outdim (int): Number of output features.
        train_ab (bool, optional): If True, ReLU thresholds are trainable. Defaults to True.
        g (int, optional): Grid size parameter controlling the number of basis points. Defaults to 5.
        k (int, optional): Parameter controlling the range of ReLU thresholds. Defaults to 3.
    """
    def __init__(self, inputdim: int, outdim: int, train_ab: bool = True, g=5, k=3):
        super().__init__()
        self.g, self.k, self.r = g, k, 4 * g * g / ((k + 1) * (k + 1))
        self.inputdim, self.outdim = inputdim, outdim
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k + 1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for _ in range(inputdim)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for _ in range(inputdim)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, outdim, (g + k, inputdim))

    def forward(self, x):
        """Transforms input through ReLU-based KAN operations and convolution.

        Args:
            x (torch.Tensor): Input tensor, either [batch_size, inputdim] or [batch_size, seq_len, inputdim].

        Returns:
            torch.Tensor: Output tensor, either [batch_size, outdim] or [batch_size, seq_len, outdim].
        """
        if x.dim() == 3:  # [batch_size, seq_len, inputdim]
            batch_size, seq_len, _ = x.shape
            x = x.view(batch_size * seq_len, self.inputdim)  # Flatten to [batch_size * seq_len, inputdim]
        else:  # [batch_size, inputdim]
            batch_size = x.size(0)
            seq_len = 1

        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_height - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape(batch_size * seq_len, 1, self.g + self.k, self.inputdim)
        x = self.equal_size_conv(x)
        x = x.squeeze(-1)  # [batch_size * seq_len, outdim]

        # Reshape back to sequence format if necessary
        if seq_len > 1:
            x = x.view(batch_size, seq_len, self.outdim)
        return x


class ReLUKAN(nn.Module):
    def __init__(self, width, grid, k):
        super().__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLU(width[i], grid, k, width[i+1]))
            # if len(width) - i > 2:
            #     self.rk_layers.append()
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        # x = x.reshape((len(x), self.width[-1]))
        return x
