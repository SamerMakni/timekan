import torch
import torch.nn as nn

# this is inpired from https://github.com/SynodicMonth/ChebyKAN

class Chebyshev(nn.Module):
    """
    A neural network layer that applies a Chebyshev polynomial transformation to the input.

    This layer approximates functions using a Chebyshev series expansion, where the input 
    is first normalized to the range [-1, 1] using the hyperbolic tangent function (`tanh`), 
    then transformed using Chebyshev polynomials of the first kind.
    Args:
        inputdim (int): The number of input features.
        outdim (int): The number of output features.
        degree (int, optional): The degree of the Chebyshev expansion. Default is 3.
    """
    def __init__(self, inputdim, outdim, degree = 3):
        super(Chebyshev, self).__init__()
        self.inputdim = inputdim
        self.outdim = outdim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(inputdim, outdim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (inputdim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        """
        Forward pass of the Chebyshev Kernel Attention Network (KAN) layer.

        The input is first normalized to the range [-1, 1] using tanh. 
        The layer then computes Chebyshev polynomials using the arccosine and cosine functions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, inputdim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, outdim), obtained via 
                          Chebyshev polynomial interpolation.
        """
        x = torch.tanh(x)
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  
        x = x.acos()
        x *= self.arange
        x = x.cos()
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        ) 
        y = y.view(-1, self.outdim)
        return y
