import torch
import torch.nn as nn

import numpy as np

# this is inpired from https://github.com/GistNoesis/FourierKAN/
class Fourier(nn.Module):
    """
    A neural network layer that approximates functions using a Fourier series expansion.

    This layer transforms input features into a high-dimensional Fourier space using
    sine and cosine functions and learns Fourier coefficients to approximate functions.
    Args:
        inputdim (int): Number of input features.
        outdim (int): Number of output features.
        gridsize (int, optional): Number of Fourier basis functions. Default is 300.
        addbias (bool, optional): Whether to include a bias term. Default is True.
    """
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True):
        super(Fourier,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self,x):
        """
        Forward pass of the Naive Fourier Kernel Attention Network (KAN) layer.

        The input is expanded using a Fourier series approximation by computing sine and 
        cosine transformations, then applying learned Fourier coefficients to approximate 
        functions.

        Args:
            x (torch.Tensor): Input tensor of shape (..., inputdim), where `...` represents 
                              arbitrary batch dimensions.

        Returns:
            torch.Tensor: Output tensor of shape (..., outdim), representing the transformed 
                          Fourier features.
        """
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        
        y = y.view(outshape)
        return y
        