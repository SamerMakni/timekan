import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import Spline, Chebyshev, Fourier

class TKANGRUCell(nn.Module):
    """
    A GRU cell enhanced with Kolmogorov-Arnold Network (KAN) layers.

    This class implements a single time step computation for a GRU cell where
    KAN layers replace traditional candidate hidden state transformations.

    Args:
        input_dim (int): Size of the input dimension.
        hidden_dim (int): Size of the hidden state dimension.
        kan_type (str, optional): Type of KAN layer ('spline', 'chebyshev', 'fourier'). Defaults to 'fourier'.
        sub_kan_configs (dict, optional): Configuration for KAN sub-layers. Defaults to None.
        sub_kan_output_dim (int, optional): Output dimension of KAN sub-layers. Defaults to None (uses input_dim).
        sub_kan_input_dim (int, optional): Input dimension of KAN sub-layers. Defaults to None (uses input_dim).
        activation (callable, optional): Activation function for candidate state. Defaults to torch.tanh.
        recurrent_activation (callable, optional): Activation for gates. Defaults to torch.sigmoid.
        use_bias (bool, optional): Whether to use bias in gates and aggregation. Defaults to True.
        dropout (float, optional): Dropout rate for input. Defaults to 0.0.
        recurrent_dropout (float, optional): Dropout rate for recurrent connections. Defaults to 0.0.
        layer_norm (bool, optional): Whether to apply layer normalization. Defaults to False.
        num_sub_layers (int, optional): Number of KAN sub-layers. Defaults to 1.

    Example:
        >>> import torch
        >>> cell = TKANGRUCell(input_dim=1, hidden_dim=16, kan_type='fourier')
        >>> x = torch.randn(32, 1)  # batch_size=32, input_dim=1
        >>> h, states = cell(x)
        >>> print(h.shape)  # Expected: torch.Size([32, 16])
    
    """
    def __init__(self, input_dim, hidden_dim, kan_type='fourier', sub_kan_configs=None, 
                 sub_kan_output_dim=None, sub_kan_input_dim=None, activation=torch.tanh, 
                 recurrent_activation=torch.sigmoid, use_bias=True, dropout=0.0, 
                 recurrent_dropout=0.0, layer_norm=False, num_sub_layers=1):
        super(TKANGRUCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sub_kan_configs = sub_kan_configs or {}
        self.sub_kan_input_dim = sub_kan_input_dim or input_dim
        self.sub_kan_output_dim = sub_kan_output_dim or input_dim
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.layer_norm = layer_norm
        self.num_sub_layers = num_sub_layers

        if kan_type == 'spline':
            model = Spline 
        elif kan_type == 'chebyshev':
            model = Chebyshev
        elif kan_type == 'fourier':
            model = Fourier
        else:
            raise ValueError("Unsupported kan_type")

        self.W_r = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_r = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_r = nn.Parameter(torch.Tensor(hidden_dim)) if use_bias else None

        self.W_z = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_z = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_z = nn.Parameter(torch.Tensor(hidden_dim)) if use_bias else None

        self.tkan_sub_layers = nn.ModuleList([
            model(self.sub_kan_input_dim, self.sub_kan_output_dim, **self.sub_kan_configs)
            for _ in range(num_sub_layers)
        ])
        self.sub_tkan_kernel = nn.Parameter(torch.Tensor(num_sub_layers, self.sub_kan_output_dim * 2))
        self.sub_tkan_recurrent_kernel_inputs = nn.Parameter(torch.Tensor(num_sub_layers, input_dim, self.sub_kan_input_dim))
        self.sub_tkan_recurrent_kernel_h = nn.Parameter(torch.Tensor(num_sub_layers, hidden_dim, self.sub_kan_input_dim))
        self.sub_tkan_recurrent_kernel_states = nn.Parameter(torch.Tensor(num_sub_layers, self.sub_kan_output_dim, self.sub_kan_input_dim))

        self.W_agg = nn.Parameter(torch.Tensor(num_sub_layers * self.sub_kan_output_dim, hidden_dim))
        self.b_agg = nn.Parameter(torch.Tensor(hidden_dim)) if use_bias else None

        if layer_norm:
            self.ln = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_r, nonlinearity='relu')
        nn.init.orthogonal_(self.U_r)
        if self.b_r is not None:
            nn.init.zeros_(self.b_r)
        
        nn.init.kaiming_uniform_(self.W_z, nonlinearity='relu')
        nn.init.orthogonal_(self.U_z)
        if self.b_z is not None:
            nn.init.zeros_(self.b_z)

        nn.init.kaiming_uniform_(self.sub_tkan_recurrent_kernel_inputs, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.sub_tkan_recurrent_kernel_h, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.sub_tkan_recurrent_kernel_states, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.W_agg, nonlinearity='relu')
        if self.b_agg is not None:
            nn.init.zeros_(self.b_agg)
        nn.init.kaiming_uniform_(self.sub_tkan_kernel, nonlinearity='relu')

    def forward(self, x, states=None):
        """Computes one time step of the KAN-enhanced GRU cell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            states (list, optional): List of previous hidden state and sub-states. Defaults to None.

        Returns:
            tuple: (hidden state, updated states), where hidden state is of shape (batch_size, hidden_dim)
                   and states is a list of tensors for the next step.
        """
        if states is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            sub_states = [torch.zeros(x.size(0), self.sub_kan_output_dim, device=x.device) 
                          for _ in range(self.num_sub_layers)]
            states = [h] + sub_states
        h_tm1, *sub_states = states

        if self.training:
            x = F.dropout(x, p=self.dropout, training=True) if self.dropout > 0 else x
            h_tm1 = F.dropout(h_tm1, p=self.recurrent_dropout, training=True) if self.recurrent_dropout > 0 else h_tm1

        r_t = self.recurrent_activation(torch.matmul(x, self.W_r) + torch.matmul(h_tm1, self.U_r) + 
                                        (self.b_r if self.b_r is not None else 0))

        z_t = self.recurrent_activation(torch.matmul(x, self.W_z) + torch.matmul(h_tm1, self.U_z) + 
                                        (self.b_z if self.b_z is not None else 0))

        sub_outputs = []
        new_sub_states = []
        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            agg_input = (torch.matmul(x, self.sub_tkan_recurrent_kernel_inputs[idx]) +
                         torch.matmul(r_t * h_tm1, self.sub_tkan_recurrent_kernel_h[idx]) +
                         torch.matmul(sub_state, self.sub_tkan_recurrent_kernel_states[idx]))
            sub_output = sub_layer(agg_input)
            sub_recurrent_kernel_h, sub_recurrent_kernel_x = torch.chunk(self.sub_tkan_kernel[idx], 2)
            new_sub_state = sub_recurrent_kernel_h * sub_output + sub_state * sub_recurrent_kernel_x
            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)

        aggregated_sub_output = torch.cat(sub_outputs, dim=-1)
        h_candidate = self.activation(torch.matmul(aggregated_sub_output, self.W_agg) + 
                                      (self.b_agg if self.b_agg is not None else 0))

        h_t = (1 - z_t) * h_tm1 + z_t * h_candidate

        if self.layer_norm:
            h_t = self.ln(h_t)

        return h_t, [h_t] + new_sub_states

class tKANGRU(nn.Module):
    """A KAN-enhanced GRU model for time series processing.

    This class wraps TKANGRUCell to process full sequences, with options for bidirectional processing
    and sequence output.

    Args:
        input_dim (int): Size of the input dimension.
        hidden_dim (int): Size of the hidden state dimension.
        sub_kan_configs (dict, optional): Configuration for KAN sub-layers. Defaults to None.
        sub_kan_output_dim (int, optional): Output dimension of KAN sub-layers. Defaults to None.
        sub_kan_input_dim (int, optional): Input dimension of KAN sub-layers. Defaults to None.
        activation (callable, optional): Activation function for candidate state. Defaults to torch.tanh.
        recurrent_activation (callable, optional): Activation for gates. Defaults to torch.sigmoid.
        dropout (float, optional): Dropout rate for input. Defaults to 0.0.
        recurrent_dropout (float, optional): Dropout rate for recurrent connections. Defaults to 0.0.
        return_sequences (bool, optional): Whether to return the full sequence. Defaults to False.
        bidirectional (bool, optional): Whether to process bidirectionally. Defaults to False.
        layer_norm (bool, optional): Whether to apply layer normalization. Defaults to False.
        kan_type (str, optional): Type of KAN layer ('spline', 'chebyshev', 'fourier'). Defaults to 'fourier'.

    Example:
        >>> import torch
        >>> model = tKANGRU(input_dim=1, hidden_dim=16, return_sequences=True, bidirectional=True)
        >>> x = torch.randn(32, 10, 1)  # batch_size=32, seq_len=10, input_dim=1
        >>> output = model(x)
        >>> print(output.shape)  # Expected: torch.Size([32, 10, 32]) due to bidirectional
    """
    def __init__(self, input_dim, hidden_dim, sub_kan_configs=None, sub_kan_output_dim=None, 
                 sub_kan_input_dim=None, activation=torch.tanh, recurrent_activation=torch.sigmoid, 
                 dropout=0.0, recurrent_dropout=0.0, return_sequences=False, 
                 bidirectional=False, layer_norm=False, kan_type='fourier'):
        super(tKANGRU, self).__init__()
        
        self.cell = TKANGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kan_type=kan_type, 
                                sub_kan_configs=sub_kan_configs, sub_kan_output_dim=sub_kan_output_dim, 
                                sub_kan_input_dim=sub_kan_input_dim, activation=activation, 
                                recurrent_activation=recurrent_activation, dropout=dropout, 
                                recurrent_dropout=recurrent_dropout, layer_norm=layer_norm)
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        if bidirectional:
            self.reverse_cell = TKANGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kan_type=kan_type, 
                                            sub_kan_configs=sub_kan_configs, sub_kan_output_dim=sub_kan_output_dim, 
                                            sub_kan_input_dim=sub_kan_input_dim, activation=activation, 
                                            recurrent_activation=recurrent_activation, dropout=dropout, 
                                            recurrent_dropout=recurrent_dropout, layer_norm=layer_norm)

    def forward(self, x, initial_states=None):
        """Processes an input sequence through the KAN-enhanced GRU.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            initial_states (list, optional): Initial states for forward pass. Defaults to None.

        Returns:
            torch.Tensor: Output tensor, shape depends on return_sequences and bidirectional settings.
        """
        batch_size, seq_len, _ = x.shape
        outputs = []
        states = initial_states

        for t in range(seq_len):
            h, states = self.cell(x[:, t, :], states)
            outputs.append(h)
        forward_outputs = torch.stack(outputs, dim=1)

        if not self.bidirectional:
            return forward_outputs if self.return_sequences else forward_outputs[:, -1, :]

        backward_outputs = []
        backward_states = initial_states
        for t in range(seq_len - 1, -1, -1):
            h, backward_states = self.reverse_cell(x[:, t, :], backward_states)
            backward_outputs.insert(0, h)
        backward_outputs = torch.stack(backward_outputs, dim=1)

        combined_outputs = torch.cat([forward_outputs, backward_outputs], dim=-1)
        return combined_outputs if self.return_sequences else combined_outputs[:, -1, :]