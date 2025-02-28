import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import KANLinear, ChebyKANLayer, NaiveFourierKANLayer

class TKANGRUCell(nn.Module):
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

        # Select KAN model based on kan_type
        if kan_type == 'spline':
            model = KANLinear  # Replace with your KAN implementation
        elif kan_type == 'chebychev':
            model = ChebyKANLayer
        elif kan_type == 'fourier':
            model = NaiveFourierKANLayer
        else:
            raise ValueError("Unsupported kan_type")

        # Reset gate parameters
        self.W_r = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_r = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_r = nn.Parameter(torch.Tensor(hidden_dim)) if use_bias else None

        # Update gate parameters
        self.W_z = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.U_z = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_z = nn.Parameter(torch.Tensor(hidden_dim)) if use_bias else None

        # KAN sub-layers and their parameters
        self.tkan_sub_layers = nn.ModuleList([
            model(self.sub_kan_input_dim, self.sub_kan_output_dim, **self.sub_kan_configs)
            for _ in range(num_sub_layers)
        ])
        self.sub_tkan_kernel = nn.Parameter(torch.Tensor(num_sub_layers, self.sub_kan_output_dim * 2))
        self.sub_tkan_recurrent_kernel_inputs = nn.Parameter(torch.Tensor(num_sub_layers, input_dim, self.sub_kan_input_dim))
        self.sub_tkan_recurrent_kernel_h = nn.Parameter(torch.Tensor(num_sub_layers, hidden_dim, self.sub_kan_input_dim))
        self.sub_tkan_recurrent_kernel_states = nn.Parameter(torch.Tensor(num_sub_layers, self.sub_kan_output_dim, self.sub_kan_input_dim))

        # Aggregation parameters
        self.W_agg = nn.Parameter(torch.Tensor(num_sub_layers * self.sub_kan_output_dim, hidden_dim))
        self.b_agg = nn.Parameter(torch.Tensor(hidden_dim)) if use_bias else None

        # Optional layer normalization
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
        # Initialize states if not provided
        if states is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            sub_states = [torch.zeros(x.size(0), self.sub_kan_output_dim, device=x.device) 
                          for _ in range(self.num_sub_layers)]
            states = [h] + sub_states
        h_tm1, *sub_states = states

        # Apply dropout if training
        if self.training:
            x = F.dropout(x, p=self.dropout, training=True) if self.dropout > 0 else x
            h_tm1 = F.dropout(h_tm1, p=self.recurrent_dropout, training=True) if self.recurrent_dropout > 0 else h_tm1

        # Reset gate
        r_t = self.recurrent_activation(torch.matmul(x, self.W_r) + torch.matmul(h_tm1, self.U_r) + 
                                        (self.b_r if self.b_r is not None else 0))

        # Update gate
        z_t = self.recurrent_activation(torch.matmul(x, self.W_z) + torch.matmul(h_tm1, self.U_z) + 
                                        (self.b_z if self.b_z is not None else 0))

        # KAN sub-layer computations
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

        # Aggregate sub-outputs to compute candidate hidden state
        aggregated_sub_output = torch.cat(sub_outputs, dim=-1)
        h_candidate = self.activation(torch.matmul(aggregated_sub_output, self.W_agg) + 
                                      (self.b_agg if self.b_agg is not None else 0))

        # Update hidden state
        h_t = (1 - z_t) * h_tm1 + z_t * h_candidate

        # Apply layer normalization if enabled
        if self.layer_norm:
            h_t = self.ln(h_t)

        return h_t, [h_t] + new_sub_states

class tKANGRU(nn.Module):
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
        batch_size, seq_len, _ = x.shape
        outputs = []
        states = initial_states

        # Forward pass
        for t in range(seq_len):
            h, states = self.cell(x[:, t, :], states)
            outputs.append(h)
        forward_outputs = torch.stack(outputs, dim=1)

        if not self.bidirectional:
            return forward_outputs if self.return_sequences else forward_outputs[:, -1, :]

        # Backward pass
        backward_outputs = []
        backward_states = initial_states
        for t in range(seq_len - 1, -1, -1):
            h, backward_states = self.reverse_cell(x[:, t, :], backward_states)
            backward_outputs.insert(0, h)
        backward_outputs = torch.stack(backward_outputs, dim=1)

        # Combine forward and backward outputs
        combined_outputs = torch.cat([forward_outputs, backward_outputs], dim=-1)
        return combined_outputs if self.return_sequences else combined_outputs[:, -1, :]