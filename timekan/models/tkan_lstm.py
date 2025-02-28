import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import KANLinear, ChebyKANLayer, NaiveFourierKANLayer



class TKANCell(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kan_type,
        sub_kan_configs=None,
        sub_kan_output_dim=None,
        sub_kan_input_dim=None,
        activation=torch.tanh,
        recurrent_activation=torch.sigmoid,
        use_bias=True,
        dropout=0.0,
        recurrent_dropout=0.0,
        layer_norm=False,
        num_sub_layers=1,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sub_kan_configs = sub_kan_configs or [None]
        self.sub_kan_input_dim = sub_kan_input_dim or input_dim
        self.sub_kan_output_dim = sub_kan_output_dim or input_dim
        
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.layer_norm = layer_norm
        self.kan_type = kan_type
    
        # LSTM
        self.kernel = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 3))
        self.recurrent_kernel = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 3))
        self.sub_kan_configs = sub_kan_configs or {}
        
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_dim * 3))
        else:
            self.register_parameter('bias', None)
        
        if kan_type == 'spline':
            model = KANLinear
        elif kan_type == 'chebychev':
            model = ChebyKANLayer
        elif kan_type == 'fourier':
            model = NaiveFourierKANLayer
        
        self.tkan_sub_layers = nn.ModuleList()
        for _ in range(num_sub_layers):
            layer = model(inputdim=self.sub_kan_input_dim, outdim=self.sub_kan_output_dim, **self.sub_kan_configs) 
            self.tkan_sub_layers.append(layer)
        
        self.sub_tkan_kernel = nn.Parameter(torch.Tensor(len(self.tkan_sub_layers), self.sub_kan_output_dim * 2))
        self.sub_tkan_recurrent_kernel_inputs = nn.Parameter(torch.Tensor(len(self.tkan_sub_layers), input_dim, self.sub_kan_input_dim))
        self.sub_tkan_recurrent_kernel_states = nn.Parameter(torch.Tensor(len(self.tkan_sub_layers), self.sub_kan_output_dim, self.sub_kan_input_dim))
        
        self.aggregated_weight = nn.Parameter(torch.Tensor(len(self.tkan_sub_layers) * self.sub_kan_output_dim, hidden_dim))
        self.aggregated_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        
        if self.layer_norm:
            self.ln = nn.LayerNorm(hidden_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kernel, nonlinearity='relu')
        nn.init.orthogonal_(self.recurrent_kernel)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        nn.init.kaiming_uniform_(self.sub_tkan_kernel, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.sub_tkan_recurrent_kernel_inputs, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.sub_tkan_recurrent_kernel_states, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.aggregated_weight, nonlinearity='relu')
        nn.init.zeros_(self.aggregated_bias)

    def forward(self, x, states=None):
        if states is None:
            h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            c = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            sub_states = [torch.zeros(x.size(0), self.sub_kan_output_dim, device=x.device) 
                          for _ in range(len(self.tkan_sub_layers))]
            states = [h, c] + sub_states
        
        h_tm1, c_tm1, *sub_states = states
        
        if self.training:
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.recurrent_dropout > 0:
                h_tm1 = F.dropout(h_tm1, p=self.recurrent_dropout, training=self.training)
        
        # Compute gates
        gates = torch.matmul(x, self.kernel) + torch.matmul(h_tm1, self.recurrent_kernel)
        if self.bias is not None:
            gates += self.bias
        
        gates = self.recurrent_activation(gates)
        i, f, c_candidate = torch.chunk(gates, 3, dim=-1)
        
        c = f * c_tm1 + i * self.activation(c_candidate)
        
        sub_outputs = []
        new_sub_states = []
        
        for idx, (sub_layer, sub_state) in enumerate(zip(self.tkan_sub_layers, sub_states)):
            sub_kernel_x = self.sub_tkan_recurrent_kernel_inputs[idx]
            sub_kernel_h = self.sub_tkan_recurrent_kernel_states[idx]
            
            agg_input = torch.matmul(x, sub_kernel_x) + torch.matmul(sub_state, sub_kernel_h)
            sub_output = sub_layer(agg_input)
            
            sub_recurrent_kernel_h, sub_recurrent_kernel_x = torch.chunk(self.sub_tkan_kernel[idx], 2)
            new_sub_state = sub_recurrent_kernel_h * sub_output + sub_state * sub_recurrent_kernel_x
            
            sub_outputs.append(sub_output)
            new_sub_states.append(new_sub_state)
        
        # Aggregate sub-layer outputs
        aggregated_sub_output = torch.cat(sub_outputs, dim=-1)
        aggregated_input = torch.matmul(aggregated_sub_output, self.aggregated_weight) + self.aggregated_bias
        
        o = self.recurrent_activation(aggregated_input)
        h = o * self.activation(c)
        
        if self.layer_norm:
            h = self.ln(h)
        
        return h, [h, c] + new_sub_states


class tKANLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        sub_kan_configs=None,
        sub_kan_output_dim=None,
        sub_kan_input_dim=None,
        activation=torch.tanh,
        recurrent_activation=torch.sigmoid,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=False,
        bidirectional=False,
        layer_norm=False,
        kan_type='fourier'
    ):
        super().__init__()
        
        self.cell = TKANCell(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sub_kan_configs=sub_kan_configs,
            sub_kan_output_dim=sub_kan_output_dim,
            sub_kan_input_dim=sub_kan_input_dim,
            activation=activation,
            recurrent_activation=recurrent_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            layer_norm=layer_norm,
            kan_type=kan_type
        )
        
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        
        if bidirectional:
            self.reverse_cell = TKANCell(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                sub_kan_configs=sub_kan_configs,
                sub_kan_output_dim=sub_kan_output_dim,
                sub_kan_input_dim=sub_kan_input_dim,
                activation=activation,
                recurrent_activation=recurrent_activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                layer_norm=layer_norm ,
                kan_type=kan_type
            )
    
    def forward(self, x, initial_states=None):
        batch_size, seq_len, input_dim = x.shape
        forward_states = initial_states[:2] if initial_states else None
        forward_outputs = []
        
        for t in range(seq_len):
            h_forward, forward_states = self.cell(x[:, t, :], forward_states)
            forward_outputs.append(h_forward)
        
        forward_outputs = torch.stack(forward_outputs, dim=1)
        
        if not self.bidirectional:
            return forward_outputs if self.return_sequences else forward_outputs[:, -1, :]
        
        backward_states = initial_states[:2] if initial_states else None
        backward_outputs = []
        
        for t in range(seq_len - 1, -1, -1):
            h_backward, backward_states = self.reverse_cell(x[:, t, :], backward_states)
            backward_outputs.insert(0, h_backward)
        
        backward_outputs = torch.stack(backward_outputs, dim=1)
        
        combined_outputs = torch.cat([forward_outputs, backward_outputs], dim=-1)
        
        return combined_outputs if self.return_sequences else combined_outputs[:, -1, :]