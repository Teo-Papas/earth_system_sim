"""
ConvLSTM implementation for spatiotemporal modeling in the physical system.
This module provides the building blocks for the PINN-based atmosphere/ocean simulation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Union

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell for processing spatiotemporal data with both convolution operations
    (for spatial features) and LSTM gating (for temporal dependencies).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = None,
        activation: str = "tanh",
        recurrent_activation: str = "sigmoid",
        bias: bool = True,
    ):
        """
        Initialize the ConvLSTM cell.
        
        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden channels
            kernel_size: Size of the convolutional kernel
            padding: Padding size (default: same as kernel_size//2)
            activation: Activation function for cell state
            recurrent_activation: Activation function for gates
            bias: Whether to use bias in convolution
        """
        super(ConvLSTMCell, self).__init__()
        
        # Save parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Handle kernel size and padding
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if padding is None:
            self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        else:
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Activation functions
        if activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
            
        if recurrent_activation == "sigmoid":
            self.recurrent_activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported recurrent activation: {recurrent_activation}")
        
        # Initialize the convolution for all gates at once
        # Input: (batch, input_dim + hidden_dim, height, width)
        # Output: (batch, 4 * hidden_dim, height, width) for i, f, o, g gates
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # 4 for input, forget, output, and cell gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, x: torch.Tensor, h_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConvLSTM cell.
        
        Args:
            x: Input tensor of shape (batch, input_dim, height, width)
            h_state: Tuple of (h, c) each with shape (batch, hidden_dim, height, width)
        
        Returns:
            Tuple of (h, c) for next time step
        """
        h_prev, c_prev = h_state
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute all gates in one convolution
        gates = self.conv(combined)
        
        # Split gates
        chunks = gates.chunk(4, dim=1)
        i = self.recurrent_activation(chunks[0])  # input gate
        f = self.recurrent_activation(chunks[1])  # forget gate
        o = self.recurrent_activation(chunks[2])  # output gate
        g = self.activation(chunks[3])            # cell gate
        
        # Update cell state
        c_next = f * c_prev + i * g
        
        # Compute hidden state
        h_next = o * self.activation(c_next)
        
        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            height: Height of feature map
            width: Width of feature map
            device: Device to create tensors on
        
        Returns:
            Tuple of (h0, c0) with shape (batch, hidden_dim, height, width)
        """
        if device is None:
            device = next(self.parameters()).device
            
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM for spatiotemporal sequence modeling.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        kernel_size: Union[int, Tuple[int, int]],
        num_layers: int,
        batch_first: bool = True,
        bias: bool = True,
        return_sequence: bool = True
    ):
        """
        Initialize the multi-layer ConvLSTM.
        
        Args:
            input_dim: Number of input channels
            hidden_dims: List of hidden dimensions for each layer
            kernel_size: Size of the convolutional kernel
            num_layers: Number of ConvLSTM layers
            batch_first: If True, batch dimension is first
            bias: Whether to use bias in convolution
            return_sequence: If True, return output for all timesteps
        """
        super(ConvLSTM, self).__init__()
        
        # Parameter validation
        if not len(hidden_dims) == num_layers:
            raise ValueError("Length of hidden_dims must match num_layers")
            
        # Model parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_sequence = return_sequence
        
        # Create ConvLSTM layers
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            cell = ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dims[i],
                kernel_size=kernel_size,
                bias=bias
            )
            cell_list.append(cell)
        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: List[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the multi-layer ConvLSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim, height, width)
               if batch_first, else (seq_len, batch, input_dim, height, width)
            hidden_states: List of initial hidden states for each layer
        
        Returns:
            Tuple of (output, hidden_states)
            - output: All sequence outputs if return_sequence, else only last output
            - hidden_states: Final hidden states for each layer
        """
        if not self.batch_first:
            # Make batch first
            x = x.permute(1, 0, 2, 3, 4)
            
        batch_size, seq_len, _, height, width = x.size()
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = []
            for i in range(self.num_layers):
                hidden_states.append(
                    self.cell_list[i].init_hidden(batch_size, height, width)
                )
        
        # Lists to hold the outputs
        layer_output_list = []
        last_states_list = []
        
        # Process each layer
        cur_layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_states[layer_idx]
            output_inner = []
            
            # Process sequence
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    cur_layer_input[:, t, :, :, :],
                    (h, c)
                )
                output_inner.append(h)
                
            # Stack time dimension
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_states_list.append((h, c))
        
        # Return based on settings
        if not self.return_sequence:
            return layer_output_list[-1][:, -1:, :, :, :], last_states_list
        else:
            return layer_output_list[-1], last_states_list