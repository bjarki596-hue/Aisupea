"""
Aisupea RNN Layers Module

Recurrent neural network layers for sequential data.
"""

from typing import Optional, Tuple, List
from ..core import Tensor
import math
import random


class RNNCell:
    """Basic RNN cell."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize RNN cell.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            bias: Whether to include bias terms
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices
        self.weight_ih = Tensor.zeros(hidden_size, input_size)
        self.weight_hh = Tensor.zeros(hidden_size, hidden_size)

        # Bias terms
        self.bias_ih = Tensor.zeros(hidden_size) if bias else None
        self.bias_hh = Tensor.zeros(hidden_size) if bias else None

        # Xavier initialization
        scale_ih = (2.0 / (input_size + hidden_size)) ** 0.5
        scale_hh = (2.0 / (hidden_size + hidden_size)) ** 0.5

        self._init_weights(self.weight_ih, scale_ih)
        self._init_weights(self.weight_hh, scale_hh)

        if self.bias_ih is not None:
            self._init_weights(self.bias_ih, scale_ih)
        if self.bias_hh is not None:
            self._init_weights(self.bias_hh, scale_hh)

    def _init_weights(self, tensor: Tensor, scale: float):
        """Initialize weights with random values."""
        if tensor.ndim == 2:
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    tensor.data[i][j] = random.gauss(0, scale)
        else:  # 1D tensor
            for i in range(tensor.shape[0]):
                tensor.data[i] = random.gauss(0, scale)

    def forward(self, input_tensor: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of RNN cell.

        Args:
            input_tensor: Input tensor of shape (batch_size, input_size)
            hidden: Previous hidden state of shape (batch_size, hidden_size)

        Returns:
            New hidden state of shape (batch_size, hidden_size)
        """
        batch_size = input_tensor.shape[0]

        if hidden is None:
            hidden = Tensor.zeros(batch_size, self.hidden_size)

        # Compute: h = tanh(W_ih * x + b_ih + W_hh * h_prev + b_hh)
        ih = input_tensor @ self.weight_ih.transpose()
        hh = hidden @ self.weight_hh.transpose()

        if self.bias_ih is not None:
            ih = ih + self.bias_ih.unsqueeze(0).expand(batch_size, -1)
        if self.bias_hh is not None:
            hh = hh + self.bias_hh.unsqueeze(0).expand(batch_size, -1)

        combined = ih + hh

        # Apply tanh activation
        output = combined.tanh()

        return output


class LSTMCell:
    """LSTM cell with forget, input, and output gates."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize LSTM cell.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            bias: Whether to include bias terms
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for all gates
        self.weight_ih = Tensor.zeros(4 * hidden_size, input_size)  # Input-to-hidden
        self.weight_hh = Tensor.zeros(4 * hidden_size, hidden_size)  # Hidden-to-hidden

        # Bias terms
        self.bias_ih = Tensor.zeros(4 * hidden_size) if bias else None
        self.bias_hh = Tensor.zeros(4 * hidden_size) if bias else None

        # Xavier initialization
        scale_ih = (2.0 / (input_size + hidden_size)) ** 0.5
        scale_hh = (2.0 / (hidden_size + hidden_size)) ** 0.5

        self._init_weights(self.weight_ih, scale_ih)
        self._init_weights(self.weight_hh, scale_hh)

        if self.bias_ih is not None:
            self._init_weights(self.bias_ih, scale_ih)
            # Initialize forget gate bias to 1 (as in PyTorch)
            for i in range(hidden_size, 2 * hidden_size):
                self.bias_ih.data[i] = 1.0
        if self.bias_hh is not None:
            self._init_weights(self.bias_hh, scale_hh)

    def _init_weights(self, tensor: Tensor, scale: float):
        """Initialize weights with random values."""
        if tensor.ndim == 2:
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    tensor.data[i][j] = random.gauss(0, scale)
        else:  # 1D tensor
            for i in range(tensor.shape[0]):
                tensor.data[i] = random.gauss(0, scale)

    def forward(self, input_tensor: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of LSTM cell.

        Args:
            input_tensor: Input tensor of shape (batch_size, input_size)
            hidden: Tuple of (hidden_state, cell_state), each of shape (batch_size, hidden_size)

        Returns:
            Tuple of (output, (new_hidden_state, new_cell_state))
        """
        batch_size = input_tensor.shape[0]

        if hidden is None:
            h_prev = Tensor.zeros(batch_size, self.hidden_size)
            c_prev = Tensor.zeros(batch_size, self.hidden_size)
        else:
            h_prev, c_prev = hidden

        # Compute all gates: [i, f, g, o]
        gates = input_tensor @ self.weight_ih.transpose() + h_prev @ self.weight_hh.transpose()

        if self.bias_ih is not None:
            gates = gates + self.bias_ih.unsqueeze(0).expand(batch_size, -1)
        if self.bias_hh is not None:
            gates = gates + self.bias_hh.unsqueeze(0).expand(batch_size, -1)

        # Split gates
        i_gate = gates[:, :self.hidden_size].sigmoid()      # Input gate
        f_gate = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()  # Forget gate
        g_gate = gates[:, 2*self.hidden_size:3*self.hidden_size].tanh()   # Cell gate
        o_gate = gates[:, 3*self.hidden_size:].sigmoid()     # Output gate

        # Update cell state: c = f * c_prev + i * g
        c_new = f_gate * c_prev + i_gate * g_gate

        # Update hidden state: h = o * tanh(c)
        h_new = o_gate * c_new.tanh()

        return h_new, (h_new, c_new)


class GRUCell:
    """GRU cell with reset and update gates."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize GRU cell.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            bias: Whether to include bias terms
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for all gates
        self.weight_ih = Tensor.zeros(3 * hidden_size, input_size)  # Input-to-hidden
        self.weight_hh = Tensor.zeros(3 * hidden_size, hidden_size)  # Hidden-to-hidden

        # Bias terms
        self.bias_ih = Tensor.zeros(3 * hidden_size) if bias else None
        self.bias_hh = Tensor.zeros(3 * hidden_size) if bias else None

        # Xavier initialization
        scale_ih = (2.0 / (input_size + hidden_size)) ** 0.5
        scale_hh = (2.0 / (hidden_size + hidden_size)) ** 0.5

        self._init_weights(self.weight_ih, scale_ih)
        self._init_weights(self.weight_hh, scale_hh)

        if self.bias_ih is not None:
            self._init_weights(self.bias_ih, scale_ih)
        if self.bias_hh is not None:
            self._init_weights(self.bias_hh, scale_hh)

    def _init_weights(self, tensor: Tensor, scale: float):
        """Initialize weights with random values."""
        if tensor.ndim == 2:
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    tensor.data[i][j] = random.gauss(0, scale)
        else:  # 1D tensor
            for i in range(tensor.shape[0]):
                tensor.data[i] = random.gauss(0, scale)

    def forward(self, input_tensor: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of GRU cell.

        Args:
            input_tensor: Input tensor of shape (batch_size, input_size)
            hidden: Previous hidden state of shape (batch_size, hidden_size)

        Returns:
            New hidden state of shape (batch_size, hidden_size)
        """
        batch_size = input_tensor.shape[0]

        if hidden is None:
            hidden = Tensor.zeros(batch_size, self.hidden_size)

        # Compute all gates: [r, z, n]
        gates = input_tensor @ self.weight_ih.transpose() + hidden @ self.weight_hh.transpose()

        if self.bias_ih is not None:
            gates = gates + self.bias_ih.unsqueeze(0).expand(batch_size, -1)
        if self.bias_hh is not None:
            gates = gates + self.bias_hh.unsqueeze(0).expand(batch_size, -1)

        # Split gates
        r_gate = gates[:, :self.hidden_size].sigmoid()      # Reset gate
        z_gate = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()  # Update gate
        n_gate = gates[:, 2*self.hidden_size:].tanh()       # New gate

        # Update hidden state: h = z * h_prev + (1 - z) * (r * h_prev + n)
        r_h = r_gate * hidden
        n_tilde = r_h @ self.weight_hh.transpose(2*self.hidden_size, 3*self.hidden_size).transpose() + n_gate

        if self.bias_hh is not None:
            n_tilde = n_tilde + self.bias_hh[2*self.hidden_size:].unsqueeze(0).expand(batch_size, -1)

        n_tilde = n_tilde.tanh()

        h_new = z_gate * hidden + (Tensor.ones_like(z_gate) - z_gate) * n_tilde

        return h_new


class RNN:
    """Multi-layer RNN."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = True):
        """
        Initialize RNN.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of RNN layers
            nonlinearity: Nonlinearity ('tanh' or 'relu')
            bias: Whether to include bias terms
            batch_first: Whether batch dimension is first
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first

        # Create RNN cells for each layer
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            if nonlinearity == 'tanh':
                cell = RNNCell(layer_input_size, hidden_size, bias)
            elif nonlinearity == 'relu':
                # For simplicity, use tanh (relu would need modification)
                cell = RNNCell(layer_input_size, hidden_size, bias)
            else:
                raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
            self.cells.append(cell)

    def forward(self, input_tensor: Tensor, hidden: Optional[List[Tensor]] = None) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass of RNN.

        Args:
            input_tensor: Input tensor of shape (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
            hidden: Initial hidden states for each layer

        Returns:
            Tuple of (output, hidden_states)
        """
        if self.batch_first:
            input_tensor = input_tensor.transpose(0, 1)  # (seq_len, batch_size, input_size)

        seq_len, batch_size, _ = input_tensor.shape

        if hidden is None:
            hidden = [Tensor.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        outputs = []
        current_hidden = hidden

        for t in range(seq_len):
            x_t = input_tensor[t]  # (batch_size, input_size)
            layer_hidden = []

            for layer in range(self.num_layers):
                h_prev = current_hidden[layer]
                h_new = self.cells[layer].forward(x_t, h_prev)
                layer_hidden.append(h_new)
                x_t = h_new  # Output of this layer is input to next

            current_hidden = layer_hidden
            outputs.append(x_t)

        # Stack outputs
        output = Tensor.zeros(seq_len, batch_size, self.hidden_size)
        for t in range(seq_len):
            output.data[t] = outputs[t].data

        if self.batch_first:
            output = output.transpose(0, 1)  # (batch_size, seq_len, hidden_size)

        return output, current_hidden


class LSTM:
    """Multi-layer LSTM."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = True):
        """
        Initialize LSTM.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            bias: Whether to include bias terms
            batch_first: Whether batch dimension is first
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Create LSTM cells for each layer
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = LSTMCell(layer_input_size, hidden_size, bias)
            self.cells.append(cell)

    def forward(self, input_tensor: Tensor, hidden: Optional[List[Tuple[Tensor, Tensor]]] = None) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Forward pass of LSTM.

        Args:
            input_tensor: Input tensor of shape (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
            hidden: Initial hidden and cell states for each layer

        Returns:
            Tuple of (output, hidden_states)
        """
        if self.batch_first:
            input_tensor = input_tensor.transpose(0, 1)  # (seq_len, batch_size, input_size)

        seq_len, batch_size, _ = input_tensor.shape

        if hidden is None:
            hidden = [(Tensor.zeros(batch_size, self.hidden_size), Tensor.zeros(batch_size, self.hidden_size))
                     for _ in range(self.num_layers)]

        outputs = []
        current_hidden = hidden

        for t in range(seq_len):
            x_t = input_tensor[t]  # (batch_size, input_size)
            layer_hidden = []

            for layer in range(self.num_layers):
                h_prev, c_prev = current_hidden[layer]
                h_new, (h_new_full, c_new) = self.cells[layer].forward(x_t, (h_prev, c_prev))
                layer_hidden.append((h_new_full, c_new))
                x_t = h_new_full  # Output of this layer is input to next

            current_hidden = layer_hidden
            outputs.append(x_t)

        # Stack outputs
        output = Tensor.zeros(seq_len, batch_size, self.hidden_size)
        for t in range(seq_len):
            output.data[t] = outputs[t].data

        if self.batch_first:
            output = output.transpose(0, 1)  # (batch_size, seq_len, hidden_size)

        return output, current_hidden


class GRU:
    """Multi-layer GRU."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = True):
        """
        Initialize GRU.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of GRU layers
            bias: Whether to include bias terms
            batch_first: Whether batch dimension is first
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Create GRU cells for each layer
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = GRUCell(layer_input_size, hidden_size, bias)
            self.cells.append(cell)

    def forward(self, input_tensor: Tensor, hidden: Optional[List[Tensor]] = None) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass of GRU.

        Args:
            input_tensor: Input tensor of shape (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
            hidden: Initial hidden states for each layer

        Returns:
            Tuple of (output, hidden_states)
        """
        if self.batch_first:
            input_tensor = input_tensor.transpose(0, 1)  # (seq_len, batch_size, input_size)

        seq_len, batch_size, _ = input_tensor.shape

        if hidden is None:
            hidden = [Tensor.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]

        outputs = []
        current_hidden = hidden

        for t in range(seq_len):
            x_t = input_tensor[t]  # (batch_size, input_size)
            layer_hidden = []

            for layer in range(self.num_layers):
                h_prev = current_hidden[layer]
                h_new = self.cells[layer].forward(x_t, h_prev)
                layer_hidden.append(h_new)
                x_t = h_new  # Output of this layer is input to next

            current_hidden = layer_hidden
            outputs.append(x_t)

        # Stack outputs
        output = Tensor.zeros(seq_len, batch_size, self.hidden_size)
        for t in range(seq_len):
            output.data[t] = outputs[t].data

        if self.batch_first:
            output = output.transpose(0, 1)  # (batch_size, seq_len, hidden_size)

        return output, current_hidden


__all__ = [
    'RNNCell',
    'LSTMCell',
    'GRUCell',
    'RNN',
    'LSTM',
    'GRU'
]