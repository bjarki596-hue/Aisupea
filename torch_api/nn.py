"""
Aisupea Neural Network Module

Torch-style nn.Module system with Linear layers, LayerNorm, and activation functions.
"""

import math
from typing import Optional, Union, List, Dict, Any
from ..core import Tensor
from . import functional as F


class Module:
    """
    Base class for all neural network modules.

    Provides parameter management, forward pass interface, and utilities.
    """

    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, 'Module'] = {}
        self._buffers: Dict[str, Tensor] = {}
        self.training = True

    def forward(self, *args, **kwargs) -> Union[Tensor, List[Tensor]]:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")

    def __call__(self, *args, **kwargs) -> Union[Tensor, List[Tensor]]:
        """Call forward method."""
        return self.forward(*args, **kwargs)

    def register_parameter(self, name: str, param: Optional[Tensor]):
        """Register a parameter."""
        self._parameters[name] = param

    def register_module(self, name: str, module: 'Module'):
        """Register a submodule."""
        self._modules[name] = module

    def register_buffer(self, name: str, buffer: Optional[Tensor]):
        """Register a buffer."""
        self._buffers[name] = buffer

    def parameters(self) -> List[Tensor]:
        """Get all parameters recursively."""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return [p for p in params if p is not None]

    def modules(self) -> List['Module']:
        """Get all modules recursively."""
        modules = [self]
        for module in self._modules.values():
            modules.extend(module.modules())
        return modules

    def named_parameters(self) -> Dict[str, Tensor]:
        """Get named parameters recursively."""
        named_params = dict(self._parameters)
        for name, module in self._modules.items():
            for param_name, param in module.named_parameters().items():
                named_params[f"{name}.{param_name}"] = param
        return named_params

    def to(self, dtype: str):
        """Convert all parameters to dtype."""
        for param in self.parameters():
            # In a real implementation, this would convert data types
            pass
        return self

    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def zero_grad(self):
        """Zero out gradients (placeholder for future autograd)."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Linear(Module):
    """
    Linear transformation layer.

    Applies linear transformation: y = x @ W^T + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with Xavier/Glorot initialization
        scale = math.sqrt(2.0 / (in_features + out_features))
        weight_data = []
        for i in range(out_features):
            row = []
            for j in range(in_features):
                row.append(scale * (2 * math.random() - 1))
            weight_data.append(row)

        self.weight = Tensor(weight_data, 'float')
        self.register_parameter('weight', self.weight)

        if bias:
            bias_data = [0.0] * out_features
            self.bias = Tensor(bias_data, 'float')
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # y = x @ W^T + b
        # In our tensor implementation, matmul expects (m, k) @ (k, n)
        # So we need x @ W^T, which means we need to transpose W
        output = x.matmul(self.weight.transpose(1, 0))

        if self.bias is not None:
            # Add bias to last dimension
            output = output + self.bias

        return output


class LayerNorm(Module):
    """
    Layer normalization.

    Normalizes input along the last dimension.
    """

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        if elementwise_affine:
            # Initialize gamma (scale) to 1 and beta (shift) to 0
            gamma_data = [1.0] * normalized_shape[-1]
            beta_data = [0.0] * normalized_shape[-1]

            self.weight = Tensor(gamma_data, 'float')
            self.bias = Tensor(beta_data, 'float')

            self.register_parameter('weight', self.weight)
            self.register_parameter('bias', self.bias)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        # Layer norm: (x - mean) / sqrt(var + eps) * gamma + beta

        # For simplicity, assume we're normalizing along the last dimension
        # In a full implementation, this would handle arbitrary normalized_shape

        # Compute mean and variance along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # Normalize
        normalized = (x - mean) / (var + self.eps).sqrt()

        # Apply affine transformation
        if self.weight is not None:
            normalized = normalized * self.weight

        if self.bias is not None:
            normalized = normalized + self.bias

        return normalized


class Embedding(Module):
    """
    Embedding layer.

    Maps integer indices to dense vectors.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embeddings randomly
        scale = math.sqrt(1.0 / embedding_dim)
        embeddings_data = []
        for i in range(num_embeddings):
            embedding = []
            for j in range(embedding_dim):
                embedding.append(scale * (2 * math.random() - 1))
            embeddings_data.append(embedding)

        self.weight = Tensor(embeddings_data, 'float')
        self.register_parameter('weight', self.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of integer indices

        Returns:
            Embedded tensor
        """
        # For simplicity, assume x contains indices
        # In a full implementation, this would do proper indexing
        # For now, return a placeholder
        batch_size = x.shape[0] if x.ndim > 0 else 1
        seq_len = x.shape[1] if x.ndim > 1 else 1

        # This is a simplified implementation
        # Real embedding would look up indices in weight matrix
        result_shape = (batch_size, seq_len, self.embedding_dim)
        result = Tensor.zeros(*result_shape, dtype='float')

        return result


class Dropout(Module):
    """
    Dropout layer.

    Randomly zeros some elements during training.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor with dropout applied
        """
        if not self.training or self.p == 0.0:
            return x

        # Generate dropout mask
        # For simplicity, this is a placeholder
        # Real implementation would generate random mask
        return x * (1.0 / (1.0 - self.p))  # Scale to maintain expected value


class Sequential(Module):
    """
    Sequential container for modules.
    """

    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            self.register_module(str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all modules in sequence.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        for module in self._modules.values():
            x = module(x)
        return x