"""
Aisupea Functional API

Functional versions of neural network operations.
"""

from typing import Optional
from ..core import Tensor
from . import gelu, relu, tanh, softmax


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Functional linear transformation."""
    output = x.matmul(weight.transpose(1, 0))
    if bias is not None:
        output = output + bias
    return output


def layer_norm(x: Tensor, normalized_shape, weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None, eps: float = 1e-5) -> Tensor:
    """Functional layer normalization."""
    # Simplified implementation
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (x - mean) / (var + eps).sqrt()

    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias

    return normalized


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Functional dropout."""
    if not training or p == 0.0:
        return x
    # Placeholder - real implementation would use random mask
    return x * (1.0 / (1.0 - p))


def embedding(x: Tensor, weight: Tensor) -> Tensor:
    """Functional embedding lookup."""
    # Placeholder - real implementation would do proper indexing
    batch_size = x.shape[0] if x.ndim > 0 else 1
    seq_len = x.shape[1] if x.ndim > 1 else 1
    embedding_dim = weight.shape[1]

    result_shape = (batch_size, seq_len, embedding_dim)
    return Tensor.zeros(*result_shape, dtype='float')