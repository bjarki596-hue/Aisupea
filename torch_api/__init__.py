"""
Aisupea Torch-style API

A lightweight torch-like interface wrapping the core tensor engine.
"""

import math
from typing import List, Optional, Union, Tuple
from ..core import Tensor


def tensor(data: Union[List, float, int], dtype: Optional[str] = None) -> Tensor:
    """Create a tensor from data."""
    return Tensor(data, dtype)


def zeros(*shape: int, dtype: str = 'float') -> Tensor:
    """Create tensor filled with zeros."""
    return Tensor.zeros(*shape, dtype=dtype)


def ones(*shape: int, dtype: str = 'float') -> Tensor:
    """Create tensor filled with ones."""
    return Tensor.ones(*shape, dtype=dtype)


def arange(start: int, end: int, step: int = 1, dtype: str = 'int') -> Tensor:
    """Create 1D tensor with range of values."""
    return Tensor.arange(start, end, step, dtype)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication."""
    return a.matmul(b)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Apply softmax along dimension."""
    return x.softmax(dim)


def argmax(x: Tensor, dim: int = -1) -> Tensor:
    """Argmax along dimension."""
    return x.argmax(dim)


def sum(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Sum tensor along dimension."""
    return x.sum(dim, keepdim)


def mean(x: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    """Mean tensor along dimension."""
    return x.mean(dim, keepdim)


def exp(x: Tensor) -> Tensor:
    """Element-wise exponential."""
    return x.exp()


def log(x: Tensor) -> Tensor:
    """Element-wise natural log."""
    return x.log()


def sqrt(x: Tensor) -> Tensor:
    """Element-wise square root."""
    return x.sqrt()


def gelu(x: Tensor) -> Tensor:
    """GELU activation function."""
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + ((sqrt_2_pi * (x + 0.044715 * (x ** 3))).tanh()))


def relu(x: Tensor) -> Tensor:
    """ReLU activation function."""
    # For now, implement element-wise max(0, x)
    # This is a simplified version
    def relu_op(val):
        return max(0, val)

    return Tensor(x._apply_func_recursive(x.data, relu_op), x.dtype)


def tanh(x: Tensor) -> Tensor:
    """Tanh activation function."""
    def tanh_op(val):
        return math.tanh(val)

    return Tensor(x._apply_func_recursive(x.data, tanh_op), x.dtype)


def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along dimension."""
    if not tensors:
        raise ValueError("At least one tensor required")
    return tensors[0].cat(tensors[1:], dim)


def chunk(x: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
    """Split tensor into chunks along dimension."""
    return x.chunk(chunks, dim)


def transpose(x: Tensor, *dims: int) -> Tensor:
    """Transpose tensor along given dimensions."""
    return x.transpose(*dims)


def reshape(x: Tensor, *shape: int) -> Tensor:
    """Reshape tensor to new shape."""
    return x.reshape(*shape)


def view(x: Tensor, *shape: int) -> Tensor:
    """View tensor with different shape."""
    return x.view(*shape)


def repeat(x: Tensor, *repeats: int) -> Tensor:
    """Repeat tensor along dimensions."""
    return x.repeat(*repeats)
