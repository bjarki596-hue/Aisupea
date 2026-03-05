"""
Aisupea NumPy Compatibility Module

Minimal ndarray-like interface with vector and matrix operations.
"""

import math
from typing import List, Union, Tuple, Optional
from ..core import Tensor


class ndarray:
    """
    Minimal ndarray-like class wrapping Tensor.
    """

    def __init__(self, data: Union[List, float, int]):
        self._tensor = Tensor(data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._tensor.shape

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    @property
    def dtype(self) -> str:
        return self._tensor.dtype

    def __repr__(self) -> str:
        return f"ndarray({self._tensor})"

    def __str__(self) -> str:
        return self.__repr__()

    def __add__(self, other: Union['ndarray', float, int]) -> 'ndarray':
        if isinstance(other, ndarray):
            result = self._tensor + other._tensor
        else:
            result = self._tensor + other
        return ndarray(result.data)

    def __sub__(self, other: Union['ndarray', float, int]) -> 'ndarray':
        if isinstance(other, ndarray):
            result = self._tensor - other._tensor
        else:
            result = self._tensor - other
        return ndarray(result.data)

    def __mul__(self, other: Union['ndarray', float, int]) -> 'ndarray':
        if isinstance(other, ndarray):
            result = self._tensor * other._tensor
        else:
            result = self._tensor * other
        return ndarray(result.data)

    def __truediv__(self, other: Union['ndarray', float, int]) -> 'ndarray':
        if isinstance(other, ndarray):
            result = self._tensor / other._tensor
        else:
            result = self._tensor / other
        return ndarray(result.data)

    def __matmul__(self, other: 'ndarray') -> 'ndarray':
        result = self._tensor @ other._tensor
        return ndarray(result.data)

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'ndarray':
        result = self._tensor.sum(axis, keepdims)
        return ndarray(result.data)

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'ndarray':
        result = self._tensor.mean(axis, keepdims)
        return ndarray(result.data)

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'ndarray':
        result = self._tensor.max(axis, keepdims)
        return ndarray(result.data)

    def argmax(self, axis: Optional[int] = None) -> 'ndarray':
        result = self._tensor.argmax(axis)
        return ndarray(result.data)

    def reshape(self, *shape) -> 'ndarray':
        result = self._tensor.reshape(*shape)
        return ndarray(result.data)

    def transpose(self, *axes) -> 'ndarray':
        result = self._tensor.transpose(*axes)
        return ndarray(result.data)

    def tolist(self) -> List:
        return self._tensor.tolist()

    def item(self) -> Union[int, float]:
        return self._tensor.item()


def zeros(shape: Union[int, Tuple[int, ...]], dtype: str = 'float') -> ndarray:
    """Create array filled with zeros."""
    if isinstance(shape, int):
        shape = (shape,)
    tensor = Tensor.zeros(*shape, dtype=dtype)
    return ndarray(tensor.data)


def ones(shape: Union[int, Tuple[int, ...]], dtype: str = 'float') -> ndarray:
    """Create array filled with ones."""
    if isinstance(shape, int):
        shape = (shape,)
    tensor = Tensor.ones(*shape, dtype=dtype)
    return ndarray(tensor.data)


def array(data: Union[List, float, int]) -> ndarray:
    """Create array from data."""
    return ndarray(data)


def arange(start: int, end: int, step: int = 1, dtype: str = 'int') -> ndarray:
    """Create 1D array with range of values."""
    tensor = Tensor.arange(start, end, step, dtype)
    return ndarray(tensor.data)


def dot(a: ndarray, b: ndarray) -> ndarray:
    """Dot product of two arrays."""
    if a.ndim == 1 and b.ndim == 1:
        # Vector dot product
        result = sum(x * y for x, y in zip(a.tolist(), b.tolist()))
        return ndarray(result)
    elif a.ndim == 2 and b.ndim == 2:
        # Matrix multiplication
        return a @ b
    else:
        raise ValueError("Unsupported dimensions for dot product")


def normalize(x: ndarray, axis: Optional[int] = None) -> ndarray:
    """Normalize array along axis."""
    if axis is None:
        # Normalize entire array
        norm = math.sqrt(sum(val ** 2 for val in x.tolist()))
        return x / norm
    else:
        # Normalize along axis
        # Simplified implementation
        return x / x.sum(axis, keepdims=True).sqrt()


def eye(n: int, dtype: str = 'float') -> ndarray:
    """Create identity matrix."""
    data = []
    for i in range(n):
        row = [1.0 if j == i else 0.0 for j in range(n)]
        data.append(row)
    return ndarray(data)


def diag(x: ndarray) -> ndarray:
    """Extract diagonal from matrix."""
    if x.ndim != 2:
        raise ValueError("diag requires 2D array")

    n = min(x.shape)
    diagonal = []
    for i in range(n):
        diagonal.append(x._tensor[i, i])
    return ndarray(diagonal)


def trace(x: ndarray) -> Union[int, float]:
    """Sum of diagonal elements."""
    return diag(x).sum().item()


def clip(x: ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None) -> ndarray:
    """Clip array values."""
    def clip_func(val):
        if min_val is not None and val < min_val:
            return min_val
        if max_val is not None and val > max_val:
            return max_val
        return val

    # This is a simplified implementation
    # Real implementation would apply clip_func recursively
    return x  # Placeholder


def exp(x: ndarray) -> ndarray:
    """Element-wise exponential."""
    result = x._tensor.exp()
    return ndarray(result.data)


def log(x: ndarray) -> ndarray:
    """Element-wise natural log."""
    result = x._tensor.log()
    return ndarray(result.data)


def sqrt(x: ndarray) -> ndarray:
    """Element-wise square root."""
    result = x._tensor.sqrt()
    return ndarray(result.data)


def sin(x: ndarray) -> ndarray:
    """Element-wise sine."""
    def sin_func(val):
        return math.sin(val)

    # Simplified implementation
    return x  # Placeholder


def cos(x: ndarray) -> ndarray:
    """Element-wise cosine."""
    def cos_func(val):
        return math.cos(val)

    # Simplified implementation
    return x  # Placeholder


def tanh(x: ndarray) -> ndarray:
    """Element-wise hyperbolic tangent."""
    def tanh_func(val):
        return math.tanh(val)

    # Simplified implementation
    return x  # Placeholder
