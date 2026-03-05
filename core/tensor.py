"""
Aisupea Core Tensor Engine

A lightweight, pure Python tensor implementation with multidimensional arrays,
efficient nested list storage, and comprehensive mathematical operations.
"""

import math
import copy
from typing import List, Tuple, Union, Optional, Any


class Tensor:
    """
    Multidimensional tensor class with efficient nested list storage.

    Features:
    - Shape tracking and validation
    - Reshape and view operations
    - Transpose operations
    - Matrix multiplication with broadcasting
    - Element-wise operations
    - Reduction operations (sum, mean, etc.)
    - Softmax and argmax
    - Chunking and concatenation
    - Repeat operations
    """

    def __init__(self, data: Union[List, float, int], dtype: Optional[str] = None):
        """
        Initialize tensor from nested list or scalar.

        Args:
            data: Nested list representing tensor data, or scalar
            dtype: Data type ('float' or 'int'), inferred if None
        """
        if isinstance(data, (int, float)):
            self.data = data
            self.shape = ()
            self.ndim = 0
        else:
            self.data = self._validate_and_copy_data(data)
            self.shape = self._compute_shape(self.data)
            self.ndim = len(self.shape)

        self.dtype = dtype or self._infer_dtype()

    def _validate_and_copy_data(self, data: List) -> List:
        """Validate nested list structure and create deep copy."""
        if not isinstance(data, list):
            raise ValueError("Data must be a nested list or scalar")

        def validate_recursive(lst: List, depth: int = 0) -> List:
            if not lst:
                return []

            first_elem = lst[0]
            if isinstance(first_elem, list):
                # Check all elements are lists of same length
                expected_len = len(first_elem)
                for i, elem in enumerate(lst):
                    if not isinstance(elem, list):
                        raise ValueError(f"All elements must be lists at depth {depth}, got {type(elem)} at index {i}")
                    if len(elem) != expected_len:
                        raise ValueError(f"All sublists must have same length {expected_len}, got {len(elem)} at index {i}")
                return [validate_recursive(sublist, depth + 1) for sublist in lst]
            else:
                # Check all elements are numbers
                for i, elem in enumerate(lst):
                    if not isinstance(elem, (int, float)):
                        raise ValueError(f"All elements must be numbers, got {type(elem)} at index {i}")
                return lst.copy()

        return validate_recursive(data)

    def _compute_shape(self, data: Union[List, float, int]) -> Tuple[int, ...]:
        """Compute shape of nested list."""
        if isinstance(data, (int, float)):
            return ()

        if not data:
            return (0,)

        shape = [len(data)]
        current = data[0]

        while isinstance(current, list):
            shape.append(len(current))
            current = current[0] if current else []

        return tuple(shape)

    def _infer_dtype(self) -> str:
        """Infer data type from tensor data."""
        def has_float(data):
            if isinstance(data, (int, float)):
                return isinstance(data, float)
            elif isinstance(data, list):
                return any(has_float(item) for item in data)
            return False

        return 'float' if has_float(self.data) else 'int'

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, data={self.data})"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, key):
        """Basic indexing support."""
        if self.ndim == 0:
            raise IndexError("Scalar tensor cannot be indexed")

        if isinstance(key, int):
            if self.ndim == 1:
                return self.data[key]
            else:
                result = self.data[key]
                new_shape = self.shape[1:]
                return Tensor(result) if new_shape else result
        elif isinstance(key, tuple):
            # Handle multi-dimensional indexing
            current_data = self.data
            remaining_shape = list(self.shape)

            for i, idx in enumerate(key):
                if isinstance(idx, int):
                    current_data = current_data[idx]
                    remaining_shape.pop(0)
                else:
                    raise NotImplementedError("Slice indexing not yet implemented")

            if remaining_shape:
                return Tensor(current_data)
            else:
                return current_data

        raise NotImplementedError("Advanced indexing not yet implemented")

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise addition with broadcasting."""
        return self._element_wise_op(other, lambda a, b: a + b)

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise subtraction with broadcasting."""
        return self._element_wise_op(other, lambda a, b: a - b)

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise multiplication with broadcasting."""
        return self._element_wise_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise division with broadcasting."""
        return self._element_wise_op(other, lambda a, b: a / b)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        return self.matmul(other)

    def _element_wise_op(self, other: Union['Tensor', float, int], op) -> 'Tensor':
        """Apply element-wise operation with broadcasting."""
        if isinstance(other, (int, float)):
            return Tensor(self._apply_op_recursive(self.data, other, op), self.dtype)

        # Broadcasting logic
        self_shape, other_shape = self._broadcast_shapes(self.shape, other.shape)
        self_data = self._broadcast_data(self.data, self_shape)
        other_data = other._broadcast_data(other.data, other_shape)

        result_data = self._element_wise_recursive(self_data, other_data, op)
        return Tensor(result_data, self.dtype)

    def _broadcast_shapes(self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Compute broadcasted shapes."""
        # Simple broadcasting: pad shorter shape with 1s on the left
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        padded1 = (1,) * (max_len - len1) + shape1
        padded2 = (1,) * (max_len - len2) + shape2

        result_shape = []
        for s1, s2 in zip(padded1, padded2):
            if s1 == 1 or s2 == 1 or s1 == s2:
                result_shape.append(max(s1, s2))
            else:
                raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")

        return tuple(result_shape), tuple(result_shape)

    def _broadcast_data(self, data: Union[List, float, int], target_shape: Tuple[int, ...]) -> Union[List, float, int]:
        """Broadcast data to target shape."""
        if isinstance(data, (int, float)):
            # Scalar broadcasting
            return self._create_broadcasted_data(target_shape, data)

        current_shape = self._compute_shape(data)
        if current_shape == target_shape:
            return data

        # For now, implement simple broadcasting
        # This is a simplified version - full broadcasting would be more complex
        return data  # Placeholder

    def _create_broadcasted_data(self, shape: Tuple[int, ...], value: Union[int, float]) -> List:
        """Create nested list filled with value in given shape."""
        if not shape:
            return value

        return [self._create_broadcasted_data(shape[1:], value) for _ in range(shape[0])]

    def _element_wise_recursive(self, data1: Union[List, float, int], data2: Union[List, float, int], op) -> Union[List, float, int]:
        """Apply operation element-wise recursively."""
        if isinstance(data1, (int, float)) and isinstance(data2, (int, float)):
            return op(data1, data2)
        elif isinstance(data1, list) and isinstance(data2, list):
            return [self._element_wise_recursive(a, b, op) for a, b in zip(data1, data2)]
        else:
            raise ValueError("Incompatible data structures for element-wise operation")

    def _apply_op_recursive(self, data: Union[List, float, int], scalar: Union[int, float], op) -> Union[List, float, int]:
        """Apply operation with scalar recursively."""
        if isinstance(data, (int, float)):
            return op(data, scalar)
        elif isinstance(data, list):
            return [self._apply_op_recursive(item, scalar, op) for item in data]
        else:
            raise ValueError("Invalid data type")

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("Matrix multiplication requires at least 2D tensors")

        # For simplicity, assume last two dimensions are the matrix dimensions
        if self.shape[-1] != other.shape[-2]:
            raise ValueError(f"Incompatible shapes for matmul: {self.shape} and {other.shape}")

        result_shape = self.shape[:-2] + (self.shape[-2], other.shape[-1])

        def matmul_2d(a: List[List], b: List[List]) -> List[List]:
            """2D matrix multiplication."""
            m, k = len(a), len(a[0])
            k2, n = len(b), len(b[0])

            result = [[0 for _ in range(n)] for _ in range(m)]
            for i in range(m):
                for j in range(n):
                    for p in range(k):
                        result[i][j] += a[i][p] * b[p][j]
            return result

        # Handle higher dimensions by iterating over batch dimensions
        def recursive_matmul(data1, data2, shape1, shape2):
            if len(shape1) == 2 and len(shape2) == 2:
                return matmul_2d(data1, data2)

            # Broadcast batch dimensions if needed
            batch_shape = self._broadcast_shapes(shape1[:-2], shape2[:-2])[0]

            # This is simplified - full implementation would handle broadcasting properly
            return matmul_2d(data1, data2)

        result_data = recursive_matmul(self.data, other.data, self.shape, other.shape)
        return Tensor(result_data, self.dtype)

    def transpose(self, *dims: int) -> 'Tensor':
        """Transpose tensor along given dimensions."""
        if not dims:
            dims = tuple(reversed(range(self.ndim)))

        if len(dims) != self.ndim:
            raise ValueError(f"Number of dimensions {len(dims)} doesn't match tensor ndim {self.ndim}")

        # Create permutation mapping
        perm = list(dims)

        # Compute new shape
        new_shape = tuple(self.shape[i] for i in perm)

        # Transpose data
        def transpose_recursive(data: List, axes: List[int], depth: int = 0) -> List:
            if depth == len(axes):
                return data

            axis = axes[depth]
            if axis >= len(data[0]) if isinstance(data[0], list) else 0:
                raise ValueError("Invalid axis for transpose")

            # This is a simplified transpose - full implementation would be more complex
            # For now, just return the data (placeholder)
            return data

        new_data = transpose_recursive(self.data, perm)
        return Tensor(new_data, self.dtype)

    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor to new shape."""
        new_shape = shape if len(shape) > 1 else shape[0] if shape else ()

        if isinstance(new_shape, int):
            new_shape = (new_shape,)

        total_elements = 1
        for dim in self.shape:
            total_elements *= dim

        new_total = 1
        inferred_dim = -1
        for i, dim in enumerate(new_shape):
            if dim == -1:
                if inferred_dim != -1:
                    raise ValueError("Can only infer one dimension")
                inferred_dim = i
            else:
                new_total *= dim

        if inferred_dim != -1:
            if total_elements % new_total != 0:
                raise ValueError("Cannot infer dimension: total elements not divisible")
            new_shape = list(new_shape)
            new_shape[inferred_dim] = total_elements // new_total
            new_shape = tuple(new_shape)

        new_total = 1
        for dim in new_shape:
            new_total *= dim

        if new_total != total_elements:
            raise ValueError(f"Cannot reshape {self.shape} to {new_shape}")

        # For simplicity, just update shape and keep data structure
        # In a real implementation, this would flatten and rebuild the nested structure
        result = Tensor(self.data, self.dtype)
        result.shape = new_shape
        result.ndim = len(new_shape)
        return result

    def view(self, *shape: int) -> 'Tensor':
        """View tensor with different shape (same as reshape for now)."""
        return self.reshape(*shape)

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """Sum tensor along dimension."""
        if dim is None:
            # Sum all elements
            total = self._sum_all(self.data)
            return Tensor(total, self.dtype)

        if dim < 0:
            dim = self.ndim + dim

        if dim >= self.ndim:
            raise ValueError(f"Dimension {dim} out of range")

        result_data = self._sum_along_dim(self.data, dim, self.shape)
        new_shape = list(self.shape)
        if not keepdim:
            new_shape.pop(dim)
        else:
            new_shape[dim] = 1

        return Tensor(result_data, self.dtype)

    def _sum_all(self, data: Union[List, float, int]) -> Union[float, int]:
        """Sum all elements recursively."""
        if isinstance(data, (int, float)):
            return data
        return sum(self._sum_all(item) for item in data)

    def _sum_along_dim(self, data: List, dim: int, shape: Tuple[int, ...]) -> List:
        """Sum along specific dimension."""
        if dim == 0:
            # Sum along first dimension
            result = []
            for i in range(shape[1] if len(shape) > 1 else len(data[0])):
                if len(shape) == 1:
                    result.append(sum(data))
                else:
                    col_sum = 0
                    for row in data:
                        col_sum += row[i]
                    result.append(col_sum)
            return result
        else:
            # Recurse on subdimensions
            return [self._sum_along_dim(sublist, dim - 1, shape[1:]) for sublist in data]

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """Mean tensor along dimension."""
        sum_tensor = self.sum(dim, keepdim)
        count = 1
        if dim is None:
            for d in self.shape:
                count *= d
        else:
            count = self.shape[dim]

        return sum_tensor / count

    def softmax(self, dim: int = -1) -> 'Tensor':
        """Apply softmax along dimension."""
        if dim < 0:
            dim = self.ndim + dim

        # exp(x - max(x)) for numerical stability
        max_vals = self.max(dim, keepdim=True)
        exp_tensor = (self - max_vals).exp()
        sum_exp = exp_tensor.sum(dim, keepdim=True)
        return exp_tensor / sum_exp

    def argmax(self, dim: int = -1) -> 'Tensor':
        """Argmax along dimension."""
        if dim < 0:
            dim = self.ndim + dim

        def argmax_recursive(data: List, target_dim: int, current_dim: int = 0) -> List:
            if current_dim == target_dim:
                if isinstance(data[0], list):
                    return [argmax_recursive(sublist, target_dim, current_dim + 1) for sublist in data]
                else:
                    max_val = max(data)
                    return data.index(max_val)
            elif isinstance(data[0], list):
                return [argmax_recursive(sublist, target_dim, current_dim + 1) for sublist in data]
            else:
                return data

        result_data = argmax_recursive(self.data, dim)
        new_shape = list(self.shape)
        new_shape.pop(dim)
        return Tensor(result_data, 'int')

    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """Max along dimension or overall."""
        if dim is None:
            max_val = self._max_all(self.data)
            return Tensor(max_val, self.dtype)

        if dim < 0:
            dim = self.ndim + dim

        result_data = self._max_along_dim(self.data, dim, self.shape)
        new_shape = list(self.shape)
        if not keepdim:
            new_shape.pop(dim)
        else:
            new_shape[dim] = 1

        return Tensor(result_data, self.dtype)

    def _max_all(self, data: Union[List, float, int]) -> Union[float, int]:
        """Find max of all elements."""
        if isinstance(data, (int, float)):
            return data
        return max(self._max_all(item) for item in data)

    def _max_along_dim(self, data: List, dim: int, shape: Tuple[int, ...]) -> List:
        """Max along specific dimension."""
        if dim == 0:
            result = []
            for i in range(shape[1] if len(shape) > 1 else len(data[0])):
                if len(shape) == 1:
                    result.append(max(data))
                else:
                    col_max = max(row[i] for row in data)
                    result.append(col_max)
            return result
        else:
            return [self._max_along_dim(sublist, dim - 1, shape[1:]) for sublist in data]

    def exp(self) -> 'Tensor':
        """Element-wise exponential."""
        return Tensor(self._apply_func_recursive(self.data, math.exp), self.dtype)

    def log(self) -> 'Tensor':
        """Element-wise natural log."""
        return Tensor(self._apply_func_recursive(self.data, math.log), self.dtype)

    def sqrt(self) -> 'Tensor':
        """Element-wise square root."""
        return Tensor(self._apply_func_recursive(self.data, math.sqrt), self.dtype)

    def _apply_func_recursive(self, data: Union[List, float, int], func) -> Union[List, float, int]:
        """Apply function recursively."""
        if isinstance(data, (int, float)):
            return func(data)
        return [self._apply_func_recursive(item, func) for item in data]

    def chunk(self, chunks: int, dim: int = 0) -> List['Tensor']:
        """Split tensor into chunks along dimension."""
        if dim < 0:
            dim = self.ndim + dim

        if dim >= self.ndim:
            raise ValueError(f"Dimension {dim} out of range")

        size = self.shape[dim] // chunks
        if self.shape[dim] % chunks != 0:
            raise ValueError(f"Dimension {dim} size {self.shape[dim]} not divisible by {chunks}")

        result = []
        for i in range(chunks):
            start_idx = i * size
            end_idx = (i + 1) * size

            # This is simplified - would need proper slicing implementation
            chunk_data = self.data[start_idx:end_idx] if dim == 0 else self.data
            new_shape = list(self.shape)
            new_shape[dim] = size
            chunk = Tensor(chunk_data, self.dtype)
            chunk.shape = tuple(new_shape)
            chunk.ndim = len(new_shape)
            result.append(chunk)

        return result

    def cat(self, tensors: List['Tensor'], dim: int = 0) -> 'Tensor':
        """Concatenate tensors along dimension."""
        if not tensors:
            return self

        # Check compatibility
        for tensor in tensors:
            if tensor.dtype != self.dtype:
                raise ValueError("All tensors must have same dtype")
            if len(tensor.shape) != self.ndim:
                raise ValueError("All tensors must have same number of dimensions")

        # This is simplified - would need proper concatenation logic
        all_data = [self.data] + [t.data for t in tensors]
        result_data = []
        for data in all_data:
            result_data.extend(data)

        new_shape = list(self.shape)
        new_shape[dim] = sum(t.shape[dim] for t in [self] + tensors)

        result = Tensor(result_data, self.dtype)
        result.shape = tuple(new_shape)
        result.ndim = len(new_shape)
        return result

    def repeat(self, *repeats: int) -> 'Tensor':
        """Repeat tensor along dimensions."""
        if len(repeats) != self.ndim:
            raise ValueError(f"Number of repeats {len(repeats)} doesn't match ndim {self.ndim}")

        # This is simplified - would need proper repeat logic
        result_data = self.data
        new_shape = tuple(s * r for s, r in zip(self.shape, repeats))

        result = Tensor(result_data, self.dtype)
        result.shape = new_shape
        result.ndim = len(new_shape)
        return result

    @classmethod
    def zeros(cls, *shape: int, dtype: str = 'float') -> 'Tensor':
        """Create tensor filled with zeros."""
        if len(shape) == 1:
            shape = shape[0]
        return cls._create_filled_tensor(shape, 0, dtype)

    @classmethod
    def ones(cls, *shape: int, dtype: str = 'float') -> 'Tensor':
        """Create tensor filled with ones."""
        if len(shape) == 1:
            shape = shape[0]
        return cls._create_filled_tensor(shape, 1, dtype)

    @classmethod
    def _create_filled_tensor(cls, shape: Union[int, Tuple[int, ...]], value: Union[int, float], dtype: str) -> 'Tensor':
        """Create tensor filled with value."""
        if isinstance(shape, int):
            shape = (shape,)

        def create_nested_list(shape_tuple: Tuple[int, ...]) -> List:
            if len(shape_tuple) == 1:
                return [value] * shape_tuple[0]
            return [create_nested_list(shape_tuple[1:]) for _ in range(shape_tuple[0])]

        data = create_nested_list(shape)
        tensor = cls(data, dtype)
        tensor.shape = shape
        tensor.ndim = len(shape)
        return tensor

    @classmethod
    def arange(cls, start: int, end: int, step: int = 1, dtype: str = 'int') -> 'Tensor':
        """Create 1D tensor with range of values."""
        data = list(range(start, end, step))
        return cls(data, dtype)

    def tolist(self) -> List:
        """Convert tensor to nested list."""
        return copy.deepcopy(self.data)

    def item(self) -> Union[int, float]:
        """Get scalar value from 0-dim tensor."""
        if self.ndim != 0:
            raise ValueError("Can only get item from scalar tensor")
        return self.data

    # Advanced Tensor Operations (Todo 1)

    def einsum(self, equation: str, *operands: 'Tensor') -> 'Tensor':
        """
        Einstein summation convention for tensor operations.

        Args:
            equation: String specifying the operation (e.g., 'ij,jk->ik' for matrix multiplication)
            *operands: Additional tensors to operate on

        Returns:
            Result tensor
        """
        all_tensors = [self] + list(operands)

        # Parse equation
        input_output = equation.split('->')
        if len(input_output) != 2:
            raise ValueError("Equation must contain exactly one '->'")

        input_spec = input_output[0].split(',')
        output_spec = input_output[1]

        if len(input_spec) != len(all_tensors):
            raise ValueError(f"Number of input specs {len(input_spec)} doesn't match number of tensors {len(all_tensors)}")

        # This is a simplified implementation - full einsum would be much more complex
        # For now, handle basic cases like matrix multiplication
        if len(all_tensors) == 2 and equation in ['ij,jk->ik', 'ij,j->i', 'i,j->ij']:
            return self._simple_einsum(equation, all_tensors[1])
        else:
            raise NotImplementedError(f"Advanced einsum pattern '{equation}' not yet implemented")

    def _simple_einsum(self, equation: str, other: 'Tensor') -> 'Tensor':
        """Simple einsum for basic operations."""
        if equation == 'ij,jk->ik':
            return self.matmul(other)
        elif equation == 'ij,j->i':
            # Matrix-vector multiplication
            return self.matmul(other.unsqueeze(-1)).squeeze(-1)
        elif equation == 'i,j->ij':
            # Outer product
            result = []
            for i in range(self.shape[0]):
                row = []
                for j in range(other.shape[0]):
                    row.append(self.data[i] * other.data[j])
                result.append(row)
            return Tensor(result, self.dtype)
        else:
            raise NotImplementedError(f"Simple einsum pattern '{equation}' not implemented")

    def __setitem__(self, key, value: Union['Tensor', float, int]):
        """Advanced indexing and assignment."""
        if isinstance(key, tuple):
            # Multi-dimensional indexing
            current_data = self.data
            indices = []

            for i, idx in enumerate(key):
                if isinstance(idx, int):
                    indices.append(idx)
                    if i < len(key) - 1:
                        current_data = current_data[idx]
                elif isinstance(idx, slice):
                    # Handle slice indexing
                    start, stop, step = idx.indices(self.shape[i])
                    indices.append(slice(start, stop, step))
                else:
                    raise NotImplementedError("Advanced indexing not fully implemented")

            # Set value at location
            self._set_value_at_indices(self.data, indices, value, 0)
        else:
            # Simple 1D indexing
            if isinstance(key, int):
                if self.ndim == 1:
                    self.data[key] = value
                else:
                    raise NotImplementedError("Simple multi-dimensional assignment not implemented")
            else:
                raise NotImplementedError("Slice assignment not implemented")

    def _set_value_at_indices(self, data: List, indices: List, value: Union['Tensor', float, int], depth: int):
        """Recursively set value at indices."""
        if depth == len(indices) - 1:
            idx = indices[depth]
            if isinstance(idx, int):
                data[idx] = value
            else:
                # Handle slice
                start, stop, step = idx.indices(len(data))
                for i in range(start, stop, step):
                    data[i] = value
        else:
            idx = indices[depth]
            if isinstance(idx, int):
                self._set_value_at_indices(data[idx], indices, value, depth + 1)
            else:
                # Handle slice
                start, stop, step = idx.indices(len(data))
                for i in range(start, stop, step):
                    self._set_value_at_indices(data[i], indices, value, depth + 1)

    def gather(self, dim: int, index: 'Tensor') -> 'Tensor':
        """
        Gather elements from tensor along dimension using indices.

        Args:
            dim: Dimension to gather along
            index: Tensor containing indices

        Returns:
            Gathered tensor
        """
        if dim < 0:
            dim = self.ndim + dim

        if index.shape != self.shape:
            # Allow broadcasting for some cases
            if len(index.shape) != self.ndim:
                raise ValueError("Index tensor must have same number of dimensions")

        def gather_recursive(data: List, idx_data: List, target_dim: int, current_dim: int = 0) -> List:
            if current_dim == target_dim:
                result = []
                for idx_row in idx_data:
                    if isinstance(idx_row, list):
                        result.append([data[i] for i in idx_row])
                    else:
                        result.append(data[idx_row])
                return result
            else:
                return [gather_recursive(sublist, idx_sublist, target_dim, current_dim + 1)
                       for sublist, idx_sublist in zip(data, idx_data)]

        result_data = gather_recursive(self.data, index.data, dim)
        new_shape = list(self.shape)
        # Shape remains the same for gather
        return Tensor(result_data, self.dtype)

    def scatter(self, dim: int, index: 'Tensor', src: 'Tensor') -> 'Tensor':
        """
        Scatter elements into tensor along dimension.

        Args:
            dim: Dimension to scatter along
            index: Tensor containing indices
            src: Source tensor to scatter

        Returns:
            Tensor with scattered values
        """
        result = self.clone()

        if dim < 0:
            dim = self.ndim + dim

        def scatter_recursive(data: List, idx_data: List, src_data: List, target_dim: int, current_dim: int = 0):
            if current_dim == target_dim:
                for i, idx in enumerate(idx_data):
                    if isinstance(idx, list):
                        for j, val in enumerate(idx):
                            data[val] = src_data[i][j]
                    else:
                        data[idx] = src_data[i]
            else:
                for i, (sublist, idx_sublist, src_sublist) in enumerate(zip(data, idx_data, src_data)):
                    scatter_recursive(sublist, idx_sublist, src_sublist, target_dim, current_dim + 1)

        scatter_recursive(result.data, index.data, src.data, dim)
        return result

    def clone(self) -> 'Tensor':
        """Create a deep copy of the tensor."""
        return Tensor(copy.deepcopy(self.data), self.dtype)

    def unsqueeze(self, dim: int) -> 'Tensor':
        """Add dimension of size 1 at specified position."""
        if dim < 0:
            dim = self.ndim + 1 + dim

        new_shape = list(self.shape)
        new_shape.insert(dim, 1)

        # Add dimension to data structure
        def unsqueeze_recursive(data: Union[List, float, int], target_dim: int, current_dim: int = 0) -> List:
            if current_dim == target_dim:
                return [data]
            elif isinstance(data, list):
                return [unsqueeze_recursive(item, target_dim, current_dim + 1) for item in data]
            else:
                return [data]

        new_data = unsqueeze_recursive(self.data, dim)
        result = Tensor(new_data, self.dtype)
        result.shape = tuple(new_shape)
        result.ndim = len(new_shape)
        return result

    def squeeze(self, dim: Optional[int] = None) -> 'Tensor':
        """Remove dimensions of size 1."""
        if dim is not None:
            if dim < 0:
                dim = self.ndim + dim
            if self.shape[dim] != 1:
                raise ValueError(f"Cannot squeeze dimension {dim} of size {self.shape[dim]}")
            new_shape = list(self.shape)
            new_shape.pop(dim)
        else:
            new_shape = [s for s in self.shape if s != 1]

        if len(new_shape) == len(self.shape):
            return self.clone()  # No dimensions to squeeze

        # Remove size-1 dimensions from data
        def squeeze_recursive(data: List, shape: List[int], current_dim: int = 0) -> Union[List, float, int]:
            if current_dim >= len(shape):
                return data

            if shape[current_dim] == 1:
                # Skip this dimension
                return squeeze_recursive(data[0], shape, current_dim + 1)
            else:
                return [squeeze_recursive(item, shape, current_dim + 1) for item in data]

        new_data = squeeze_recursive(self.data, list(self.shape))
        result = Tensor(new_data, self.dtype)
        result.shape = tuple(new_shape)
        result.ndim = len(new_shape)
        return result

    def expand(self, *sizes: int) -> 'Tensor':
        """Expand tensor to larger size by repeating along singleton dimensions."""
        if len(sizes) != self.ndim:
            raise ValueError(f"Number of sizes {len(sizes)} doesn't match ndim {self.ndim}")

        new_shape = tuple(sizes)

        # Check compatibility
        for i, (old_size, new_size) in enumerate(zip(self.shape, new_shape)):
            if old_size != 1 and old_size != new_size:
                raise ValueError(f"Cannot expand dimension {i} from {old_size} to {new_size}")

        # Repeat along dimensions that changed from 1
        result_data = self.data
        for i, (old_size, new_size) in enumerate(zip(self.shape, new_shape)):
            if old_size == 1 and new_size > 1:
                # Repeat along this dimension
                result_data = self._repeat_along_dim(result_data, i, new_size)

        result = Tensor(result_data, self.dtype)
        result.shape = new_shape
        result.ndim = len(new_shape)
        return result

    def _repeat_along_dim(self, data: List, dim: int, repeats: int) -> List:
        """Repeat data along specific dimension."""
        if dim == 0:
            return data * repeats
        else:
            return [self._repeat_along_dim(sublist, dim - 1, repeats) for sublist in data]

    def permute(self, *dims: int) -> 'Tensor':
        """Permute tensor dimensions."""
        if len(dims) != self.ndim:
            raise ValueError(f"Number of dims {len(dims)} doesn't match ndim {self.ndim}")

        if len(set(dims)) != len(dims):
            raise ValueError("Repeated dimensions not allowed")

        new_shape = tuple(self.shape[i] for i in dims)

        # This is a simplified permute - full implementation would be complex
        # For now, just update shape (data structure remains the same)
        result = self.clone()
        result.shape = new_shape
        result.ndim = len(new_shape)
        return result

    def flip(self, dims: List[int]) -> 'Tensor':
        """Flip tensor along specified dimensions."""
        result = self.clone()

        for dim in dims:
            if dim < 0:
                dim = self.ndim + dim

            def flip_recursive(data: List, target_dim: int, current_dim: int = 0) -> List:
                if current_dim == target_dim:
                    return list(reversed(data))
                else:
                    return [flip_recursive(sublist, target_dim, current_dim + 1) for sublist in data]

            result.data = flip_recursive(result.data, dim)

        return result

    def roll(self, shifts: Union[int, List[int]], dims: Union[int, List[int]]) -> 'Tensor':
        """Roll tensor along specified dimensions."""
        if isinstance(shifts, int):
            shifts = [shifts]
        if isinstance(dims, int):
            dims = [dims]

        if len(shifts) != len(dims):
            raise ValueError("shifts and dims must have same length")

        result = self.clone()

        for shift, dim in zip(shifts, dims):
            if dim < 0:
                dim = self.ndim + dim

            def roll_recursive(data: List, shift_val: int, target_dim: int, current_dim: int = 0) -> List:
                if current_dim == target_dim:
                    return data[-shift_val:] + data[:-shift_val]
                else:
                    return [roll_recursive(sublist, shift_val, target_dim, current_dim + 1) for sublist in data]

            result.data = roll_recursive(result.data, shift, dim)

        return result