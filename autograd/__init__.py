"""
Aisupea Automatic Differentiation Module

Gradient computation and backpropagation system.
"""

from typing import Dict, List, Optional, Callable, Any
from ..core import Tensor
import math


class Variable:
    """A tensor with gradient tracking for automatic differentiation."""

    def __init__(self, data: Tensor, requires_grad: bool = False):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.grad_fn = None  # Function that created this variable
        self._backward_hooks = []

    def __repr__(self):
        return f"Variable(data={self.data}, requires_grad={self.requires_grad})"

    def backward(self, gradient: Optional['Variable'] = None):
        """Compute gradients through backpropagation."""
        if gradient is None:
            if self.data.ndim != 0:
                raise ValueError("Gradient must be specified for non-scalar outputs")
            gradient = Variable(Tensor(1.0), requires_grad=False)

        # Topological sort of computation graph
        topo_order = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                if node.grad_fn:
                    for parent in node.grad_fn.parents:
                        build_topo(parent)
                topo_order.append(node)

        build_topo(self)

        # Initialize gradients
        self.grad = gradient

        # Backward pass
        for node in reversed(topo_order):
            if node.grad_fn and node.grad is not None:
                grads = node.grad_fn.backward(node.grad)
                for parent, grad in zip(node.grad_fn.parents, grads):
                    if parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = grad
                        else:
                            parent.grad = parent.grad + grad

    def detach(self) -> 'Variable':
        """Detach from computation graph."""
        return Variable(self.data, requires_grad=False)

    def zero_grad(self):
        """Zero out gradients."""
        self.grad = None

    # Arithmetic operations with gradient tracking
    def __add__(self, other: 'Variable') -> 'Variable':
        if isinstance(other, (int, float)):
            other = Variable(Tensor(other), requires_grad=False)
        return Add.apply(self, other)

    def __sub__(self, other: 'Variable') -> 'Variable':
        if isinstance(other, (int, float)):
            other = Variable(Tensor(other), requires_grad=False)
        return Sub.apply(self, other)

    def __mul__(self, other: 'Variable') -> 'Variable':
        if isinstance(other, (int, float)):
            other = Variable(Tensor(other), requires_grad=False)
        return Mul.apply(self, other)

    def __truediv__(self, other: 'Variable') -> 'Variable':
        if isinstance(other, (int, float)):
            other = Variable(Tensor(other), requires_grad=False)
        return Div.apply(self, other)

    def __matmul__(self, other: 'Variable') -> 'Variable':
        return MatMul.apply(self, other)

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Variable':
        return Sum.apply(self, dim, keepdim)

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Variable':
        return Mean.apply(self, dim, keepdim)

    def exp(self) -> 'Variable':
        return Exp.apply(self)

    def log(self) -> 'Variable':
        return Log.apply(self)

    def sqrt(self) -> 'Variable':
        return Sqrt.apply(self)

    def relu(self) -> 'Variable':
        return ReLU.apply(self)

    def sigmoid(self) -> 'Variable':
        return Sigmoid.apply(self)

    def tanh(self) -> 'Variable':
        return Tanh.apply(self)


class Function:
    """Base class for differentiable operations."""

    @staticmethod
    def forward(*args, **kwargs) -> Tensor:
        """Forward pass implementation."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs) -> Variable:
        """Apply function and create Variable with gradient tracking."""
        # Create output
        output_data = cls.forward(*args, **kwargs)
        output = Variable(output_data, requires_grad=any(arg.requires_grad for arg in args if hasattr(arg, 'requires_grad')))

        if output.requires_grad:
            output.grad_fn = cls(*args, **kwargs)
            output.grad_fn.parents = [arg for arg in args if hasattr(arg, 'requires_grad')]

        return output

    def backward(self, grad_output: Variable) -> List[Variable]:
        """Backward pass implementation."""
        raise NotImplementedError


class Add(Function):
    @staticmethod
    def forward(a: Variable, b: Variable) -> Tensor:
        return a.data + b.data

    def backward(self, grad_output: Variable) -> List[Variable]:
        # Gradient flows equally to both inputs
        return [grad_output, grad_output]


class Sub(Function):
    @staticmethod
    def forward(a: Variable, b: Variable) -> Tensor:
        return a.data - b.data

    def backward(self, grad_output: Variable) -> List[Variable]:
        return [grad_output, Variable(-grad_output.data)]


class Mul(Function):
    @staticmethod
    def forward(a: Variable, b: Variable) -> Tensor:
        return a.data * b.data

    def backward(self, grad_output: Variable) -> List[Variable]:
        a, b = self.parents
        return [grad_output * b, grad_output * a]


class Div(Function):
    @staticmethod
    def forward(a: Variable, b: Variable) -> Tensor:
        return a.data / b.data

    def backward(self, grad_output: Variable) -> List[Variable]:
        a, b = self.parents
        return [
            grad_output / b,
            Variable(-grad_output.data * a.data / (b.data * b.data))
        ]


class MatMul(Function):
    @staticmethod
    def forward(a: Variable, b: Variable) -> Tensor:
        return a.data @ b.data

    def backward(self, grad_output: Variable) -> List[Variable]:
        a, b = self.parents
        return [
            grad_output @ Variable(b.data.transpose()),
            Variable(a.data.transpose()) @ grad_output
        ]


class Sum(Function):
    def __init__(self, input_var, dim=None, keepdim=False):
        self.dim = dim
        self.keepdim = keepdim

    @staticmethod
    def forward(input_var: Variable, dim=None, keepdim=False) -> Tensor:
        return input_var.data.sum(dim, keepdim)

    def backward(self, grad_output: Variable) -> List[Variable]:
        input_var = self.parents[0]
        # Broadcast gradient back to input shape
        if self.dim is None:
            # Sum all elements
            grad_input = Variable(Tensor.ones(*input_var.data.shape) * grad_output.data.item())
        else:
            # Sum along dimension - broadcast back
            grad_input_data = grad_output.data
            if not self.keepdim:
                grad_input_data = grad_input_data.unsqueeze(self.dim)
            # Repeat along summed dimension
            repeats = [1] * input_var.data.ndim
            repeats[self.dim] = input_var.data.shape[self.dim]
            grad_input = Variable(grad_input_data.repeat(*repeats))
        return [grad_input]


class Mean(Function):
    def __init__(self, input_var, dim=None, keepdim=False):
        self.dim = dim
        self.keepdim = keepdim

    @staticmethod
    def forward(input_var: Variable, dim=None, keepdim=False) -> Tensor:
        return input_var.data.mean(dim, keepdim)

    def backward(self, grad_output: Variable) -> List[Variable]:
        input_var = self.parents[0]
        if self.dim is None:
            # Mean of all elements
            count = 1
            for d in input_var.data.shape:
                count *= d
            grad_input = Variable(Tensor.ones(*input_var.data.shape) * (grad_output.data.item() / count))
        else:
            # Mean along dimension
            count = input_var.data.shape[self.dim]
            grad_input_data = grad_output.data
            if not self.keepdim:
                grad_input_data = grad_input_data.unsqueeze(self.dim)
            repeats = [1] * input_var.data.ndim
            repeats[self.dim] = input_var.data.shape[self.dim]
            grad_input = Variable(grad_input_data.repeat(*repeats) / count)
        return [grad_input]


class Exp(Function):
    @staticmethod
    def forward(input_var: Variable) -> Tensor:
        return input_var.data.exp()

    def backward(self, grad_output: Variable) -> List[Variable]:
        input_var = self.parents[0]
        return [grad_output * Variable(input_var.data.exp())]


class Log(Function):
    @staticmethod
    def forward(input_var: Variable) -> Tensor:
        return input_var.data.log()

    def backward(self, grad_output: Variable) -> List[Variable]:
        input_var = self.parents[0]
        return [grad_output / input_var]


class Sqrt(Function):
    @staticmethod
    def forward(input_var: Variable) -> Tensor:
        return input_var.data.sqrt()

    def backward(self, grad_output: Variable) -> List[Variable]:
        input_var = self.parents[0]
        return [grad_output / (Variable(input_var.data.sqrt()) * 2)]


class ReLU(Function):
    @staticmethod
    def forward(input_var: Variable) -> Tensor:
        # Simple ReLU implementation
        def relu_op(x):
            return max(0, x)
        return Tensor(input_var.data._apply_func_recursive(input_var.data.data, relu_op), input_var.data.dtype)

    def backward(self, grad_output: Variable) -> List[Variable]:
        input_var = self.parents[0]
        # Gradient is 1 where input > 0, 0 otherwise
        grad_mask = Variable((input_var > 0).data, requires_grad=False)
        return [grad_output * grad_mask]


class Sigmoid(Function):
    @staticmethod
    def forward(input_var: Variable) -> Tensor:
        def sigmoid_op(x):
            return 1 / (1 + math.exp(-x))
        return Tensor(input_var.data._apply_func_recursive(input_var.data.data, sigmoid_op), input_var.data.dtype)

    def backward(self, grad_output: Variable) -> List[Variable]:
        input_var = self.parents[0]
        sigmoid_output = Variable(self.forward(input_var), requires_grad=False)
        return [grad_output * sigmoid_output * (Variable(Tensor(1.0)) - sigmoid_output)]


class Tanh(Function):
    @staticmethod
    def forward(input_var: Variable) -> Tensor:
        def tanh_op(x):
            return math.tanh(x)
        return Tensor(input_var.data._apply_func_recursive(input_var.data.data, tanh_op), input_var.data.dtype)

    def backward(self, grad_output: Variable) -> List[Variable]:
        input_var = self.parents[0]
        tanh_output = Variable(self.forward(input_var), requires_grad=False)
        return [grad_output * (Variable(Tensor(1.0)) - tanh_output * tanh_output)]


def no_grad():
    """Context manager to disable gradient computation."""
    class NoGradContext:
        def __enter__(self):
            self.old_requires_grad = []
            # In a full implementation, this would track and disable gradients globally

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore gradient requirements
            pass

    return NoGradContext()


__all__ = [
    'Variable',
    'Function',
    'Add', 'Sub', 'Mul', 'Div', 'MatMul',
    'Sum', 'Mean', 'Exp', 'Log', 'Sqrt',
    'ReLU', 'Sigmoid', 'Tanh',
    'no_grad'
]