"""
Aisupea Example: Basic Tensor Operations

Demonstrates basic tensor creation and operations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Tensor
from torch_api import zeros, ones, arange, matmul, softmax
import math


def main():
    print("🤖 Aisupea Tensor Operations Example")
    print("=" * 50)

    # Create tensors
    print("\n1. Creating tensors:")
    a = Tensor([[1, 2, 3], [4, 5, 6]], 'float')
    b = Tensor([[7, 8], [9, 10], [11, 12]], 'float')

    print(f"a (shape {a.shape}): {a.data}")
    print(f"b (shape {b.shape}): {b.data}")

    # Matrix multiplication
    print("\n2. Matrix multiplication:")
    c = matmul(a, b)
    print(f"a @ b (shape {c.shape}): {c.data}")

    # Element-wise operations
    print("\n3. Element-wise operations:")
    d = a + 1
    print(f"a + 1: {d.data}")

    e = a * 2
    print(f"a * 2: {e.data}")

    # Softmax
    print("\n4. Softmax:")
    f = Tensor([[1.0, 2.0, 3.0]], 'float')
    g = softmax(f, dim=1)
    print(f"softmax([1, 2, 3]): {g.data}")

    # Reshaping
    print("\n5. Reshaping:")
    h = arange(0, 12, dtype='float')
    print(f"arange(0, 12): {h.data}")

    i = h.reshape(3, 4)
    print(f"reshaped to (3, 4): {i.data}")

    # Transpose
    print("\n6. Transpose:")
    j = i.transpose(1, 0)
    print(f"transposed: {j.data}")

    print("\n✅ All tensor operations completed successfully!")


if __name__ == "__main__":
    main()