"""
Aisupea CNN Layers Module

Convolutional neural network layers for vision tasks.
"""

from typing import Optional, Tuple, Union
from ..core import Tensor
from ..autograd import Variable


class Conv2D:
    """2D Convolutional layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 bias: bool = True):
        """
        Initialize Conv2D layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding added to input
            bias: Whether to include bias term
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        # Initialize weights and bias
        self.weight = Tensor.zeros(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        self.bias = Tensor.zeros(out_channels) if bias else None

        # Xavier initialization
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = out_channels * self.kernel_size[0] * self.kernel_size[1]
        scale = (2.0 / (fan_in + fan_out)) ** 0.5

        # Random initialization (simplified)
        import random
        for i in range(out_channels):
            for j in range(in_channels):
                for k in range(self.kernel_size[0]):
                    for l in range(self.kernel_size[1]):
                        self.weight.data[i][j][k][l] = random.gauss(0, scale)

        if self.bias is not None:
            for i in range(out_channels):
                self.bias.data[i] = random.gauss(0, scale)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of Conv2D.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor
        """
        batch_size, in_channels, height, width = x.shape

        # Calculate output dimensions
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        # Initialize output
        output = Tensor.zeros(batch_size, self.out_channels, out_height, out_width)

        # Add padding to input
        if self.padding[0] > 0 or self.padding[1] > 0:
            padded_x = self._add_padding(x, self.padding)
        else:
            padded_x = x

        # Convolution operation
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride[0]
                        w_start = w * self.stride[1]

                        # Compute convolution
                        conv_sum = 0.0
                        for in_c in range(in_channels):
                            for kh in range(self.kernel_size[0]):
                                for kw in range(self.kernel_size[1]):
                                    conv_sum += (padded_x.data[b][in_c][h_start + kh][w_start + kw] *
                                               self.weight.data[out_c][in_c][kh][kw])

                        if self.bias is not None:
                            conv_sum += self.bias.data[out_c]

                        output.data[b][out_c][h][w] = conv_sum

        return output

    def _add_padding(self, x: Tensor, padding: Tuple[int, int]) -> Tensor:
        """Add padding to input tensor."""
        batch_size, channels, height, width = x.shape
        padded_height = height + 2 * padding[0]
        padded_width = width + 2 * padding[1]

        padded_data = []
        for b in range(batch_size):
            batch_data = []
            for c in range(channels):
                channel_data = []
                # Add top padding
                for _ in range(padding[0]):
                    channel_data.append([0.0] * padded_width)

                for h in range(height):
                    row = [0.0] * padding[1]  # Left padding
                    row.extend(x.data[b][c][h])  # Original data
                    row.extend([0.0] * padding[1])  # Right padding
                    channel_data.append(row)

                # Add bottom padding
                for _ in range(padding[0]):
                    channel_data.append([0.0] * padded_width)

                batch_data.append(channel_data)
            padded_data.append(batch_data)

        return Tensor(padded_data)


class MaxPool2D:
    """2D Max Pooling layer."""

    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = None):
        """
        Initialize MaxPool2D layer.

        Args:
            kernel_size: Size of pooling kernel
            stride: Stride of pooling (defaults to kernel_size)
        """
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of MaxPool2D.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor
        """
        batch_size, channels, height, width = x.shape

        out_height = (height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width - self.kernel_size[1]) // self.stride[1] + 1

        output = Tensor.zeros(batch_size, channels, out_height, out_width)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride[0]
                        w_start = w * self.stride[1]

                        # Find max in pooling window
                        max_val = float('-inf')
                        for kh in range(self.kernel_size[0]):
                            for kw in range(self.kernel_size[1]):
                                val = x.data[b][c][h_start + kh][w_start + kw]
                                if val > max_val:
                                    max_val = val

                        output.data[b][c][h][w] = max_val

        return output


class AvgPool2D:
    """2D Average Pooling layer."""

    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: Optional[Union[int, Tuple[int, int]]] = None):
        """
        Initialize AvgPool2D layer.

        Args:
            kernel_size: Size of pooling kernel
            stride: Stride of pooling (defaults to kernel_size)
        """
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of AvgPool2D.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor
        """
        batch_size, channels, height, width = x.shape

        out_height = (height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width - self.kernel_size[1]) // self.stride[1] + 1

        output = Tensor.zeros(batch_size, channels, out_height, out_width)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride[0]
                        w_start = w * self.stride[1]

                        # Compute average in pooling window
                        total = 0.0
                        count = 0
                        for kh in range(self.kernel_size[0]):
                            for kw in range(self.kernel_size[1]):
                                total += x.data[b][c][h_start + kh][w_start + kw]
                                count += 1

                        output.data[b][c][h][w] = total / count

        return output


class BatchNorm2D:
    """2D Batch Normalization layer."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Initialize BatchNorm2D layer.

        Args:
            num_features: Number of features/channels
            eps: Small value for numerical stability
            momentum: Momentum for running statistics
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.weight = Tensor.ones(num_features)  # Scale parameter
        self.bias = Tensor.zeros(num_features)   # Shift parameter

        # Running statistics
        self.running_mean = Tensor.zeros(num_features)
        self.running_var = Tensor.ones(num_features)

        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of BatchNorm2D.

        Args:
            x: Input tensor of shape (batch_size, num_features, height, width)

        Returns:
            Normalized tensor
        """
        if self.training:
            # Compute batch statistics
            batch_mean = self._compute_batch_mean(x)
            batch_var = self._compute_batch_var(x, batch_mean)

            # Update running statistics
            self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + batch_var * self.momentum
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Normalize
        normalized = (x - batch_mean.unsqueeze(-1).unsqueeze(-1)) / (batch_var.unsqueeze(-1).unsqueeze(-1) + self.eps).sqrt()

        # Scale and shift
        return normalized * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1)

    def _compute_batch_mean(self, x: Tensor) -> Tensor:
        """Compute mean across batch, height, and width dimensions."""
        batch_size, channels, height, width = x.shape
        mean = Tensor.zeros(channels)

        for c in range(channels):
            total = 0.0
            count = 0
            for b in range(batch_size):
                for h in range(height):
                    for w in range(width):
                        total += x.data[b][c][h][w]
                        count += 1
            mean.data[c] = total / count

        return mean

    def _compute_batch_var(self, x: Tensor, mean: Tensor) -> Tensor:
        """Compute variance across batch, height, and width dimensions."""
        batch_size, channels, height, width = x.shape
        var = Tensor.zeros(channels)

        for c in range(channels):
            total = 0.0
            count = 0
            for b in range(batch_size):
                for h in range(height):
                    for w in range(width):
                        diff = x.data[b][c][h][w] - mean.data[c]
                        total += diff * diff
                        count += 1
            var.data[c] = total / count

        return var


class AdaptiveAvgPool2D:
    """Adaptive Average Pooling 2D layer."""

    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        """
        Initialize AdaptiveAvgPool2D layer.

        Args:
            output_size: Desired output size (height, width)
        """
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of AdaptiveAvgPool2D.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, channels, output_size[0], output_size[1])
        """
        batch_size, channels, height, width = x.shape
        out_height, out_width = self.output_size

        output = Tensor.zeros(batch_size, channels, out_height, out_width)

        # Compute pooling regions
        h_stride = height / out_height
        w_stride = width / out_width

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # Compute region bounds
                        h_start = int(oh * h_stride)
                        h_end = int((oh + 1) * h_stride)
                        w_start = int(ow * w_stride)
                        w_end = int((ow + 1) * w_stride)

                        # Average pooling in region
                        total = 0.0
                        count = 0
                        for h in range(h_start, min(h_end, height)):
                            for w in range(w_start, min(w_end, width)):
                                total += x.data[b][c][h][w]
                                count += 1

                        output.data[b][c][oh][ow] = total / count if count > 0 else 0.0

        return output


__all__ = [
    'Conv2D',
    'MaxPool2D',
    'AvgPool2D',
    'BatchNorm2D',
    'AdaptiveAvgPool2D'
]