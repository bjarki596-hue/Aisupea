"""
Aisupea Distributed Computing Module

Basic distributed tensor operations and multi-device support.
"""

import threading
import queue
import time
from typing import List, Dict, Any, Optional, Callable
from ..core import Tensor


class Device:
    """Represents a computing device (CPU, GPU, etc.)."""

    def __init__(self, device_type: str = "cpu", device_id: int = 0):
        self.device_type = device_type
        self.device_id = device_id
        self.memory = {}  # Simple memory storage

    def __str__(self):
        return f"{self.device_type}:{self.device_id}"

    def allocate(self, tensor: Tensor) -> str:
        """Allocate tensor on device."""
        tensor_id = f"tensor_{len(self.memory)}"
        self.memory[tensor_id] = tensor
        return tensor_id

    def get(self, tensor_id: str) -> Tensor:
        """Get tensor from device memory."""
        return self.memory.get(tensor_id)

    def free(self, tensor_id: str):
        """Free tensor from device memory."""
        if tensor_id in self.memory:
            del self.memory[tensor_id]


class DistributedContext:
    """Context for distributed computing operations."""

    def __init__(self):
        self.devices = []
        self.current_device = None
        self.communication_queue = queue.Queue()
        self.workers = []
        self.is_initialized = False

    def init_process_group(self, num_devices: int = 1, device_type: str = "cpu"):
        """Initialize process group with multiple devices."""
        self.devices = [Device(device_type, i) for i in range(num_devices)]
        self.current_device = self.devices[0] if self.devices else None
        self.is_initialized = True

        # Start worker threads for each device
        for device in self.devices:
            worker = threading.Thread(target=self._device_worker, args=(device,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _device_worker(self, device: Device):
        """Worker thread for device operations."""
        while True:
            try:
                task = self.communication_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break

                operation, args, callback = task
                try:
                    result = operation(device, *args)
                    if callback:
                        callback(result)
                except Exception as e:
                    if callback:
                        callback(e)

                self.communication_queue.task_done()
            except queue.Empty:
                continue

    def shutdown(self):
        """Shutdown distributed context."""
        # Send shutdown signals
        for _ in self.workers:
            self.communication_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)

        self.workers.clear()
        self.devices.clear()
        self.is_initialized = False

    def get_device(self, device_id: int = 0) -> Device:
        """Get device by ID."""
        if device_id < len(self.devices):
            return self.devices[device_id]
        raise ValueError(f"Device {device_id} not available")

    def set_device(self, device: Device):
        """Set current device."""
        self.current_device = device

    def to_device(self, tensor: Tensor, device: Device) -> str:
        """Move tensor to device."""
        return device.allocate(tensor)

    def from_device(self, tensor_id: str, device: Device) -> Tensor:
        """Get tensor from device."""
        return device.get(tensor_id)


# Global distributed context
_distributed_context = DistributedContext()


def init_distributed(num_devices: int = 1, device_type: str = "cpu"):
    """Initialize distributed computing."""
    _distributed_context.init_process_group(num_devices, device_type)


def get_device(device_id: int = 0) -> Device:
    """Get computing device."""
    return _distributed_context.get_device(device_id)


def set_device(device: Device):
    """Set current device."""
    _distributed_context.set_device(device)


def tensor_to_device(tensor: Tensor, device: Device) -> str:
    """Move tensor to device."""
    return _distributed_context.to_device(tensor, device)


def tensor_from_device(tensor_id: str, device: Device) -> Tensor:
    """Get tensor from device."""
    return _distributed_context.from_device(tensor_id, device)


def all_reduce(tensor: Tensor, op: str = "sum") -> Tensor:
    """
    All-reduce operation across devices.

    Args:
        tensor: Input tensor
        op: Reduction operation ("sum", "mean", "max", "min")

    Returns:
        Reduced tensor
    """
    if not _distributed_context.is_initialized:
        return tensor  # No-op if not distributed

    # Simple implementation - in real distributed system this would
    # communicate across actual devices/processes
    results = [tensor]  # Simulate gathering from all devices

    if op == "sum":
        result = results[0]
        for t in results[1:]:
            result = result + t
    elif op == "mean":
        sum_tensor = results[0]
        for t in results[1:]:
            sum_tensor = sum_tensor + t
        result = sum_tensor / len(results)
    elif op == "max":
        result = results[0]
        for t in results[1:]:
            result = Tensor._element_wise_op(result, t, lambda a, b: max(a, b))
    elif op == "min":
        result = results[0]
        for t in results[1:]:
            result = Tensor._element_wise_op(result, t, lambda a, b: min(a, b))
    else:
        raise ValueError(f"Unknown reduction operation: {op}")

    return result


def all_gather(tensor: Tensor) -> List[Tensor]:
    """
    All-gather operation across devices.

    Args:
        tensor: Input tensor

    Returns:
        List of tensors from all devices
    """
    if not _distributed_context.is_initialized:
        return [tensor]

    # Simulate gathering from all devices
    num_devices = len(_distributed_context.devices)
    return [tensor] * num_devices


def reduce_scatter(tensor: Tensor, op: str = "sum") -> Tensor:
    """
    Reduce-scatter operation across devices.

    Args:
        tensor: Input tensor
        op: Reduction operation

    Returns:
        Reduced and scattered tensor
    """
    if not _distributed_context.is_initialized:
        return tensor

    # First reduce
    reduced = all_reduce(tensor, op)

    # Then scatter (simple split for simulation)
    num_devices = len(_distributed_context.devices)
    chunk_size = reduced.shape[0] // num_devices if reduced.ndim > 0 else 1

    # Return first chunk (simplified)
    if reduced.ndim == 1:
        return Tensor(reduced.data[:chunk_size], reduced.dtype)
    else:
        return reduced  # Simplified


def barrier():
    """Synchronization barrier across all devices."""
    if not _distributed_context.is_initialized:
        return

    # In real implementation, this would synchronize all processes
    time.sleep(0.001)  # Simulate synchronization


def broadcast(tensor: Tensor, src_device: int = 0) -> Tensor:
    """
    Broadcast tensor from source device to all devices.

    Args:
        tensor: Tensor to broadcast
        src_device: Source device ID

    Returns:
        Broadcast tensor
    """
    if not _distributed_context.is_initialized:
        return tensor

    # Simulate broadcast
    return tensor


class DistributedDataParallel:
    """Simple distributed data parallel wrapper."""

    def __init__(self, model):
        self.model = model
        self.devices = _distributed_context.devices

    def forward(self, *inputs):
        """Forward pass with distributed processing."""
        if not self.devices:
            return self.model(*inputs)

        # Simple implementation - process on first device
        device = self.devices[0]
        # In real implementation, would split inputs across devices
        return self.model(*inputs)

    def backward(self, loss):
        """Backward pass with gradient synchronization."""
        # Compute gradients
        # In real implementation, would aggregate gradients across devices
        return all_reduce(loss, "sum")


def parallel_apply(func: Callable, tensors: List[Tensor]) -> List[Tensor]:
    """
    Apply function in parallel across tensors.

    Args:
        func: Function to apply
        tensors: List of input tensors

    Returns:
        List of output tensors
    """
    if not _distributed_context.is_initialized or len(tensors) == 1:
        return [func(tensor) for tensor in tensors]

    # Simple parallel execution using threads
    results = []
    threads = []

    def worker(tensor, result_list, index):
        result_list[index] = func(tensor)

    for i, tensor in enumerate(tensors):
        results.append(None)
        thread = threading.Thread(target=worker, args=(tensor, results, i))
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    return results


__all__ = [
    'Device',
    'DistributedContext',
    'init_distributed',
    'get_device',
    'set_device',
    'tensor_to_device',
    'tensor_from_device',
    'all_reduce',
    'all_gather',
    'reduce_scatter',
    'barrier',
    'broadcast',
    'DistributedDataParallel',
    'parallel_apply'
]