"""
Aisupea Utilities

General utility functions and helpers.
"""

import os
import json
import pickle
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def save_json(data: Any, file_path: str, indent: int = 2):
    """Save data to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """Load data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, file_path: str):
    """Save data to pickle file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: str) -> Any:
    """Load data from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(dir_path: str):
    """Ensure directory exists."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def list_files(dir_path: str, pattern: str = "*") -> List[str]:
    """List files in directory matching pattern."""
    import glob
    return glob.glob(os.path.join(dir_path, pattern))


def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Read entire file as string."""
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def write_file(file_path: str, content: str, encoding: str = 'utf-8'):
    """Write string to file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)


def split_text(text: str, chunk_size: int = 1000, overlap: int = 0) -> List[str]:
    """Split text into chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Find word boundary if possible
        if end < len(text):
            # Look for whitespace
            while end > start and text[end] not in ' \t\n':
                end -= 1
            if end == start:
                end = start + chunk_size  # No word boundary found

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if overlap > 0 else end

    return chunks


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity (Jaccard similarity)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


def format_bytes(size: int) -> str:
    """Format bytes into human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return ".1f"
        size /= 1024.0
    return ".1f"


def get_system_info() -> Dict[str, Any]:
    """Get basic system information."""
    import platform
    import psutil

    try:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available
        }
    except ImportError:
        # Fallback if psutil not available
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count()
        }


def time_function(func):
    """Decorator to time function execution."""
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print(".4f")
        return result

    return wrapper


class Config:
    """Simple configuration management."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value

    def update(self, other: Dict[str, Any]):
        """Update configuration with another dict."""
        self._config.update(other)

    def save(self, file_path: str):
        """Save configuration to file."""
        save_json(self._config, file_path)

    @classmethod
    def load(cls, file_path: str) -> 'Config':
        """Load configuration from file."""
        return cls(load_json(file_path))

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._config


def create_progress_bar(total: int, prefix: str = "", suffix: str = "",
                       length: int = 50, fill: str = "█") -> 'ProgressBar':
    """Create a progress bar."""
    return ProgressBar(total, prefix, suffix, length, fill)


class ProgressBar:
    """Simple progress bar."""

    def __init__(self, total: int, prefix: str = "", suffix: str = "",
                 length: int = 50, fill: str = "█"):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.current = 0

    def update(self, amount: int = 1):
        """Update progress."""
        self.current += amount
        self._display()

    def set_progress(self, current: int):
        """Set current progress."""
        self.current = current
        self._display()

    def _display(self):
        """Display progress bar."""
        percent = (self.current / self.total) * 100
        filled_length = int(self.length * self.current // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)

        print("2d"
              "50", end="", flush=True)

        if self.current >= self.total:
            print()  # New line when complete


def parallel_map(func, items: List[Any], num_workers: int = 4) -> List[Any]:
    """Apply function to items in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        results = []

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error in parallel execution: {e}")
                results.append(None)

    return results


def debounce(wait_time: float):
    """Decorator to debounce function calls."""
    import time
    from functools import wraps

    def decorator(func):
        last_called = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if current_time - last_called[0] >= wait_time:
                last_called[0] = current_time
                return func(*args, **kwargs)

        return wrapper

    return decorator


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to retry function calls on failure."""
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator