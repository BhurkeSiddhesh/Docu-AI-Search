import os
import psutil


def get_memory_usage() -> float:
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return float(memory_info.rss) / (1024 * 1024)
