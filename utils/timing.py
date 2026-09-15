"""
utils.timing — GPU-aware wall-clock timing helpers.

timer(device)
    Context manager that measures elapsed wall-clock seconds.  When the
    active device is CUDA, torch.cuda.synchronize() is called before
    and after the timed block so that all pending GPU kernels are
    flushed and the reported time accurately reflects GPU computation
    rather than just CPU dispatch latency.  Usage::

        with timer(device) as t:
            do_work()
        print(t["elapsed"])   # seconds, float

peak_gpu_memory_mb(device) -> float
    Returns the peak GPU memory allocated (in MB) since the last reset.
    Returns 0.0 on CPU.  Wraps torch.cuda.max_memory_allocated.

reset_peak_gpu_memory(device) -> None
    Resets the peak-memory tracker so the next call to
    peak_gpu_memory_mb() reflects only the work done after the reset.
    Called once per epoch in the Trainer so per-epoch peak memory is
    reported rather than the session-wide high watermark.
"""
import time
from contextlib import contextmanager

import torch


@contextmanager
def timer(device: torch.device):
    """
    Context manager returning elapsed wall-clock seconds, synchronizing
    CUDA before/after so GPU work is actually accounted for.
    """
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = {"elapsed": None}
    try:
        yield result
    finally:
        if is_cuda:
            torch.cuda.synchronize()
        result["elapsed"] = time.perf_counter() - start


def peak_gpu_memory_mb(device: torch.device) -> float:
    if torch.device(device).type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def reset_peak_gpu_memory(device: torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
