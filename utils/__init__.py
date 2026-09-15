"""
utils — shared infrastructure utilities.

This package collects small, reusable helpers that are independent of
any specific algorithm, model, or dataset.  None of the modules here
contain algorithm-specific mathematics.

Modules
-------
device.py
    resolve_device(cfg) — single entry point for selecting the
    torch.device for a run; warns when CUDA is requested but unavailable
    and falls back to CPU.

logging.py
    setup_logger(name, log_dir) — configures a Python Logger that
    writes to both stdout and a per-run .log file.
    JsonlLogger — append-only structured event log (one JSON per line).

plots.py
    Matplotlib-based plotting utilities that operate purely on the
    JSONL results schema (utils/results.py).  Exports:
    plot_loss_vs_epoch, plot_training_time_comparison,
    plot_sampling_time_vs_nfe, plot_fid_vs_nfe, plot_is_vs_nfe,
    plot_gpu_memory_comparison, plot_fid_vs_sampling_time.

results.py
    ResultRecord — dataclass for one metrics row (training, sampling,
    or evaluation).  ResultsWriter — appends records as JSONL and
    optionally exports to CSV.

seed.py
    set_seed(seed) — seeds Python random, NumPy, and PyTorch (CPU + GPU).
    set_deterministic(flag) — toggles cudnn determinism.

timing.py
    timer(device) — context manager that measures wall-clock seconds
    with CUDA synchronization so GPU work is fully accounted for.
    peak_gpu_memory_mb(device), reset_peak_gpu_memory(device).
"""
# No default re-exports: import from the individual sub-modules directly
# to keep the dependency graph explicit.
