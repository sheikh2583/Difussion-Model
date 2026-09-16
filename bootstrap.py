#!/usr/bin/env python3
"""
bootstrap.py — Universal one-shot project setup.

Run once after cloning on any machine (Windows / Linux / macOS):
    python bootstrap.py           # interactive
    python bootstrap.py --yes     # non-interactive (CI / lab PC)

What this does:
  1. Checks Python version (≥ 3.9 required)
  2. Creates a virtual environment in ./venv  (skips if already exists)
  3. Detects GPU: CUDA (nvidia-smi) → ROCm (Linux) → CPU fallback
  4. Installs the correct PyTorch build for the detected hardware
  5. Installs remaining project dependencies from requirements.txt
  6. Creates runtime directories: data/raw/, results/, results/metrics/
  7. Prints a getting-started summary

No shell-specific syntax — pure stdlib Python 3, works everywhere.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = platform.system() == "Windows"

VENV_PYTHON = (
    ROOT / "venv" / "Scripts" / "python.exe"
    if IS_WINDOWS
    else ROOT / "venv" / "bin" / "python"
)

VENV_PIP = (
    ROOT / "venv" / "Scripts" / "pip.exe"
    if IS_WINDOWS
    else ROOT / "venv" / "bin" / "pip"
)


def _c(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code (skipped on Windows without VT support)."""
    if IS_WINDOWS and os.environ.get("TERM") is None:
        return text
    return f"\033[{code}m{text}\033[0m"


def info(msg: str)  -> None: print(_c(f"  ✓ {msg}", "32"))
def warn(msg: str)  -> None: print(_c(f"  ⚠ {msg}", "33"))
def error(msg: str) -> None: print(_c(f"  ✗ {msg}", "31"))
def head(msg: str)  -> None: print(_c(f"\n{msg}", "36;1"))


def run(cmd: list, check=True, capture=False) -> subprocess.CompletedProcess:
    kwargs = dict(check=check)
    if capture:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Step 1 — Python version check
# ---------------------------------------------------------------------------

def check_python() -> None:
    head("Step 1 — Python version")
    v = sys.version_info
    if v < (3, 9):
        error(f"Python {v.major}.{v.minor} detected. Python ≥ 3.9 is required.")
        sys.exit(1)
    info(f"Python {v.major}.{v.minor}.{v.micro} OK")


# ---------------------------------------------------------------------------
# Step 2 — Create virtual environment
# ---------------------------------------------------------------------------

def create_venv() -> None:
    head("Step 2 — Virtual environment")
    venv_dir = ROOT / "venv"
    if VENV_PYTHON.exists():
        info("venv/ already exists — skipping creation")
        return
    print("  Creating venv/ ...")
    run([sys.executable, "-m", "venv", str(venv_dir)])
    info("venv/ created")


# ---------------------------------------------------------------------------
# Step 3 — GPU detection
# ---------------------------------------------------------------------------

def detect_gpu() -> str:
    """
    Returns one of:
      'cuda118'  — CUDA ≥ 11.8  (most RTX cards, recommended)
      'cuda121'  — CUDA ≥ 12.1  (newer drivers / H100)
      'rocm'     — AMD ROCm (Linux only)
      'cpu'      — no GPU / unknown
    """
    head("Step 3 — GPU detection")

    # --- NVIDIA via nvidia-smi ---
    try:
        result = run(["nvidia-smi", "--query-gpu=name,driver_version",
                      "--format=csv,noheader"], capture=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().splitlines()[0]
            info(f"NVIDIA GPU detected: {gpu_info}")

            # Try nvcc to get CUDA toolkit version
            nvcc = run(["nvcc", "--version"], capture=True, check=False)
            if nvcc.returncode == 0:
                for line in nvcc.stdout.splitlines():
                    if "release" in line.lower():
                        # e.g. "Cuda compilation tools, release 12.1, V12.1.105"
                        parts = line.split("release")[-1].strip().split(",")[0].strip()
                        major, minor = (int(x) for x in parts.split(".")[:2])
                        if (major, minor) >= (12, 1):
                            info(f"CUDA toolkit {major}.{minor} → using cu121 wheels")
                            return "cuda121"
                        else:
                            info(f"CUDA toolkit {major}.{minor} → using cu118 wheels")
                            return "cuda118"

            # nvcc not in PATH — default to cu118 (broadest compatibility)
            warn("nvcc not found in PATH. Defaulting to CUDA 11.8 wheels.")
            warn("If you have CUDA 12+, rerun with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return "cuda118"
    except FileNotFoundError:
        pass  # nvidia-smi not found

    # --- AMD ROCm (Linux only) ---
    if not IS_WINDOWS:
        try:
            result = run(["rocm-smi", "--showproductname"], capture=True, check=False)
            if result.returncode == 0:
                info("AMD ROCm GPU detected")
                return "rocm"
        except FileNotFoundError:
            pass

    warn("No GPU detected — installing CPU-only PyTorch.")
    warn("Training will be slow. Connect a CUDA/ROCm GPU for real runs.")
    return "cpu"


# ---------------------------------------------------------------------------
# Step 4 — Install PyTorch
# ---------------------------------------------------------------------------

TORCH_INDEX_URLS = {
    "cuda118": "https://download.pytorch.org/whl/cu118",
    "cuda121": "https://download.pytorch.org/whl/cu121",
    "rocm":    "https://download.pytorch.org/whl/rocm6.0",
    "cpu":     "https://download.pytorch.org/whl/cpu",
}

TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]


def install_torch(gpu_type: str) -> None:
    head("Step 4 — PyTorch installation")
    index_url = TORCH_INDEX_URLS[gpu_type]
    print(f"  Index URL : {index_url}")
    print(f"  Packages  : {' '.join(TORCH_PACKAGES)}")
    print("  (this may take a few minutes on first install) ...")
    run([
        str(VENV_PIP), "install", "--upgrade",
        *TORCH_PACKAGES,
        "--index-url", index_url,
    ])
    info(f"PyTorch installed ({gpu_type})")


# ---------------------------------------------------------------------------
# Step 5 — Install remaining requirements
# ---------------------------------------------------------------------------

def install_requirements() -> None:
    head("Step 5 — Project dependencies")
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        warn("requirements.txt not found — skipping")
        return

    # Install everything except torch/torchvision/torchaudio (already installed)
    with open(req_file) as f:
        lines = [
            ln.strip() for ln in f
            if ln.strip()
            and not ln.startswith("#")
            and not any(ln.lower().startswith(pkg)
                        for pkg in ("torch", "torchvision", "torchaudio"))
        ]

    if not lines:
        info("No additional dependencies to install")
        return

    print(f"  Installing {len(lines)} package(s) ...")
    run([str(VENV_PIP), "install", "--upgrade", *lines])
    info("Dependencies installed")


# ---------------------------------------------------------------------------
# Step 6 — Runtime directories
# ---------------------------------------------------------------------------

RUNTIME_DIRS = [
    "data/raw",
    "results",
    "results/metrics",
]


def create_dirs() -> None:
    head("Step 6 — Runtime directories")
    for d in RUNTIME_DIRS:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        info(f"{'(exists)' if path.exists() else 'created'} {d}/")


# ---------------------------------------------------------------------------
# Step 7 — Summary
# ---------------------------------------------------------------------------

def print_summary(gpu_type: str) -> None:
    head("Setup complete!")
    print()

    activate = (
        r"  venv\Scripts\activate"
        if IS_WINDOWS else
        "  source venv/bin/activate"
    )
    python_cmd = "  python" if IS_WINDOWS else "  python"

    gpu_label = {
        "cuda118": "CUDA 11.8 (NVIDIA)",
        "cuda121": "CUDA 12.1 (NVIDIA)",
        "rocm":    "ROCm (AMD)",
        "cpu":     "CPU only",
    }[gpu_type]

    print(_c("  Hardware target:", "1") + f" {gpu_label}")
    print()
    print(_c("  Next steps:", "1"))
    print(f"{activate}          ← activate the environment")
    print()
    print("  # Quick smoke test (2 epochs, no GPU required):")
    print(f"{python_cmd} train.py --algorithm mock --config config/smoke_fast.json")
    print()
    print("  # Train all 6 algorithms (CIFAR-10):")
    print("  bash scripts/train_all.sh          # Linux/macOS")
    print(r"  .\scripts\train_all.ps1            # Windows PowerShell")
    print()
    print("  # Start the inference UI:")
    print(f"{python_cmd} web/inference_server.py")
    print("  → Open http://127.0.0.1:8000 in a browser")
    print()
    print(_c("  Docs:", "1") + " docs/IMPLEMENTATION_LOG.md  (status + commands)")
    print(_c("  Theory:", "1") + " docs/THEORY_NOTES.md        (maths + citations)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-shot project bootstrap.")
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Non-interactive: skip all prompts and proceed automatically."
    )
    p.add_argument(
        "--gpu", choices=["cuda118", "cuda121", "rocm", "cpu"], default=None,
        help="Force a specific PyTorch build instead of auto-detecting."
    )
    p.add_argument(
        "--skip-torch", action="store_true",
        help="Skip PyTorch installation (use if already installed in the venv)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(_c("=" * 58, "36;1"))
    print(_c("  DiffusionProject — Bootstrap", "36;1"))
    print(_c("=" * 58, "36;1"))
    print(f"  OS      : {platform.system()} {platform.machine()}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Root    : {ROOT}")

    check_python()
    create_venv()

    gpu_type = args.gpu if args.gpu else detect_gpu()

    if not args.yes and not args.gpu:
        print(f"\n  Detected GPU type: {_c(gpu_type, '33')}")
        ans = input("  Proceed with this selection? [Y/n]: ").strip().lower()
        if ans in ("n", "no"):
            gpu_type = input(
                "  Enter GPU type [cuda118 / cuda121 / rocm / cpu]: "
            ).strip()

    if not args.skip_torch:
        install_torch(gpu_type)

    install_requirements()
    create_dirs()
    print_summary(gpu_type)


if __name__ == "__main__":
    main()
