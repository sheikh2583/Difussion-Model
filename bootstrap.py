#!/usr/bin/env python3
"""
bootstrap.py — Universal one-shot project setup.

Run once after cloning on any machine (Windows / Linux / macOS):
    python bootstrap.py           # interactive
    python bootstrap.py --yes     # non-interactive (CI / lab PC)

What this does:
  1. Checks Python version (≥ 3.10 required)
  2. Creates a virtual environment in ./venv  (skips if already exists)
  3. Detects GPU: CUDA (nvidia-smi) → ROCm (Linux) → CPU fallback
  4. Installs the correct PyTorch build for the detected hardware
  5. Installs remaining project dependencies from requirements.txt
  6. Downloads/verifies the pretrained metric feature extractor
  7. Creates runtime directories: data/raw/, results/, results/metrics/
  8. Downloads the selected dataset(s), including the large source archives
  9. Prints a getting-started summary

No shell-specific syntax — pure stdlib Python 3, works everywhere.
"""

import argparse
import os
import platform
import re
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

def _c(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code (skipped on Windows without VT support)."""
    if not sys.stdout.isatty():
        return text
    if IS_WINDOWS and not (os.environ.get("TERM") or os.environ.get("WT_SESSION")):
        return text
    return f"\033[{code}m{text}\033[0m"


def info(msg: str)  -> None: print(_c(f"  [OK] {msg}", "32"))
def warn(msg: str)  -> None: print(_c(f"  [WARN] {msg}", "33"))
def error(msg: str) -> None: print(_c(f"  [ERROR] {msg}", "31"))
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
    head("Step 1 - Python version")
    v = sys.version_info
    if v < (3, 10):
        error(f"Python {v.major}.{v.minor} detected. Python 3.10+ is required.")
        sys.exit(1)
    info(f"Python {v.major}.{v.minor}.{v.micro} OK")


# ---------------------------------------------------------------------------
# Step 2 — Create virtual environment
# ---------------------------------------------------------------------------

def create_venv() -> None:
    head("Step 2 - Virtual environment")
    venv_dir = ROOT / "venv"
    if VENV_PYTHON.exists():
        probe = run([str(VENV_PYTHON), "--version"], check=False, capture=True)
        if probe.returncode == 0:
            info("venv/ already exists - skipping creation")
            return
        warn("venv/ exists but its Python launcher is stale; repairing it")
        run([sys.executable, "-m", "venv", "--upgrade", str(venv_dir)])
        info("venv/ repaired")
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
      'cuda128'  — CUDA ≥ 12.8  (current NVIDIA drivers)
      'rocm'     — AMD ROCm (Linux only)
      'cpu'      — no GPU / unknown
    """
    head("Step 3 - GPU detection")

    # --- NVIDIA via nvidia-smi ---
    try:
        result = run(["nvidia-smi", "--query-gpu=name,driver_version",
                      "--format=csv,noheader"], capture=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().splitlines()[0]
            info(f"NVIDIA GPU detected: {gpu_info}")

            # Try nvcc to get CUDA toolkit version. The toolkit is optional;
            # PyTorch wheels only require a sufficiently recent NVIDIA driver.
            try:
                nvcc = run(["nvcc", "--version"], capture=True, check=False)
            except FileNotFoundError:
                nvcc = None
            if nvcc is not None and nvcc.returncode == 0:
                for line in nvcc.stdout.splitlines():
                    if "release" in line.lower():
                        # e.g. "Cuda compilation tools, release 12.1, V12.1.105"
                        parts = line.split("release")[-1].strip().split(",")[0].strip()
                        major, minor = (int(x) for x in parts.split(".")[:2])
                        if (major, minor) >= (12, 8):
                            info(f"CUDA toolkit {major}.{minor} -> using cu128 wheels")
                            return "cuda128"
                        if (major, minor) >= (12, 1):
                            info(f"CUDA toolkit {major}.{minor} -> using cu121 wheels")
                            return "cuda121"
                        else:
                            info(f"CUDA toolkit {major}.{minor} -> using cu118 wheels")
                            return "cuda118"

            # Modern NVIDIA drivers support the CUDA runtime bundled with the
            # wheel; a separately installed CUDA toolkit is not required.
            try:
                driver_major = int(gpu_info.rsplit(",", 1)[1].strip().split(".")[0])
            except (IndexError, ValueError):
                driver_major = 0
            if driver_major >= 570:
                target = "cuda128"
            elif driver_major >= 525:
                target = "cuda121"
            else:
                target = "cuda118"
            warn(f"nvcc not found in PATH; selecting {target} from the NVIDIA driver.")
            return target
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

    warn("No GPU detected - installing CPU-only PyTorch.")
    warn("Training will be slow. Connect a CUDA/ROCm GPU for real runs.")
    return "cpu"


# ---------------------------------------------------------------------------
# Step 4 — Install PyTorch
# ---------------------------------------------------------------------------

TORCH_INDEX_URLS = {
    "cuda118": "https://download.pytorch.org/whl/cu118",
    "cuda121": "https://download.pytorch.org/whl/cu121",
    "cuda128": "https://download.pytorch.org/whl/cu128",
    "rocm":    "https://download.pytorch.org/whl/rocm6.0",
    "cpu":     "https://download.pytorch.org/whl/cpu",
}

TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]


def install_torch(gpu_type: str) -> str:
    head("Step 4 - PyTorch installation")
    index_url = TORCH_INDEX_URLS[gpu_type]
    print(f"  Index URL : {index_url}")
    print(f"  Packages  : {' '.join(TORCH_PACKAGES)}")
    print("  (this may take a few minutes on first install) ...")
    command = [
        str(VENV_PYTHON), "-m", "pip", "install", "--upgrade",
        *TORCH_PACKAGES, "--index-url", index_url,
    ]
    try:
        run(command)
    except subprocess.CalledProcessError:
        if gpu_type == "cpu":
            raise
        warn(f"The {gpu_type} wheel is unavailable on this OS/Python combination.")
        warn("Falling back to the portable CPU PyTorch build; rerun setup later to change it.")
        gpu_type = "cpu"
        run([
            str(VENV_PYTHON), "-m", "pip", "install", "--upgrade",
            *TORCH_PACKAGES, "--index-url", TORCH_INDEX_URLS["cpu"],
        ])
    info(f"PyTorch installed ({gpu_type})")
    return gpu_type


# ---------------------------------------------------------------------------
# Step 5 — Install remaining requirements
# ---------------------------------------------------------------------------

def install_requirements() -> None:
    head("Step 5 - Project dependencies")
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        warn("requirements.txt not found - skipping")
        return

    # Install everything except torch/torchvision/torchaudio (already installed)
    lines = []
    with open(req_file, encoding="utf-8") as f:
        for raw_line in f:
            requirement = raw_line.split("#", 1)[0].strip()
            if not requirement:
                continue
            match = re.match(r"[A-Za-z0-9_.-]+", requirement)
            package = match.group(0).lower().replace("_", "-") if match else ""
            if package in {"torch", "torchvision", "torchaudio"}:
                continue
            lines.append(requirement)

    if not lines:
        info("No additional dependencies to install")
        return

    print(f"  Installing {len(lines)} package(s) ...")
    run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", *lines])
    info("Dependencies installed")


def prefetch_metric_assets() -> None:
    """Download the pretrained Inception weights used by FID and IS."""
    head("Step 6 - Evaluation assets")
    code = (
        "import warnings; "
        "warnings.filterwarnings('ignore', message='Metric `InceptionScore`.*'); "
        "from torchmetrics.image.fid import FrechetInceptionDistance; "
        "from torchmetrics.image.inception import InceptionScore; "
        "FrechetInceptionDistance(normalize=False); "
        "InceptionScore(normalize=False)"
    )
    print("  Downloading/verifying pretrained Inception metric weights ...")
    run([str(VENV_PYTHON), "-c", code])
    info("FID/Inception Score assets are ready")


# ---------------------------------------------------------------------------
# Step 6 — Runtime directories
# ---------------------------------------------------------------------------

RUNTIME_DIRS = [
    "data/raw",
    "results",
    "results/metrics",
]


def create_dirs() -> None:
    head("Step 7 - Runtime directories")
    for d in RUNTIME_DIRS:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        info(f"{'(exists)' if path.exists() else 'created'} {d}/")


# ---------------------------------------------------------------------------
# Step 7 — Dataset download
# ---------------------------------------------------------------------------

def download_datasets(selection: str) -> None:
    """Download datasets through torchvision into the project's data directory."""
    head("Step 8 - Dataset download")
    if selection == "none":
        info("Dataset download skipped")
        return

    dataset_root = ROOT / "data" / "raw"
    selected = ("cifar10", "celeba") if selection == "all" else (selection,)
    commands = {
        "cifar10": (
            "from torchvision import datasets; "
            f"root = {str(dataset_root)!r}; "
            "datasets.CIFAR10(root=root, train=True, download=True); "
            "datasets.CIFAR10(root=root, train=False, download=True)"
        ),
        "celeba": (
            "from torchvision import datasets; "
            f"root = {str(dataset_root)!r}; "
            "datasets.CelebA(root=root, split='train', download=True); "
            "datasets.CelebA(root=root, split='valid', download=True)"
        ),
    }
    sizes = {"cifar10": "~170 MB", "celeba": "~1.4 GB"}

    for dataset in selected:
        print(f"  Downloading/verifying {dataset} ({sizes[dataset]}) ...")
        try:
            run([str(VENV_PYTHON), "-c", commands[dataset]])
            info(f"{dataset} is ready in data/raw/")
        except subprocess.CalledProcessError:
            error(f"Could not download {dataset}.")
            if dataset == "celeba":
                warn("CelebA downloads can be blocked by Google Drive quotas.")
                warn("See README.md for the manual-download fallback.")
            raise


# ---------------------------------------------------------------------------
# Step 8 — Summary
# ---------------------------------------------------------------------------

def print_summary(gpu_type: str, datasets: str) -> None:
    head("Setup complete!")
    print()

    python_cmd = (
        r"venv\Scripts\python.exe" if IS_WINDOWS else "venv/bin/python"
    )

    gpu_label = {
        "cuda118": "CUDA 11.8 (NVIDIA)",
        "cuda121": "CUDA 12.1 (NVIDIA)",
        "cuda128": "CUDA 12.8 (NVIDIA)",
        "rocm":    "ROCm (AMD)",
        "cpu":     "CPU only",
    }[gpu_type]

    print(_c("  Hardware target:", "1") + f" {gpu_label}")
    print(_c("  Dataset(s):", "1") + f" {datasets}")
    print()
    print(_c("  Next steps:", "1"))
    print("  # Beginner training menu:")
    if IS_WINDOWS:
        print(r"  scripts\windows\train.cmd                         # double-click or run")
    else:
        print("  ./scripts/linux/train.sh")
    print()
    print("  # Quick smoke test (2 epochs, no GPU required):")
    print(f"  {python_cmd} train.py --algorithm mock --config config/smoke_fast.json --mode fresh")
    print()
    print("  # Train all 6 algorithms (CIFAR-10):")
    print("  ./scripts/linux/train_cifar.sh             # Linux")
    print(r"  scripts\windows\train_cifar.cmd             # Windows")
    print()
    print("  # Preview the complete two-dataset tournament (no training):")
    print("  ./scripts/linux/run_full_tournament.sh --dry-run")
    print(r"  .\scripts\windows\run_full_tournament.ps1 -DryRun")
    print()
    print("  # Start the inference UI:")
    if IS_WINDOWS:
        print(r"  .\scripts\windows\run_inference.ps1")
    else:
        print("  ./scripts/linux/run_inference.sh")
    print("  -> Open http://127.0.0.1:8000 in a browser")
    print()
    print(_c("  Docs:", "1") + " README.md                    (setup + commands)")
    print(_c("  Theory:", "1") + " docs/THEORY_NOTES.md         (maths + citations)")
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
        "--gpu", choices=["cuda118", "cuda121", "cuda128", "rocm", "cpu"], default=None,
        help="Force a specific PyTorch build instead of auto-detecting."
    )
    p.add_argument(
        "--skip-torch", action="store_true",
        help="Skip PyTorch installation (use if already installed in the venv)."
    )
    p.add_argument(
        "--datasets", "--dataset", dest="datasets",
        choices=["cifar10", "celeba", "all", "none"], default="cifar10",
        help=("Dataset download selection (default: cifar10). 'celeba' is ~1.4 GB; "
              "use 'all' for both or 'none' to skip downloads.")
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(_c("=" * 58, "36;1"))
    print(_c("  DiffusionProject - Bootstrap", "36;1"))
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
                "  Enter GPU type [cuda118 / cuda121 / cuda128 / rocm / cpu]: "
            ).strip()

    if not args.skip_torch:
        gpu_type = install_torch(gpu_type)

    install_requirements()
    prefetch_metric_assets()
    create_dirs()
    download_datasets(args.datasets)
    print_summary(gpu_type, args.datasets)


if __name__ == "__main__":
    main()
