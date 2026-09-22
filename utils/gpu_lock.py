"""Ownership-checked, nestable project GPU lock.

The lock is deliberately independent of Git.  A random token identifies the
owner, while PID plus Linux process-start ticks distinguish a live process from
a reused PID.  Child processes may inherit the exact token through
``DIFFUSION_GPU_LOCK_TOKEN``; they never remove the parent's lock.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


LOCK_ENV = "DIFFUSION_GPU_LOCK_TOKEN"
DEFAULT_LOCK_PATH = Path("results/.lock")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GpuLockError(RuntimeError):
    """Base class for safe lock failures."""


class ActiveGpuLockError(GpuLockError):
    """Raised when another live workflow owns the GPU lock."""


class StaleGpuLockError(GpuLockError):
    """Raised when explicit operator recovery is required."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_start_ticks(pid: int) -> Optional[int]:
    """Return Linux /proc start ticks, or None on unsupported platforms."""
    try:
        # The command name in /proc/<pid>/stat may contain spaces and ')'.
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, PermissionError, IndexError, ValueError, OSError):
        return None


def _pid_exists(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def owner_is_active(payload: dict[str, Any]) -> bool:
    """Return whether the recorded owner still denotes the same live process."""
    if payload.get("hostname") != socket.gethostname():
        # A shared filesystem cannot safely prove a remote host is dead.
        return True
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if not _pid_exists(pid):
        return False
    recorded = payload.get("process_start_ticks")
    current = _process_start_ticks(pid)
    return recorded is None or current is None or int(recorded) == current


def read_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise StaleGpuLockError(
            f"GPU lock is unreadable and must be inspected manually: {path}\n"
            f"After confirming no GPU workflow is active, run:\n"
            f"  python scripts/workflow_guard.py lock-recover --lock-file {path}"
        ) from exc
    if not isinstance(payload, dict) or not payload.get("token"):
        raise StaleGpuLockError(f"GPU lock has invalid metadata: {path}")
    return payload


@dataclass
class GpuLockLease:
    path: Path
    token: str
    owned: bool

    def release(self) -> bool:
        """Remove the lock only if this lease created and still owns it."""
        if not self.owned:
            return False
        try:
            payload = read_lock(self.path)
        except FileNotFoundError:
            self.owned = False
            return False
        if payload.get("token") != self.token:
            self.owned = False
            raise GpuLockError(
                f"Refusing to remove GPU lock now owned by another token: {self.path}"
            )
        self.path.unlink()
        self.owned = False
        return True

    def __enter__(self) -> "GpuLockLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def acquire_gpu_lock(
    path: Path | str = DEFAULT_LOCK_PATH,
    *,
    command: str,
    inherited_token: Optional[str] = None,
    recover_stale: bool = False,
    owner_pid: Optional[int] = None,
) -> GpuLockLease:
    """Acquire a new lock or validate explicit nested ownership."""
    requested = Path(path).expanduser()
    lock_path = (requested if requested.is_absolute() else PROJECT_ROOT / requested).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    inherited_token = inherited_token or os.environ.get(LOCK_ENV)

    if lock_path.exists():
        payload = read_lock(lock_path)
        if inherited_token and payload.get("token") == inherited_token:
            return GpuLockLease(lock_path, inherited_token, owned=False)
        owner = (
            f"pid={payload.get('pid')} host={payload.get('hostname')} "
            f"command={payload.get('command')} acquired={payload.get('acquired_utc')}"
        )
        if owner_is_active(payload):
            raise ActiveGpuLockError(
                f"GPU workflow lock is actively owned: {lock_path}\n  {owner}"
            )
        if not recover_stale:
            raise StaleGpuLockError(
                f"Stale GPU workflow lock found: {lock_path}\n  {owner}\n"
                "After confirming no trainer, evaluator, generator, codec, or suite "
                "process is active, recover it explicitly with:\n"
                f"  python scripts/workflow_guard.py lock-recover --lock-file {lock_path}"
            )
        lock_path.unlink()

    token = uuid.uuid4().hex
    recorded_pid = owner_pid or os.getpid()
    payload = {
        "schema_version": 1,
        "token": token,
        "pid": recorded_pid,
        "process_start_ticks": _process_start_ticks(recorded_pid),
        "hostname": socket.gethostname(),
        "command": command,
        "acquired_utc": _utc_now(),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError:
        return acquire_gpu_lock(
            lock_path, command=command, inherited_token=inherited_token,
            recover_stale=False,
        )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return GpuLockLease(lock_path, token, owned=True)


def recover_stale_lock(path: Path | str = DEFAULT_LOCK_PATH) -> bool:
    """Remove only a lock whose recorded process is provably stale."""
    requested = Path(path).expanduser()
    lock_path = (requested if requested.is_absolute() else PROJECT_ROOT / requested).resolve()
    if not lock_path.exists():
        return False
    payload = read_lock(lock_path)
    if owner_is_active(payload):
        raise ActiveGpuLockError(
            f"Refusing stale recovery because the owner is active: {lock_path}"
        )
    lock_path.unlink()
    return True
