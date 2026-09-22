#!/usr/bin/env python3
"""CLI for the shared GPU lock and frozen source identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.gpu_lock import acquire_gpu_lock, recover_stale_lock
from utils.source_identity import (
    build_source_manifest,
    verify_source_manifest,
    write_source_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    acquire = sub.add_parser("lock-acquire")
    acquire.add_argument("--lock-file", type=Path, default=Path("results/.lock"))
    acquire.add_argument("--command", required=True)
    acquire.add_argument("--token")
    acquire.add_argument("--owner-pid", type=int)
    recover = sub.add_parser("lock-recover")
    recover.add_argument("--lock-file", type=Path, default=Path("results/.lock"))
    release = sub.add_parser("lock-release")
    release.add_argument("--lock-file", type=Path, default=Path("results/.lock"))
    release.add_argument("--token", required=True)
    freeze = sub.add_parser("source-freeze")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--config", action="append", default=[])
    freeze.add_argument("--launcher", action="append", default=[])
    verify = sub.add_parser("source-verify")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "lock-acquire":
        lease = acquire_gpu_lock(
            PROJECT_ROOT / args.lock_file if not args.lock_file.is_absolute() else args.lock_file,
            command=args.command,
            inherited_token=args.token,
            owner_pid=args.owner_pid,
        )
        print(f"{lease.token}\t{'true' if lease.owned else 'false'}")
    elif args.action == "lock-release":
        path = PROJECT_ROOT / args.lock_file if not args.lock_file.is_absolute() else args.lock_file
        from utils.gpu_lock import GpuLockLease
        GpuLockLease(path.resolve(), args.token, owned=True).release()
    elif args.action == "lock-recover":
        path = PROJECT_ROOT / args.lock_file if not args.lock_file.is_absolute() else args.lock_file
        print("removed" if recover_stale_lock(path) else "absent")
    elif args.action == "source-freeze":
        manifest = build_source_manifest(
            PROJECT_ROOT, configs=args.config, launchers=args.launcher
        )
        path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
        write_source_manifest(path, manifest)
        print(manifest["source_identity_sha256"])
    else:
        path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
        manifest = json.loads(path.read_text(encoding="utf-8"))
        verify_source_manifest(PROJECT_ROOT, manifest)
        print(manifest["source_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
