"""Archive a completed V1H run to STORE, omitting only its dataset directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile


def archive_run(run_dir: Path, store: Path) -> dict:
    run_dir, store = run_dir.resolve(strict=True), store.resolve(strict=True)
    if not run_dir.is_dir() or not store.is_dir():
        raise ValueError("Run and STORE must be directories")
    if store == run_dir or run_dir in store.parents:
        raise ValueError("STORE must be outside the run directory")
    destination = store / f"{run_dir.name}.tar"
    partial = store / f"{run_dir.name}.tar.partial"
    checksum = store / f"{run_dir.name}.tar.sha256"
    if destination.exists() or checksum.exists():
        raise FileExistsError(f"Archive already exists: {destination}")

    print(f"Archiving {run_dir} to {destination}; excluding dataset/", flush=True)
    # Avoid writing to our own Slurm log until it has been added to the tar.
    # Do not dereference symlinks: dataset/cache links must never copy raw data.
    prefix = run_dir.name + "/dataset"

    def retained(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if member.name == prefix or member.name.startswith(prefix + "/") else member

    with partial.open("xb") as handle:
        with tarfile.open(fileobj=handle, mode="w", dereference=False) as archive:
            archive.add(run_dir, arcname=run_dir.name, filter=retained)
        handle.flush()
        os.fsync(handle.fileno())

    digest = hashlib.sha256()
    with partial.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    with tarfile.open(partial, "r:") as archive:
        members = archive.getmembers()
        if not members or any(m.name == prefix or m.name.startswith(prefix + "/") for m in members):
            raise ValueError("Archive inventory is empty or contains the excluded dataset")

    # Publish without overwriting an earlier archive, even with concurrent jobs.
    os.link(partial, destination)
    partial.unlink()
    with checksum.open("x") as handle:
        handle.write(f"{digest.hexdigest()}  {destination.name}\n")
    receipt = {
        "archive": str(destination),
        "sha256": digest.hexdigest(),
        "size_bytes": destination.stat().st_size,
        "members": len(members),
        "excluded": ["dataset/"],
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_text = json.dumps(receipt, indent=2) + "\n"
    (run_dir / "archive_receipt.json").write_text(receipt_text)
    with (store / f"{run_dir.name}.archive.json").open("x") as handle:
        handle.write(receipt_text)
    print(receipt_text, end="", flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    archive_run(args.run_dir, args.store)
