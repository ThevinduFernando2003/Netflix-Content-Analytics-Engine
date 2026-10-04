"""Content-addressed stage cache. A stage is skipped when inputs, config and code are unchanged."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.logging_conf import get_logger

log = get_logger(__name__)

SUCCESS_NAME = "_SUCCESS"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_parquet_excluding(path: Path, exclude: Iterable[str]) -> str:
    """Hash column values, skipping columns a later stage writes back into the same file.

    Arrow buffer bytes are not used: rewriting the file can change encoding while the
    values stay the same, and that would invalidate the cache on every run.
    """
    if not path.exists():
        return "missing"
    import pandas as pd

    frame = pd.read_parquet(path)
    drop = set(exclude)
    names = [name for name in frame.columns if name not in drop]
    digest = hashlib.sha256()
    digest.update("\0".join(names).encode())
    hashed = pd.util.hash_pandas_object(frame[names], index=False).to_numpy()
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(sha256_file(child).encode())
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def config_hash(cfg: AppConfig) -> str:
    payload = json.dumps(
        {
            "base": cfg.base,
            "features": cfg.features,
            "models": cfg.models,
            "insights": cfg.insights,
        },
        sort_keys=True,
        default=str,
    )
    return sha256_text(payload)


def code_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.as_posix()):
        if path.exists() and path.is_file():
            digest.update(path.as_posix().encode())
            digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def cache_key(input_hashes: dict[str, str], cfg_hash: str, src_hash: str) -> str:
    payload = json.dumps(
        {"inputs": input_hashes, "config": cfg_hash, "code": src_hash},
        sort_keys=True,
    )
    return sha256_text(payload)


def success_path(output_dir: Path) -> Path:
    return output_dir / SUCCESS_NAME


def read_manifest(output_dir: Path) -> dict[str, Any] | None:
    path = success_path(output_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = success_path(output_dir)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def remove_success(output_dir: Path) -> None:
    path = success_path(output_dir)
    if path.exists():
        path.unlink()


def is_cache_hit(output_dir: Path, expected_key: str) -> bool:
    manifest = read_manifest(output_dir)
    if not manifest:
        return False
    return manifest.get("cache_key") == expected_key
