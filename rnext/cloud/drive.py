"""Safe Google Drive (rclone) storage adapter for heavy FLAC/WAV masters and outputs.

Safety invariants:
  - ``dry_run=True`` by default on all write/transfer operations.
  - Non-destructive only: uses ``rclone copy`` / ``rclone copyto`` and
    ``rclone check --one-way``. Never invokes ``move``, ``sync``, ``delete``,
    or ``purge``, and never removes local originals.
  - Explicit retry policy: ``--retries 1 --low-level-retries 1`` (no hidden retries).
  - Blocks secrets, cookies, keys, and ``.git`` directories anywhere in the
    target path or its parents/descendants.
  - Portable: zero hardcoded machine paths; resolves ``rclone`` via explicit
    argument, ``MR_RCLONE_BIN``, ``PATH``, or ``Path.home() / '.local/bin/rclone'``.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REMOTE_BASE = "gdrive:darkfactory-storage/music-restoration"
_RCLONE_NO_RETRY_FLAGS = ("--retries", "1", "--low-level-retries", "1")

_BLOCKED_DIR_NAMES = {".git", ".secrets", ".ssh", ".gnupg"}
_BLOCKED_EXTENSIONS = {".pem", ".key", ".p12", ".pfx", ".kdbx"}
_BLOCKED_NAME_TOKENS = ("cookie", "secret", "token", "credential", "password")


class DriveError(RuntimeError):
    """Raised on invalid remote paths, blocked secrets, missing rclone, or failed transfers."""


@dataclass(frozen=True)
class DriveCheckResult:
    """Result of an ``rclone check --one-way`` integrity verification."""

    ok: bool
    remote_uri: str
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class DriveTransferResult:
    """Result (or dry-run plan) of a non-destructive Google Drive transfer."""

    direction: str
    local_path: str
    remote_uri: str
    dry_run: bool
    executed: bool
    commands: list[list[str]] = field(default_factory=list)
    verified: bool | None = None
    check_result: DriveCheckResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "local_path": self.local_path,
            "remote_uri": self.remote_uri,
            "dry_run": self.dry_run,
            "executed": self.executed,
            "commands": [shlex.join(cmd) for cmd in self.commands],
            "verified": self.verified,
        }


def default_remote_base() -> str:
    """Return configured remote base from ``MR_RCLONE_REMOTE`` or the default."""
    return os.environ.get("MR_RCLONE_REMOTE", "").strip() or DEFAULT_REMOTE_BASE


def find_rclone_bin(explicit: str | None = None) -> str | None:
    """Locate the ``rclone`` binary without relying on machine-specific paths."""
    if explicit:
        p = Path(explicit).expanduser()
        return str(p) if p.is_file() else None
    env_bin = os.environ.get("MR_RCLONE_BIN", "").strip()
    if env_bin:
        p = Path(env_bin).expanduser()
        return str(p) if p.is_file() else None
    found = shutil.which("rclone")
    if found:
        return found
    local_bin = Path.home() / ".local" / "bin" / "rclone"
    if local_bin.is_file():
        return str(local_bin)
    return None


def _require_rclone_bin(explicit: str | None = None) -> str:
    rclone_bin = find_rclone_bin(explicit)
    if not rclone_bin:
        raise DriveError(
            "rclone binary not found (checked --rclone-bin, MR_RCLONE_BIN, PATH, ~/.local/bin/rclone)"
        )
    return rclone_bin


def validate_remote_subpath(subpath: str) -> str:
    """Validate and normalize a relative remote subpath (no traversal or remote injection)."""
    raw = (subpath or "").strip()
    if not raw:
        raise DriveError("remote_subpath must not be empty")
    if "\\" in raw or ":" in raw or "\x00" in raw:
        raise DriveError(f"invalid characters in remote_subpath: {subpath!r}")
    parts = [part for part in raw.split("/") if part]
    if not parts:
        raise DriveError("remote_subpath must not be empty")
    for part in parts:
        if part in (".", ".."):
            raise DriveError(f"path traversal is forbidden in remote_subpath: {subpath!r}")
    return "/".join(parts)


def build_remote_uri(remote_subpath: str, remote_base: str | None = None) -> str:
    """Join ``remote_base`` and a validated ``remote_subpath``."""
    base = (remote_base or default_remote_base()).rstrip("/")
    if ":" not in base:
        raise DriveError(f"remote_base must include an rclone remote prefix (e.g. 'gdrive:...'): {base!r}")
    clean_sub = validate_remote_subpath(remote_subpath)
    return f"{base}/{clean_sub}"


def _is_blocked_name(name: str) -> bool:
    low = name.lower()
    if low in _BLOCKED_DIR_NAMES or low == ".env" or low.startswith(".env."):
        return True
    if Path(low).suffix in _BLOCKED_EXTENSIONS:
        return True
    return any(tok in low for tok in _BLOCKED_NAME_TOKENS)


def assert_safe_local_path(local_path: Path) -> None:
    """Refuse to transfer secrets, cookies, keys, or git metadata (including parent dirs)."""
    target = Path(local_path).expanduser().resolve()
    if not target.exists():
        raise DriveError(f"local path does not exist: {target}")
    for part in target.parts:
        if _is_blocked_name(part):
            raise DriveError(f"refusing to transfer sensitive path component: {part}")
    if target.is_dir():
        for child in target.rglob("*"):
            rel_parts = child.relative_to(target).parts
            for part in rel_parts:
                if _is_blocked_name(part):
                    raise DriveError(
                        f"refusing to transfer directory containing sensitive entry: {'/'.join(rel_parts)}"
                    )


def _escape_rclone_filter_name(name: str) -> str:
    """Escape rclone glob metacharacters so ``--include`` matches a literal filename."""
    return re.sub(r"([*?\[\]{}\\])", r"\\\1", name)


def _single_file_remote_layout(target: Path, remote_uri: str) -> tuple[bool, str]:
    """Return ``(is_exact_file_uri, remote_parent_dir)`` consistently for upload and verify."""
    if remote_uri.endswith("/" + target.name):
        return True, remote_uri.rsplit("/", 1)[0]
    return False, remote_uri


def _build_check_cmd(bin_path: str, target: Path, remote_uri: str) -> list[str]:
    if target.is_file():
        _, remote_parent = _single_file_remote_layout(target, remote_uri)
        escaped = _escape_rclone_filter_name(target.name)
        return [
            bin_path,
            "check",
            "--one-way",
            *_RCLONE_NO_RETRY_FLAGS,
            "--include",
            f"/{escaped}",
            "--exclude",
            "*",
            str(target.parent),
            remote_parent,
        ]
    return [bin_path, "check", "--one-way", *_RCLONE_NO_RETRY_FLAGS, str(target), remote_uri]


def _check_output_matched_files(target: Path, stderr: str) -> bool:
    """Ensure ``rclone check`` did not silently pass with 0 matched files."""
    low = (stderr or "").lower()
    return not ("0 matching files" in low and (target.is_file() or any(target.rglob("*"))))


def verify_one_way(
    local_path: Path,
    remote_subpath: str,
    *,
    remote_base: str | None = None,
    rclone_bin: str | None = None,
    runner: Any = subprocess.run,
) -> DriveCheckResult:
    """Verify local files exist with matching checksums on remote via ``rclone check --one-way``."""
    target = Path(local_path).expanduser().resolve()
    assert_safe_local_path(target)
    remote_uri = build_remote_uri(remote_subpath, remote_base=remote_base)
    bin_path = _require_rclone_bin(rclone_bin)
    cmd = _build_check_cmd(bin_path, target, remote_uri)

    proc = runner(cmd, capture_output=True, text=True)
    stderr = getattr(proc, "stderr", "") or ""
    ok = (int(proc.returncode) == 0) and _check_output_matched_files(target, stderr)
    return DriveCheckResult(
        ok=ok,
        remote_uri=remote_uri,
        command=cmd,
        returncode=int(proc.returncode),
        stdout=getattr(proc, "stdout", "") or "",
        stderr=stderr,
    )


def upload_path(
    local_path: Path,
    remote_subpath: str,
    *,
    remote_base: str | None = None,
    dry_run: bool = True,
    verify: bool = True,
    rclone_bin: str | None = None,
    runner: Any = subprocess.run,
) -> DriveTransferResult:
    """Non-destructively copy a local file or directory to Google Drive.

    Defaults to ``dry_run=True``. Local files are never deleted or modified.
    When ``dry_run=False`` and ``verify=True``, runs ``rclone check --one-way``
    after upload and raises ``DriveError`` if verification fails.
    """
    target = Path(local_path).expanduser().resolve()
    assert_safe_local_path(target)
    remote_uri = build_remote_uri(remote_subpath, remote_base=remote_base)
    bin_path = find_rclone_bin(rclone_bin) if dry_run else _require_rclone_bin(rclone_bin)
    bin_display = bin_path or "rclone"

    if target.is_file():
        is_exact, _ = _single_file_remote_layout(target, remote_uri)
        op = "copyto" if is_exact else "copy"
        copy_cmd = [bin_display, op, *_RCLONE_NO_RETRY_FLAGS, str(target), remote_uri]
    else:
        copy_cmd = [bin_display, "copy", *_RCLONE_NO_RETRY_FLAGS, str(target), remote_uri]

    check_cmd = _build_check_cmd(bin_display, target, remote_uri)
    planned_cmds = [copy_cmd] + ([check_cmd] if verify else [])
    if dry_run:
        return DriveTransferResult(
            direction="upload",
            local_path=str(target),
            remote_uri=remote_uri,
            dry_run=True,
            executed=False,
            commands=planned_cmds,
            verified=None,
        )

    proc = runner(copy_cmd, capture_output=True, text=True)
    if int(proc.returncode) != 0:
        raise DriveError(
            f"rclone upload failed (exit {proc.returncode}) for {remote_uri}: "
            f"{(getattr(proc, 'stderr', '') or '').strip()[:400]}"
        )

    check_res: DriveCheckResult | None = None
    verified: bool | None = None
    if verify:
        check_proc = runner(check_cmd, capture_output=True, text=True)
        check_stderr = getattr(check_proc, "stderr", "") or ""
        check_ok = (int(check_proc.returncode) == 0) and _check_output_matched_files(target, check_stderr)
        check_res = DriveCheckResult(
            ok=check_ok,
            remote_uri=remote_uri,
            command=check_cmd,
            returncode=int(check_proc.returncode),
            stdout=getattr(check_proc, "stdout", "") or "",
            stderr=check_stderr,
        )
        verified = check_res.ok
        if not check_res.ok:
            raise DriveError(
                f"rclone check --one-way failed (exit {check_res.returncode}) for {remote_uri}: "
                f"{check_res.stderr.strip()[:400]}"
            )

    return DriveTransferResult(
        direction="upload",
        local_path=str(target),
        remote_uri=remote_uri,
        dry_run=False,
        executed=True,
        commands=planned_cmds,
        verified=verified,
        check_result=check_res,
    )


def download_path(
    remote_subpath: str,
    local_dest: Path,
    *,
    remote_base: str | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    rclone_bin: str | None = None,
    runner: Any = subprocess.run,
) -> DriveTransferResult:
    """Non-destructively copy from Google Drive into ``local_dest``.

    Defaults to ``dry_run=True`` and ``overwrite=False`` (``--ignore-existing``)
    so local originals are never overwritten or deleted.
    """
    dest = Path(local_dest).expanduser().resolve()
    remote_uri = build_remote_uri(remote_subpath, remote_base=remote_base)
    bin_path = find_rclone_bin(rclone_bin) if dry_run else _require_rclone_bin(rclone_bin)
    bin_display = bin_path or "rclone"

    cmd = [bin_display, "copy", *_RCLONE_NO_RETRY_FLAGS]
    if not overwrite:
        cmd.append("--ignore-existing")
    cmd.extend([remote_uri, str(dest)])

    if dry_run:
        return DriveTransferResult(
            direction="download",
            local_path=str(dest),
            remote_uri=remote_uri,
            dry_run=True,
            executed=False,
            commands=[cmd],
            verified=None,
        )

    dest.mkdir(parents=True, exist_ok=True)
    proc = runner(cmd, capture_output=True, text=True)
    if int(proc.returncode) != 0:
        raise DriveError(
            f"rclone download failed (exit {proc.returncode}) for {remote_uri}: "
            f"{(getattr(proc, 'stderr', '') or '').strip()[:400]}"
        )
    return DriveTransferResult(
        direction="download",
        local_path=str(dest),
        remote_uri=remote_uri,
        dry_run=False,
        executed=True,
        commands=[cmd],
        verified=None,
    )


def list_remote(
    remote_subpath: str = "",
    *,
    remote_base: str | None = None,
    rclone_bin: str | None = None,
    runner: Any = subprocess.run,
) -> list[dict[str, Any]]:
    """List remote entries via ``rclone lsjson``."""
    base = (remote_base or default_remote_base()).rstrip("/")
    remote_uri = (
        build_remote_uri(remote_subpath, remote_base=base)
        if remote_subpath.strip()
        else base
    )
    bin_path = _require_rclone_bin(rclone_bin)
    cmd = [bin_path, "lsjson", *_RCLONE_NO_RETRY_FLAGS, remote_uri]
    proc = runner(cmd, capture_output=True, text=True)
    if int(proc.returncode) != 0:
        raise DriveError(
            f"rclone lsjson failed (exit {proc.returncode}) for {remote_uri}: "
            f"{(getattr(proc, 'stderr', '') or '').strip()[:400]}"
        )
    raw = (getattr(proc, "stdout", "") or "").strip() or "[]"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DriveError(f"rclone lsjson returned invalid JSON for {remote_uri}") from exc
    if not isinstance(data, list):
        raise DriveError(f"rclone lsjson expected a JSON list for {remote_uri}")
    return data
