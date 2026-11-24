"""Cloud GPU orchestration over ``colab`` CLI.

Design guarantees:
  - Strict ``dry_run`` mode: validates inputs and prints exact planned steps
    without starting a Colab VM or spending Compute Units.
  - Hard auto-stop guarantee: ``run_colab_job`` stops the VM in a ``finally``
    block (checking ``colab stop`` exit code) and verifies via
    ``colab --auth <auth> sessions`` that neither our session nor any unlabelled
    (``?``) active VM remains running. Never returns 0 if teardown is unconfirmed.
  - Never deletes prior run artifacts in ``out_dir``: refuses to overwrite
    existing output audio or ``worker_report.json`` so previous results are safe.
  - Uploads ``rnext/features.py`` to ``/content/rnext_features.py`` so the GPU
    worker reuses the repository's single audio decoder (one-owner rule).
  - Requires a fresh ``worker_report.json`` with ``status == "ok"``,
    ``cuda_available is True``, and matching input/output SHA-256 hashes.
  - Uploads ONLY the explicit artifacts produced by the current run (restored
    audio, ``worker_report.json``, ``manifest.json``) — never arbitrary files
    sitting in ``out_dir``.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rnext.cloud.colab_worker import VALID_TASKS, render_worker_script
from rnext.cloud.drive import DriveTransferResult, build_remote_uri, upload_path
from rnext.cloud.metrics import (
    build_manifest,
    measure_audio_file,
    sha256_file,
    write_manifest,
)

VALID_GPUS = ("T4", "L4", "G4", "A100", "H100")
VALID_AUTHS = ("adc", "oauth2")
DEFAULT_GPU = "L4"
DEFAULT_AUTH = "adc"
SUPPORTED_AUDIO_EXT = {".wav", ".flac", ".aiff", ".mp3", ".m4a", ".ogg", ".opus"}
_UNLABELLED_SESSION_RE = re.compile(r"(?:^|\s|\|)\?(?:\s|\||$)", re.MULTILINE)


class ColabError(RuntimeError):
    """Raised when Colab configuration, execution, verification, or teardown fails."""


@dataclass(frozen=True)
class ColabJobConfig:
    """Configuration for a single cloud GPU audio restoration job."""

    input_path: Path
    out_dir: Path
    task: str = "gpu_smoke"
    gpu: str = DEFAULT_GPU
    auth: str = DEFAULT_AUTH
    session: str | None = None
    timeout_s: float = 600.0
    high_mem: bool = False
    keep_session: bool = False
    dry_run: bool = False
    drive_subpath: str | None = None
    drive_remote_base: str | None = None
    colab_bin: str | None = None
    rclone_bin: str | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ColabPlanStep:
    """Single planned CLI step in a Colab GPU job."""

    stage: str
    cmd: list[str]
    always: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "command": shlex.join(self.cmd),
            "always": self.always,
        }


@dataclass(frozen=True)
class ColabJobPlan:
    """Validated, deterministic execution plan for a Colab GPU job."""

    task: str
    gpu: str
    auth: str
    session: str
    input_path: str
    output_filename: str
    out_dir: str
    worker_script_path: str
    steps: list[ColabPlanStep]
    post_check_cmd: list[str]
    drive_subpath: str | None = None
    keep_session: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "gpu": self.gpu,
            "auth": self.auth,
            "session": self.session,
            "input_name": Path(self.input_path).name,
            "output_filename": self.output_filename,
            "out_dir": self.out_dir,
            "keep_session": self.keep_session,
            "drive_subpath": self.drive_subpath,
            "steps": [step.to_dict() for step in self.steps],
            "post_check_command": shlex.join(self.post_check_cmd),
        }


@dataclass(frozen=True)
class ColabJobResult:
    """Result of a Colab GPU job (or dry-run plan)."""

    executed: bool
    dry_run: bool
    plan: ColabJobPlan
    returncode: int = 0
    output_audio_path: str | None = None
    manifest_path: str | None = None
    manifest: dict[str, Any] | None = None
    sessions_clean_after: bool | None = None
    drive_transfers: list[DriveTransferResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "dry_run": self.dry_run,
            "returncode": self.returncode,
            "plan": self.plan.to_dict(),
            "output_audio_name": Path(self.output_audio_path).name if self.output_audio_path else None,
            "manifest_path": self.manifest_path,
            "sessions_clean_after": self.sessions_clean_after,
            "drive_transfers": [t.to_dict() for t in self.drive_transfers],
            "manifest": self.manifest,
        }


def find_colab_bin(explicit: str | None = None) -> str | None:
    """Locate the ``colab`` CLI binary without hardcoded machine paths."""
    if explicit:
        p = Path(explicit).expanduser()
        return str(p) if p.is_file() else None
    env_bin = os.environ.get("MR_COLAB_BIN", "").strip()
    if env_bin:
        p = Path(env_bin).expanduser()
        return str(p) if p.is_file() else None
    found = shutil.which("colab")
    if found:
        return found
    local_bin = Path.home() / ".local" / "bin" / "colab"
    if local_bin.is_file():
        return str(local_bin)
    return None


def _features_module_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "features.py"
    if not path.is_file():
        raise ColabError(f"rnext/features.py not found at expected location: {path}")
    return path


def _sanitize_session_name(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-")
    if not cleaned:
        raise ColabError(f"invalid Colab session name: {raw!r}")
    return cleaned[:32]


def build_colab_job_plan(config: ColabJobConfig) -> ColabJobPlan:
    """Validate ``config`` and construct the exact ``colab`` session execution plan."""
    if config.task not in VALID_TASKS:
        raise ColabError(f"unsupported task {config.task!r}; expected one of {VALID_TASKS}")
    if config.gpu not in VALID_GPUS:
        raise ColabError(f"unsupported GPU {config.gpu!r}; expected one of {VALID_GPUS}")
    if config.auth not in VALID_AUTHS:
        raise ColabError(f"unsupported auth strategy {config.auth!r}; expected one of {VALID_AUTHS}")
    if config.timeout_s < 0:
        raise ColabError(f"timeout_s must be >= 0, got {config.timeout_s}")

    input_path = Path(config.input_path).expanduser().resolve()
    if not input_path.is_file():
        raise ColabError(f"input audio file not found: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_AUDIO_EXT:
        raise ColabError(
            f"unsupported audio extension {input_path.suffix!r}; expected one of {sorted(SUPPORTED_AUDIO_EXT)}"
        )

    if config.drive_subpath:
        build_remote_uri(config.drive_subpath, remote_base=config.drive_remote_base)

    out_dir = Path(config.out_dir).expanduser().resolve()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    session = _sanitize_session_name(config.session or f"mr-{config.task[:10]}-{ts}")
    output_filename = f"{input_path.stem}_{config.task}.wav"
    worker_script_path = out_dir / f"_mr_colab_worker_{config.task}.py"
    features_path = _features_module_path()

    colab_bin = find_colab_bin(config.colab_bin)
    if not config.dry_run and not colab_bin:
        raise ColabError(
            "colab CLI not found (checked --colab-bin, MR_COLAB_BIN, PATH, ~/.local/bin/colab)"
        )
    base = [colab_bin or "colab", "--auth", config.auth]

    new_cmd = [*base, "new", "--session", session, "--gpu", config.gpu]
    if config.high_mem:
        new_cmd.append("--high-mem")

    steps: list[ColabPlanStep] = [
        ColabPlanStep(stage="create_session", cmd=new_cmd),
        ColabPlanStep(
            stage="upload_features",
            cmd=[*base, "upload", "-s", session, str(features_path), "/content/rnext_features.py"],
        ),
        ColabPlanStep(
            stage="upload_input",
            cmd=[*base, "upload", "-s", session, str(input_path), f"/content/{input_path.name}"],
        ),
        ColabPlanStep(
            stage="exec_worker",
            cmd=[
                *base,
                "exec",
                "-s",
                session,
                "-f",
                str(worker_script_path),
                "--timeout",
                str(config.timeout_s),
            ],
        ),
        ColabPlanStep(
            stage="download_audio",
            cmd=[
                *base,
                "download",
                "-s",
                session,
                f"/content/mr_out/{output_filename}",
                str(out_dir / output_filename),
            ],
        ),
        ColabPlanStep(
            stage="download_report",
            cmd=[
                *base,
                "download",
                "-s",
                session,
                "/content/mr_out/worker_report.json",
                str(out_dir / "worker_report.json"),
            ],
        ),
    ]
    if not config.keep_session:
        steps.append(
            ColabPlanStep(
                stage="stop_session",
                cmd=[*base, "stop", "-s", session],
                always=True,
            )
        )

    post_check_cmd = [*base, "sessions"]

    return ColabJobPlan(
        task=config.task,
        gpu=config.gpu,
        auth=config.auth,
        session=session,
        input_path=str(input_path),
        output_filename=output_filename,
        out_dir=str(out_dir),
        worker_script_path=str(worker_script_path),
        steps=steps,
        post_check_cmd=post_check_cmd,
        drive_subpath=config.drive_subpath,
        keep_session=config.keep_session,
    )


def check_active_sessions(
    *,
    auth: str = DEFAULT_AUTH,
    colab_bin: str | None = None,
    runner: Any = subprocess.run,
) -> tuple[bool, str]:
    """Query ``colab --auth <auth> sessions`` and return ``(no_active_sessions, output)``.

    Raises ``ColabError`` if ``colab sessions`` returns a non-zero exit code.
    """
    bin_path = find_colab_bin(colab_bin)
    if not bin_path:
        raise ColabError("colab CLI not found")
    proc = runner([bin_path, "--auth", auth, "sessions"], capture_output=True, text=True)
    combined = ((getattr(proc, "stdout", "") or "") + "\n" + (getattr(proc, "stderr", "") or "")).strip()
    if int(proc.returncode) != 0:
        raise ColabError(f"colab sessions failed (exit {proc.returncode}): {combined[-400:]}")
    no_active = "no active sessions" in combined.lower()
    return no_active, combined


def _session_confirmed_absent(session: str, no_active: bool, output: str) -> bool:
    """Confirm ``session`` is absent AND no unlabelled (``?``) VM is running."""
    if no_active:
        return True
    if session in output:
        return False
    # `colab sessions` renders active VMs without a local session alias as `?`;
    # we cannot distinguish an unlabelled active VM from our own session.
    return not bool(_UNLABELLED_SESSION_RE.search(output))


def ensure_session_stopped(
    session: str,
    *,
    auth: str = DEFAULT_AUTH,
    colab_bin: str | None = None,
    runner: Any = subprocess.run,
) -> bool:
    """Confirm ``session`` is stopped; return ``False`` on CLI error, active session, or unlabelled ``?`` VM."""
    bin_path = find_colab_bin(colab_bin)
    if not bin_path:
        return False
    try:
        no_active, text = check_active_sessions(auth=auth, colab_bin=bin_path, runner=runner)
    except ColabError:
        return False
    if _session_confirmed_absent(session, no_active, text):
        return True
    stop_proc = runner([bin_path, "--auth", auth, "stop", "-s", session], capture_output=True, text=True)
    if int(stop_proc.returncode) != 0:
        return False
    try:
        no_active_after, text_after = check_active_sessions(auth=auth, colab_bin=bin_path, runner=runner)
    except ColabError:
        return False
    return _session_confirmed_absent(session, no_active_after, text_after)


def _validate_worker_report(
    report: dict[str, Any],
    *,
    expected_task: str,
    input_sha256: str,
    output_sha256: str,
) -> None:
    if report.get("status") != "ok":
        raise ColabError(f"worker_report.json did not report status='ok': {report.get('status')!r}")
    if report.get("task") != expected_task:
        raise ColabError(
            f"worker_report.json task mismatch: expected {expected_task!r}, got {report.get('task')!r}"
        )
    if report.get("cuda_available") is not True:
        raise ColabError("worker_report.json reported cuda_available != True on a GPU job")
    if report.get("input_sha256") != input_sha256:
        raise ColabError(
            f"worker_report.json input_sha256 mismatch: expected {input_sha256}, got {report.get('input_sha256')}"
        )
    if report.get("output_sha256") != output_sha256:
        raise ColabError(
            f"worker_report.json output_sha256 mismatch: expected {output_sha256}, got {report.get('output_sha256')}"
        )


def run_colab_job(
    config: ColabJobConfig,
    *,
    runner: Any = subprocess.run,
) -> ColabJobResult:
    """Execute (or dry-run) a Colab GPU restoration job with hard VM auto-stop confirmation."""
    plan = build_colab_job_plan(config)
    if config.dry_run:
        return ColabJobResult(
            executed=False,
            dry_run=True,
            plan=plan,
            returncode=0,
        )

    out_dir = Path(plan.out_dir)
    input_path = Path(plan.input_path)
    worker_script_path = Path(plan.worker_script_path)
    output_audio_path = out_dir / plan.output_filename
    worker_report_path = out_dir / "worker_report.json"

    # Refuse to overwrite prior run artifacts instead of deleting them prematurely.
    for existing in (output_audio_path, worker_report_path):
        if existing.exists():
            raise ColabError(
                f"refusing to overwrite existing output artifact: {existing} (use a fresh --out-dir per run)"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_before = measure_audio_file(input_path)
    input_digest = metrics_before["sha256"]

    worker_source = render_worker_script(
        task=plan.task,
        input_filename=input_path.name,
        output_filename=plan.output_filename,
        extra_options=config.extra_options,
    )
    worker_script_path.write_text(worker_source, encoding="utf-8")

    normal_steps = [s for s in plan.steps if not s.always]
    always_steps = [s for s in plan.steps if s.always]
    session_created = False
    stop_step_ok = True
    sessions_clean: bool | None = None

    try:
        for step in normal_steps:
            proc = runner(step.cmd, capture_output=True, text=True)
            rc = int(proc.returncode)
            if step.stage == "create_session" and rc == 0:
                session_created = True
            if rc != 0:
                stderr_s = (getattr(proc, "stderr", "") or "").strip()
                stdout_s = (getattr(proc, "stdout", "") or "").strip()
                detail = f"stderr={stderr_s[-800:]!r} stdout={stdout_s[-400:]!r}"
                raise ColabError(
                    f"Colab stage '{step.stage}' failed (exit {rc}) for session {plan.session}: {detail}"
                )
    finally:
        if session_created:
            for step in always_steps:
                stop_proc = runner(step.cmd, capture_output=True, text=True)
                if int(stop_proc.returncode) != 0:
                    stop_step_ok = False
        if not plan.keep_session:
            sessions_clean = stop_step_ok and ensure_session_stopped(
                plan.session,
                auth=plan.auth,
                colab_bin=config.colab_bin,
                runner=runner,
            )

    if not plan.keep_session and sessions_clean is not True:
        raise ColabError(
            f"could not confirm Colab session {plan.session!r} stopped; "
            f"verify immediately with `colab --auth {plan.auth} sessions`"
        )

    if not output_audio_path.is_file():
        raise ColabError(f"Colab job finished without expected output artifact: {output_audio_path}")
    if not worker_report_path.is_file():
        raise ColabError(f"Colab job finished without required worker_report.json: {worker_report_path}")

    try:
        colab_meta = json.loads(worker_report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ColabError(f"worker_report.json is not valid JSON: {worker_report_path}") from exc

    output_digest = sha256_file(output_audio_path)
    _validate_worker_report(
        colab_meta,
        expected_task=plan.task,
        input_sha256=input_digest,
        output_sha256=output_digest,
    )

    metrics_after = measure_audio_file(output_audio_path)

    drive_uri: str | None = (
        build_remote_uri(config.drive_subpath, remote_base=config.drive_remote_base)
        if config.drive_subpath
        else None
    )

    manifest_path = out_dir / "manifest.json"
    manifest = build_manifest(
        task=plan.task,
        gpu=plan.gpu,
        session=plan.session,
        input_path=input_path,
        output_paths=[output_audio_path, worker_report_path],
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        colab_metadata=colab_meta,
        drive_uri=drive_uri,
    )
    write_manifest(manifest_path, manifest)

    drive_transfers: list[DriveTransferResult] = []
    if config.drive_subpath:
        # Upload ONLY the explicit artifacts of this run (never arbitrary files in out_dir).
        base_sub = config.drive_subpath.rstrip("/")
        for artifact in (output_audio_path, worker_report_path, manifest_path):
            drive_transfers.append(
                upload_path(
                    artifact,
                    f"{base_sub}/{artifact.name}",
                    remote_base=config.drive_remote_base,
                    dry_run=False,
                    verify=True,
                    rclone_bin=config.rclone_bin,
                    runner=runner,
                )
            )

    return ColabJobResult(
        executed=True,
        dry_run=False,
        plan=plan,
        returncode=0,
        output_audio_path=str(output_audio_path),
        manifest_path=str(manifest_path),
        manifest=manifest,
        sessions_clean_after=sessions_clean,
        drive_transfers=drive_transfers,
    )
