"""Offline unit tests for rnext.cloud.colab_runner, colab_worker, and thin CLIs."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rnext.cloud.colab_runner import (
    ColabError,
    ColabJobConfig,
    build_colab_job_plan,
    check_active_sessions,
    ensure_session_stopped,
    run_colab_job,
)
from rnext.cloud.colab_worker import VALID_TASKS, render_worker_script
from rnext.cloud.metrics import sha256_file

REPO = Path(__file__).resolve().parents[1]


def _make_wav(path: Path, sr: int = 44100, dur_s: float = 4.0) -> Path:
    t = np.arange(int(sr * dur_s), dtype=np.float32) / sr
    sig = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([sig, sig], axis=1)
    pcm = np.clip(np.round(stereo * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


@pytest.mark.parametrize("task", VALID_TASKS)
def test_render_worker_script_is_valid_python_requires_cuda_and_uses_rnext_features(task: str) -> None:
    src = render_worker_script(
        task=task,
        input_filename="synthetic.wav",
        output_filename=f"synthetic_{task}.wav",
    )
    ast.parse(src)
    assert f'"task": "{task}"' in src
    assert "_require_cuda()" in src
    assert "import rnext_features as F" in src
    assert "F.decode_f32" in src
    assert 'torch.device("cpu")' not in src


def test_build_colab_job_plan_validates_inputs_and_uploads_features(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "synth.wav", dur_s=0.5)
    cfg = ColabJobConfig(
        input_path=wav,
        out_dir=tmp_path / "out",
        task="gpu_smoke",
        gpu="T4",
        auth="adc",
        session="mr-unit-test",
        dry_run=True,
    )
    plan = build_colab_job_plan(cfg)
    assert plan.gpu == "T4"
    assert plan.auth == "adc"
    stages = [s.stage for s in plan.steps]
    assert stages == [
        "create_session",
        "upload_features",
        "upload_input",
        "exec_worker",
        "download_audio",
        "download_report",
        "stop_session",
    ]
    assert plan.steps[-1].always is True
    exec_step = next(s for s in plan.steps if s.stage == "exec_worker")
    assert "-f" in exec_step.cmd
    assert plan.post_check_cmd[-3:] == ["--auth", "adc", "sessions"]

    with pytest.raises(ColabError, match="unsupported task"):
        build_colab_job_plan(ColabJobConfig(input_path=wav, out_dir=tmp_path / "out", task="bad", dry_run=True))

    with pytest.raises(ColabError, match="unsupported GPU"):
        build_colab_job_plan(ColabJobConfig(input_path=wav, out_dir=tmp_path / "out", gpu="RTX9999", dry_run=True))

    # Two concurrent plans built in the same second get distinct default session names.
    p_a = build_colab_job_plan(ColabJobConfig(input_path=wav, out_dir=tmp_path / "a", task="gpu_smoke", dry_run=True))
    p_b = build_colab_job_plan(ColabJobConfig(input_path=wav, out_dir=tmp_path / "b", task="gpu_smoke", dry_run=True))
    assert p_a.session != p_b.session


def test_run_colab_job_refuses_active_session_collision_and_tracks_endpoint_id(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "synth.wav", dur_s=4.0)

    def active_collision_runner(cmd, **kwargs):
        if "sessions" in cmd:
            return SimpleNamespace(returncode=0, stdout="[mr-busy] endpoint m-s-9999  T4", stderr="")
        raise AssertionError(f"must not create/run when session collides: {cmd}")

    with pytest.raises(ColabError, match="already active"):
        run_colab_job(
            ColabJobConfig(
                input_path=wav,
                out_dir=tmp_path / "out_busy",
                task="gpu_smoke",
                gpu="T4",
                session="mr-busy",
                dry_run=False,
                colab_bin="/bin/true",
            ),
            runner=active_collision_runner,
        )

    # If allocated endpoint ID still appears under another alias, stop is NOT confirmed.
    def endpoint_still_alive_runner(cmd, **kwargs):
        if "sessions" in cmd:
            return SimpleNamespace(returncode=0, stdout="[other-name] endpoint m-s-alive123  T4", stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    assert (
        ensure_session_stopped(
            "mr-mine",
            endpoint_id="m-s-alive123",
            colab_bin="/bin/true",
            runner=endpoint_still_alive_runner,
        )
        is False
    )


def test_run_colab_job_dry_run_never_invokes_subprocess(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "synth.wav", dur_s=0.5)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry_run=True must not invoke subprocess")

    res = run_colab_job(
        ColabJobConfig(
            input_path=wav,
            out_dir=tmp_path / "out",
            task="neural_declip",
            gpu="L4",
            dry_run=True,
        ),
        runner=fail_if_called,
    )
    assert res.dry_run is True
    assert res.executed is False
    assert res.manifest is None


def test_run_colab_job_refuses_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "synth.wav", dur_s=0.5)
    for filename in ("synth_gpu_smoke.wav", "worker_report.json", "manifest.json", "_mr_colab_worker_gpu_smoke.py"):
        out_dir = tmp_path / f"out_{filename}"
        out_dir.mkdir()
        prior = out_dir / filename
        prior.write_bytes(b"precious-prior-output")

        with pytest.raises(ColabError, match="refusing to overwrite existing output artifact"):
            run_colab_job(
                ColabJobConfig(
                    input_path=wav,
                    out_dir=out_dir,
                    task="gpu_smoke",
                    gpu="T4",
                    dry_run=False,
                    colab_bin="/bin/true",
                )
            )
        assert prior.read_bytes() == b"precious-prior-output"


def test_run_colab_job_end_to_end_mocked_uploads_only_run_artifacts(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "synth.wav", dur_s=4.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    unrelated = out_dir / "unrelated_note.txt"
    unrelated.write_text("do not upload", encoding="utf-8")

    input_sha = sha256_file(wav)
    calls: list[list[str]] = []

    def fake_runner(cmd, **kwargs):
        calls.append(list(cmd))
        if "download" in cmd and cmd[-1].endswith("synth_gpu_smoke.wav"):
            _make_wav(Path(cmd[-1]), dur_s=4.0)
            return SimpleNamespace(returncode=0, stdout="downloaded", stderr="")
        if "download" in cmd and cmd[-1].endswith("worker_report.json"):
            out_sha = sha256_file(out_dir / "synth_gpu_smoke.wav")
            Path(cmd[-1]).write_text(
                json.dumps(
                    {
                        "task": "gpu_smoke",
                        "status": "ok",
                        "gpu_name": "Tesla T4",
                        "cuda_available": True,
                        "elapsed_s": 1.23,
                        "input_sha256": input_sha,
                        "output_sha256": out_sha,
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="downloaded", stderr="")
        if "sessions" in cmd:
            return SimpleNamespace(returncode=0, stdout="[colab] No active sessions found on server.", stderr="")
        if "check" in cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="1 matching files")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    res = run_colab_job(
        ColabJobConfig(
            input_path=wav,
            out_dir=out_dir,
            task="gpu_smoke",
            gpu="T4",
            session="mr-mock-01",
            dry_run=False,
            drive_subpath="smoke/mock-01",
            colab_bin="/bin/true",
            rclone_bin="/bin/true",
        ),
        runner=fake_runner,
    )
    assert res.executed is True
    assert res.sessions_clean_after is True
    assert len(res.drive_transfers) == 3
    assert all(t.verified is True for t in res.drive_transfers)
    uploaded_names = {Path(t.local_path).name for t in res.drive_transfers}
    assert uploaded_names == {"synth_gpu_smoke.wav", "worker_report.json", "manifest.json"}
    assert "unrelated_note.txt" not in uploaded_names
    assert res.manifest is not None
    assert res.manifest["colab_metadata"]["gpu_name"] == "Tesla T4"
    assert res.manifest["drive_uri"].endswith("/smoke/mock-01")
    assert any("stop" in c and "mr-mock-01" in c for c in calls)


def test_session_check_nonzero_or_unlabelled_vm_fails_stop_confirmation(tmp_path: Path) -> None:
    def auth_err_runner(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="auth failed")

    with pytest.raises(ColabError, match="colab sessions failed"):
        check_active_sessions(colab_bin="/bin/true", runner=auth_err_runner)

    assert ensure_session_stopped("mr-s1", colab_bin="/bin/true", runner=auth_err_runner) is False

    # Unlabelled active VM shown as `[?] endpoint ...` or `?` must NOT be treated as confirmed stopped!
    def unlabelled_vm_runner(cmd, **kwargs):
        if "sessions" in cmd:
            return SimpleNamespace(returncode=0, stdout="[?] endpoint m-s-abc123  T4  RUNNING", stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    assert ensure_session_stopped("mr-s1", colab_bin="/bin/true", runner=unlabelled_vm_runner) is False

    # Non-zero exit from `stop` must STILL call `ensure_session_stopped` and fail confirmation.
    wav = _make_wav(tmp_path / "synth.wav", dur_s=4.0)
    out_dir = tmp_path / "out"
    stop_fail_calls: list[list[str]] = []

    def stop_fail_runner(cmd, **kwargs):
        stop_fail_calls.append(list(cmd))
        if "download" in cmd and cmd[-1].endswith("synth_gpu_smoke.wav"):
            _make_wav(Path(cmd[-1]), dur_s=4.0)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "download" in cmd and cmd[-1].endswith("worker_report.json"):
            Path(cmd[-1]).write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "stop" in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="stop failed")
        if "sessions" in cmd:
            return SimpleNamespace(returncode=0, stdout="[colab] No active sessions found on server.", stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with pytest.raises(ColabError, match="could not confirm Colab session"):
        run_colab_job(
            ColabJobConfig(
                input_path=wav,
                out_dir=out_dir,
                task="gpu_smoke",
                gpu="T4",
                session="mr-stop-fail",
                dry_run=False,
                colab_bin="/bin/true",
            ),
            runner=stop_fail_runner,
        )
    assert any("sessions" in c for c in stop_fail_calls)


def test_run_colab_job_rejects_mismatched_worker_report_hash(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "synth.wav", dur_s=4.0)
    out_dir = tmp_path / "out"

    def bad_hash_runner(cmd, **kwargs):
        if "download" in cmd and cmd[-1].endswith("synth_gpu_smoke.wav"):
            _make_wav(Path(cmd[-1]), dur_s=4.0)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "download" in cmd and cmd[-1].endswith("worker_report.json"):
            Path(cmd[-1]).write_text(
                json.dumps(
                    {
                        "task": "gpu_smoke",
                        "status": "ok",
                        "cuda_available": True,
                        "input_sha256": "deadbeef",
                        "output_sha256": "deadbeef",
                    }
                ),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        if "sessions" in cmd:
            return SimpleNamespace(returncode=0, stdout="[colab] No active sessions found on server.", stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with pytest.raises(ColabError, match="input_sha256 mismatch"):
        run_colab_job(
            ColabJobConfig(
                input_path=wav,
                out_dir=out_dir,
                task="gpu_smoke",
                gpu="T4",
                session="mr-bad-hash",
                dry_run=False,
                colab_bin="/bin/true",
            ),
            runner=bad_hash_runner,
        )


def test_auto_stop_fires_even_when_worker_exec_fails(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "synth.wav", dur_s=4.0)
    calls: list[list[str]] = []

    def fake_runner(cmd, **kwargs):
        calls.append(list(cmd))
        if "exec" in cmd:
            return SimpleNamespace(returncode=1, stdout="", stderr="CUDA OOM")
        if "sessions" in cmd:
            return SimpleNamespace(returncode=0, stdout="[colab] No active sessions found on server.", stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with pytest.raises(ColabError, match="Colab stage 'exec_worker' failed"):
        run_colab_job(
            ColabJobConfig(
                input_path=wav,
                out_dir=tmp_path / "out",
                task="gpu_smoke",
                gpu="T4",
                session="mr-fail-01",
                dry_run=False,
                colab_bin="/bin/true",
            ),
            runner=fake_runner,
        )

    assert any("stop" in c and "mr-fail-01" in c for c in calls)


def test_ensure_session_stopped_stops_dangling_session() -> None:
    calls: list[list[str]] = []

    def fake_runner(cmd, **kwargs):
        calls.append(list(cmd))
        if "sessions" in cmd and len(calls) == 1:
            return SimpleNamespace(returncode=0, stdout="mr-dangling  T4", stderr="")
        if "stop" in cmd:
            return SimpleNamespace(returncode=0, stdout="stopped", stderr="")
        return SimpleNamespace(returncode=0, stdout="[colab] No active sessions found on server.", stderr="")

    assert ensure_session_stopped("mr-dangling", colab_bin="/bin/true", runner=fake_runner) is True
    assert any("stop" in c and "mr-dangling" in c for c in calls)


def test_thin_cli_dry_run_scripts(tmp_path: Path) -> None:
    wav = _make_wav(tmp_path / "cli_test.wav", dur_s=0.5)
    p1 = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "colab_restore.py"),
            "--input",
            str(wav),
            "--out-dir",
            str(tmp_path / "out"),
            "--task",
            "gpu_smoke",
            "--gpu",
            "T4",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data1 = json.loads(p1.stdout)
    assert data1["dry_run"] is True
    assert data1["plan"]["gpu"] == "T4"
    assert any(s["stage"] == "stop_session" for s in data1["plan"]["steps"])

    p2 = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "drive_sync.py"),
            "upload",
            str(wav),
            "smoke/cli-test",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data2 = json.loads(p2.stdout)
    assert data2["dry_run"] is True
    assert data2["executed"] is False


def test_cloud_modules_have_no_hardcoded_machine_paths() -> None:
    files = [
        REPO / "rnext" / "cloud" / "__init__.py",
        REPO / "rnext" / "cloud" / "metrics.py",
        REPO / "rnext" / "cloud" / "drive.py",
        REPO / "rnext" / "cloud" / "colab_worker.py",
        REPO / "rnext" / "cloud" / "colab_runner.py",
        REPO / "scripts" / "colab_restore.py",
        REPO / "scripts" / "drive_sync.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text, f"hardcoded /home/ path in {path}"
        assert "/media/" not in text, f"hardcoded /media/ path in {path}"
