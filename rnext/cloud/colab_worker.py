"""Self-contained Colab GPU worker script generator.

Produces a standalone Python 3 script executed inside the Colab VM
(``/content/``) with zero dependencies on local workspace paths. Supports:
  - ``gpu_smoke``: fast CUDA GPU verification + PyTorch spectral soft-declip pass
  - ``neural_declip``: PyTorch CUDA STFT-domain iterative declipping & peak recovery
  - ``demucs``: Demucs v4 (htdemucs) GPU stem separation
  - ``audiosr``: AudioSR GPU bandwidth extension
"""
from __future__ import annotations

import json
from typing import Any

VALID_TASKS = ("gpu_smoke", "neural_declip", "demucs", "audiosr")


def render_worker_script(
    *,
    task: str,
    input_filename: str,
    output_filename: str,
    extra_options: dict[str, Any] | None = None,
) -> str:
    """Render a self-contained Colab GPU worker script for ``task``."""
    if task not in VALID_TASKS:
        raise ValueError(f"unsupported Colab task {task!r}; expected one of {VALID_TASKS}")
    if not input_filename or "/" in input_filename or "\\" in input_filename:
        raise ValueError(f"invalid input_filename: {input_filename!r}")
    if not output_filename or "/" in output_filename or "\\" in output_filename:
        raise ValueError(f"invalid output_filename: {output_filename!r}")

    cfg = {
        "task": task,
        "input_path": f"/content/{input_filename}",
        "output_path": f"/content/mr_out/{output_filename}",
        "report_path": "/content/mr_out/worker_report.json",
        "options": extra_options or {},
    }
    cfg_json = json.dumps(cfg, sort_keys=True)

    return f'''#!/usr/bin/env python3
"""Auto-generated Colab GPU audio restoration worker (task={task})."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

CONFIG = json.loads({cfg_json!r})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_audio_f32(path: Path) -> tuple[np.ndarray, int]:
    """Decode WAV/FLAC to float32 [frames, channels] using scipy/wave/ffmpeg."""
    if path.suffix.lower() == ".wav":
        try:
            from scipy.io import wavfile

            sr, data = wavfile.read(str(path))
            if data.ndim == 1:
                data = data[:, np.newaxis]
            if np.issubdtype(data.dtype, np.integer):
                info = np.iinfo(data.dtype)
                scale = float(max(abs(info.min), info.max))
                arr = (data.astype(np.float32) / scale)
            else:
                arr = data.astype(np.float32)
            return arr, int(sr)
        except Exception:
            pass
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    sr = int(stream["sample_rate"])
    ch = int(stream["channels"])
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-map", "0:a:0", "-ac", str(ch), "-c:a", "pcm_f32le", "-f", "f32le", "-",
        ],
        capture_output=True, check=True,
    ).stdout
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).copy()
    return arr, sr


def _write_wav_pcm16(path: Path, x: np.ndarray, sr: int) -> None:
    """Write float32 [-1, 1] array to 16-bit PCM WAV with true-peak headroom."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0.94:
        arr = arr * (0.94 / peak)
    pcm = np.clip(np.round(arr * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def _gpu_spectral_declip(x: np.ndarray, sr: int, iterations: int = 8) -> tuple[np.ndarray, dict]:
    """Run PyTorch CUDA STFT-domain iterative soft-declipping & peak recovery."""
    import torch

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    meta = {{
        "torch_version": torch.__version__,
        "cuda_available": bool(use_cuda),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "cuda_version": torch.version.cuda if use_cuda else None,
        "vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            if use_cuda
            else 0.0
        ),
    }}

    t = torch.from_numpy(x.T.copy()).to(device=device, dtype=torch.float32)  # [C, N]
    clip_thresh = 0.985
    clipped_mask = t.abs() >= clip_thresh
    n_fft = min(2048, max(256, 1 << (t.shape[-1] - 1).bit_length()))
    if n_fft > t.shape[-1]:
        n_fft = max(64, 1 << (t.shape[-1].bit_length() - 1))
    hop = max(16, n_fft // 4)
    window = torch.hann_window(n_fft, device=device, dtype=torch.float32)

    restored = t.clone()
    if bool(clipped_mask.any()) and t.shape[-1] >= n_fft:
        for _ in range(max(1, iterations)):
            spec = torch.stft(restored, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
            mag = spec.abs()
            # Soft-threshold sparse spectral shrinkage to suppress clipping harmonics.
            tau = torch.quantile(mag.reshape(mag.shape[0], -1), 0.70, dim=-1, keepdim=True).unsqueeze(-1) * 0.15
            mag_shrunk = torch.clamp(mag - tau, min=0.0)
            phase = spec / (mag + 1e-8)
            recon = torch.istft(
                mag_shrunk * phase,
                n_fft=n_fft,
                hop_length=hop,
                window=window,
                length=t.shape[-1],
            )
            # Project unclipped samples back to exact original values; keep reconstructed peaks.
            restored = torch.where(clipped_mask, recon, t)
    else:
        # Still exercise real CUDA FFT round-trip + gentle peak safety for smoke verification.
        if t.shape[-1] >= n_fft:
            spec = torch.stft(restored, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
            restored = torch.istft(spec, n_fft=n_fft, hop_length=hop, window=window, length=t.shape[-1])

    # Apply conservative -0.5 dBFS ceiling so restored peaks never re-clip on PCM16 export.
    peak = float(restored.abs().max().item()) if restored.numel() else 0.0
    ceiling = 10.0 ** (-0.5 / 20.0)
    if peak > ceiling:
        restored = restored * (ceiling / peak)

    if use_cuda:
        torch.cuda.synchronize()
    out_np = restored.detach().cpu().numpy().T
    meta["clipped_samples_detected"] = int(clipped_mask.sum().item())
    meta["iterations"] = int(iterations)
    return out_np, meta


def _run_demucs(input_path: Path, output_path: Path, options: dict) -> dict:
    import torch

    if shutil.which("demucs") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "demucs"],
            check=True,
        )
    model = str(options.get("model", "htdemucs"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stems_root = output_path.parent / "stems"
    stems_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "-n", model,
            "--device", device,
            "-o", str(stems_root),
            str(input_path),
        ],
        check=True,
    )
    stem_dir = stems_root / model / input_path.stem
    stem_files = sorted(stem_dir.glob("*.wav")) if stem_dir.is_dir() else []
    if not stem_files:
        raise RuntimeError(f"demucs produced no stem wavs in {{stem_dir}}")
    # Recombine stems cleanly into output_path so before/after metrics run seamlessly.
    acc = None
    sr_out = 44100
    for sf in stem_files:
        arr, sr_out = _read_audio_f32(sf)
        acc = arr.astype(np.float64) if acc is None else (acc + arr.astype(np.float64))
    _write_wav_pcm16(output_path, acc.astype(np.float32), sr_out)
    return {{
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "demucs_model": model,
        "stems": [sf.name for sf in stem_files],
    }}


def _run_audiosr(input_path: Path, output_path: Path, options: dict) -> dict:
    import torch

    if shutil.which("audiosr") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "audiosr"],
            check=True,
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp_dir = output_path.parent / "audiosr_out"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "audiosr",
            "-i", str(input_path),
            "-s", str(tmp_dir),
            "--device", device,
        ],
        check=True,
    )
    produced = sorted(tmp_dir.rglob("*.wav"))
    if not produced:
        raise RuntimeError(f"audiosr produced no wav in {{tmp_dir}}")
    shutil.copy2(produced[0], output_path)
    return {{
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "audiosr_output": produced[0].name,
    }}


def main() -> int:
    t0 = time.monotonic()
    task = CONFIG["task"]
    input_path = Path(CONFIG["input_path"])
    output_path = Path(CONFIG["output_path"])
    report_path = Path(CONFIG["report_path"])
    options = CONFIG.get("options") or {{}}

    if not input_path.is_file():
        raise FileNotFoundError(f"missing uploaded input audio: {{input_path}}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if task in ("gpu_smoke", "neural_declip"):
        x, sr = _read_audio_f32(input_path)
        iters = int(options.get("iterations", 4 if task == "gpu_smoke" else 12))
        y, meta = _gpu_spectral_declip(x, sr, iterations=iters)
        _write_wav_pcm16(output_path, y, sr)
        sample_rate, channels, num_frames = sr, int(y.shape[1]), int(y.shape[0])
    elif task == "demucs":
        meta = _run_demucs(input_path, output_path, options)
        y, sr = _read_audio_f32(output_path)
        sample_rate, channels, num_frames = sr, int(y.shape[1]), int(y.shape[0])
    elif task == "audiosr":
        meta = _run_audiosr(input_path, output_path, options)
        y, sr = _read_audio_f32(output_path)
        sample_rate, channels, num_frames = sr, int(y.shape[1]), int(y.shape[0])
    else:
        raise ValueError(f"unsupported task: {{task}}")

    elapsed_s = round(time.monotonic() - t0, 3)
    report = {{
        "task": task,
        "status": "ok",
        "elapsed_s": elapsed_s,
        "input_name": input_path.name,
        "input_sha256": _sha256(input_path),
        "output_name": output_path.name,
        "output_sha256": _sha256(output_path),
        "sample_rate": sample_rate,
        "channels": channels,
        "num_frames": num_frames,
        **meta,
    }}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(f"[mr-colab-worker] task={{task}} gpu={{report.get('gpu_name')}} elapsed={{elapsed_s}}s out={{output_path.name}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
