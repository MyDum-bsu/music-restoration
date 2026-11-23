# E7 · BWE Preview with Marked Synthesized Band — Research Report

**Card:** Board #1133 · **Spec:** E0 charter + spec v1 (Section E7) · **Dataset:** `dev-e7-bwe-2026-11-23` (`17` stereo 48 kHz cases) · **Neural BWE Arm:** `blocked_external_weights_or_colab`

## 1. Scope, Safety Boundaries & Provenance

- **Sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`):** `sealed_corpora_opened = False` (strictly untouched).
- **Frozen E4 Preserve hard-declip pipeline:** `18/18` SHA-256 pins verified untouched (`all_untouched = True`).
- **Preserve promotion / neural production dependency / Colab automation:** `False` (isolated opt-in Enhance preview study only).
- **Manifest SHA-256 (`corpus/dev-e7-bwe-v1.json`):** `0cef1dad5b41758ec6412f998d523a89c4a14dc3a65663231c05c4fdd5912334`
- **Corpus Builder SHA-256 (`research/build_e7_bwe_corpus.py`):** `2b80e00479bf0245ebc6a8a903d95853ea37730d22228979ef2152d814a8157d`
- **E7 Harness SHA-256 (`research/e7_bwe_preview.py`):** `e0b84e6c7d3db6dbe97c8964f116037cb2d5f23c44b8d84b8296e85b4e03614a`
- **Additive & Immutability Contract:** `34/34` runs passed `preview == input + synthesized_band` (`max_err = 0.00e+00`) with `0` input mutations.

> **Honest Semantics Notice:** High-frequency energy above `observed_cutoff_hz` produced by `dsp_harmonic_bwe_preview` is a **synthesized preview** (`synthesized_hf_band = true`, `is_generative_preview = true`, `not_original_restoration = true`), **never** a recovered original waveform. Missing-band LSD and spectral tilt evaluate envelope plausibility only.

---

## 2. Band-Separated Comparison by Stratum

### 2.1 Observed Band Preservation (`[0, observed_cutoff_hz]`) vs Missing Band Envelope (`(observed_cutoff_hz, 22 kHz]`)

| Stratum (n) | Arm | Obs ΔSNR (dB) | Obs ΔLSD (dB) | Obs Max Err vs In | Miss LSD In (dB) | Miss LSD Out (dB) | Miss ΔLSD (dB) | Synth Energy (dBFS) | Wall (s/min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bandlimit_8k` (3) | `safe_resample_only` | +0.000 | +0.0000 | 0.00e+00 | 36.51 | 36.51 | +0.00 | 0 (none) | 0.03 |
| `bandlimit_8k` (3) | `dsp_harmonic_bwe_preview` | +0.000 | -0.0000 | 3.43e-05 | 36.51 | 17.63 | -18.88 | -43.8 | 4.21 |
| `bandlimit_11k` (3) | `safe_resample_only` | +0.000 | +0.0000 | 0.00e+00 | 33.74 | 33.74 | +0.00 | 0 (none) | 0.04 |
| `bandlimit_11k` (3) | `dsp_harmonic_bwe_preview` | +0.000 | -0.0000 | 1.12e-05 | 33.74 | 17.90 | -15.84 | -42.4 | 5.13 |
| `bandlimit_16k` (2) | `safe_resample_only` | +0.000 | +0.0000 | 0.00e+00 | 20.62 | 20.62 | +0.00 | 0 (none) | 0.05 |
| `bandlimit_16k` (2) | `dsp_harmonic_bwe_preview` | +0.000 | -0.0000 | 1.07e-05 | 20.62 | 17.92 | -2.71 | -42.7 | 6.23 |
| `codec_roundtrip` (3) | `safe_resample_only` | +0.000 | +0.0000 | 0.00e+00 | 33.94 | 33.94 | +0.00 | 0 (none) | 0.03 |
| `codec_roundtrip` (3) | `dsp_harmonic_bwe_preview` | -0.000 | +0.0000 | 1.60e-05 | 33.94 | 18.69 | -15.25 | -45.3 | 3.20 |
| `fullband_clean_control` (3) | `safe_resample_only` | +0.000 | +0.0000 | 0.00e+00 | 0.00 | 0.00 | +0.00 | 0 (none) | 0.02 |
| `fullband_clean_control` (3) | `dsp_harmonic_bwe_preview` | +0.000 | +0.0000 | 0.00e+00 | 0.00 | 0.00 | +0.00 | 0 (none) | 0.38 |
| `ood` (3) | `safe_resample_only` | +0.000 | +0.0000 | 0.00e+00 | 39.59 | 39.59 | +0.00 | 0 (none) | 0.03 |
| `ood` (3) | `dsp_harmonic_bwe_preview` | +0.000 | +0.0000 | 2.81e-04 | 39.59 | 19.60 | -19.99 | -45.5 | 2.93 |

### 2.2 Stereo Coherence (L/R Correlation, ILD, IPD) & Artifact Checks

| Stratum | Arm | Obs L/R Drift | Obs ILD Drift (dB) | Obs IPD Err (deg) | Miss L/R Drift | Miss ILD Drift (dB) | Miss IPD Err (deg) | Sub-Cutoff Leak (dB) | Pre-Ring Rise (dB) | False HF Cases |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bandlimit_8k` | `safe_resample_only` | 0.0022 | 0.009 | 0.05 | 0.021 | 0.29 | 40.9 | 0 (none) | 0.000 | 0/3 |
| `bandlimit_8k` | `dsp_harmonic_bwe_preview` | 0.0022 | 0.009 | 0.05 | 0.049 | 0.70 | 50.6 | -89.1 | 0.185 | 0/3 |
| `bandlimit_11k` | `safe_resample_only` | 0.0037 | 0.023 | 0.05 | 0.043 | 0.32 | 47.1 | 0 (none) | 0.000 | 0/3 |
| `bandlimit_11k` | `dsp_harmonic_bwe_preview` | 0.0037 | 0.023 | 0.05 | 0.035 | 0.40 | 54.2 | -100.5 | 0.147 | 0/3 |
| `bandlimit_16k` | `safe_resample_only` | 0.0054 | 0.003 | 0.09 | 0.014 | 0.28 | 48.7 | 0 (none) | 0.000 | 0/2 |
| `bandlimit_16k` | `dsp_harmonic_bwe_preview` | 0.0054 | 0.003 | 0.09 | 0.084 | 0.31 | 55.1 | -105.1 | 0.024 | 0/2 |
| `codec_roundtrip` | `safe_resample_only` | 0.0441 | 0.023 | 3.06 | 0.143 | 0.53 | 41.2 | 0 (none) | 0.000 | 0/3 |
| `codec_roundtrip` | `dsp_harmonic_bwe_preview` | 0.0441 | 0.023 | 3.06 | 0.102 | 0.24 | 44.8 | -100.1 | 0.135 | 0/3 |
| `fullband_clean_control` | `safe_resample_only` | 0.0000 | 0.000 | 0.00 | 0.000 | 0.00 | 0.0 | 0 (none) | 0.000 | 0/3 |
| `fullband_clean_control` | `dsp_harmonic_bwe_preview` | 0.0000 | 0.000 | 0.00 | 0.000 | 0.00 | 0.0 | 0 (none) | 0.000 | 0/3 |
| `ood` | `safe_resample_only` | 0.0315 | 0.249 | 1.55 | 0.163 | 0.63 | 51.8 | 0 (none) | 0.000 | 0/3 |
| `ood` | `dsp_harmonic_bwe_preview` | 0.0315 | 0.249 | 1.55 | 0.168 | 0.64 | 54.0 | -70.0 | 0.271 | 0/3 |

---

## 3. Codec Roundtrip & Full-Band Control Diagnostics

### 3.1 Per-Case Codec Roundtrip & OOD Behavior (`dsp_harmonic_bwe_preview`)

| Case ID | Codec / Backend | Cutoff (Hz) | Obs ΔSNR (dB) | Miss LSD In -> Out (dB) | Tilt Err (dB/oct) | Miss ILD Drift (dB) | Pre-Ring (dB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `e7-codec-mp3-64k-11k-48k` | `ffmpeg_libmp3lame_64k` | 11025 | +0.000 | 39.85 -> 19.90 | 14.41 | 0.34 | 0.058 |
| `e7-codec-mp3-96k-14k-48k` | `ffmpeg_libmp3lame_96k` | 14000 | -0.000 | 27.21 -> 18.30 | 38.40 | 0.26 | 0.022 |
| `e7-codec-opus-32k-12k-48k` | `ffmpeg_libopus_32k` | 12000 | -0.000 | 34.76 -> 17.88 | 19.45 | 0.10 | 0.135 |
| `e7-ood-narrowband-4k-48k` | `polyphase_kaiser14` | 4000 | +0.000 | 42.85 -> 20.22 | 5.60 | 1.22 | 0.271 |
| `e7-ood-wide-stereo-8k-48k` | `polyphase_kaiser14` | 8000 | -0.000 | 35.09 -> 19.10 | 2.91 | 0.67 | 0.110 |
| `e7-ood-double-codec-8k-48k` | `ffmpeg_libmp3lame_64k->ffmpeg_libopus_32k` | 8000 | -0.000 | 40.85 -> 19.49 | 3.17 | 0.03 | 0.035 |

### 3.2 Full-Band Negative Control: Default Guard vs Forced BWE (`bypass_fullband=False`)

| Case ID | Mode | Bypassed | False HF Injected | Synth Energy (dBFS) | Obs Max Err vs In | Miss ΔLSD (dB) |
|---|---|---|---:|---:|---:|---:|
| `e7-ctl-fullband-percussion-48k` | default (`bypass_fullband=True`) | `True` | `False` | 0 (none) | 0.00e+00 | +0.00 |
| `e7-ctl-fullband-percussion-48k` | forced (`bypass_fullband=False`) | `False` | `True` | -41.8 | 1.11e-05 | +13.47 |
| `e7-ctl-fullband-plucked-48k` | default (`bypass_fullband=True`) | `True` | `False` | 0 (none) | 0.00e+00 | +0.00 |
| `e7-ctl-fullband-plucked-48k` | forced (`bypass_fullband=False`) | `False` | `True` | -45.2 | 4.67e-06 | +14.89 |
| `e7-ctl-fullband-harmonic-48k` | default (`bypass_fullband=True`) | `True` | `False` | 0 (none) | 0.00e+00 | +0.00 |
| `e7-ctl-fullband-harmonic-48k` | forced (`bypass_fullband=False`) | `False` | `True` | -42.4 | 3.47e-06 | +11.69 |

---

## 4. Opt-In Derivative Export & Mandatory Tagging Contract

- `export_bwe_preview()` writes **separate** opt-in files (`*.bwe_preview.flac`, `*.synthesized_band_only.flac`, and `*.bwe_preview.sidecar.json`), rejects any symlink or source-aliasing output path via `lstat`/`realpath` before writing, and verifies `source_sha256` before and after writing all three files (zero source mutation or library replacement).
- Both the FLAC Vorbis comments and the JSON sidecar embed `synthesized_hf_band`, `observed_cutoff_hz`, `is_generative_preview`, `not_original_restoration`, `enhance_opt_in_derivative`, and `source_sha256`; the post-tag hashes `derivative_sha256` and `synthesized_band_sha256` are recorded in the JSON sidecar.

## 5. Neural BWE Arm Audit (AudioSR / UniverSR / BABE-2 / SAMECFM) & Colab Contract

- **Status:** `blocked_external_weights_or_colab` (`executed = False`, `production_dependency_added = False`).
- **Blocked Reasons:** `no_local_licensed_bwe_weights_audiosr_universr_babe2_samecfm, no_local_cuda_gpu, colab_connection_not_authorized_or_configured`.
- **Environment Audit:** `torch = 2.14.1+cpu`, `cuda_available = False`, `local_bwe_weights_found = 0`, `colab_authorized = False`.
- **License & Colab Gate:** Target models (`AudioSR`, `UniverSR`, `BABE-2`, `SAMECFM`) require explicit license verification and checkpoint SHA-256 pinning prior to download or execution. Authorized Colab runs may use `corpus/dev-e7-bwe-v1.json` (`rights.colab == True`) ONLY — never `corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`, or private masters — and must pass local `evaluate_e7_case` verification plus mandatory `synthesized_hf_band` tagging.

## 6. Limitations & Honest Method Boundaries

1. **Synthesized HF Is Preview, Not Original Restoration:** `dsp_harmonic_bwe_preview` improves missing-band spectral envelope distance on average (negative `Miss ΔLSD`) by transposing and harmonically exciting the upper observed octave, but it **cannot** reconstruct true lost micro-transients, unique cymbal overtones, or unpitched HF phase. Missing-band IPD error remains high (~45-55 deg across strata) because phase in the missing band is unobservable from band-limited input. On `e7-bl16k-bright-mix-48k`, where the clean reference rolls off steeply above 16 kHz (`-87.5 dB/oct`), SBR extrapolation (`-10.4 dB/oct`) slightly worsens missing-band LSD by `+0.22 dB` (`19.36 -> 19.58 dB`).
2. **Cutoff Transition Guard Band (`-2% / +4%`):** Band-separated metrics evaluate `observed_band` over `[20 Hz, 0.98 * observed_cutoff_hz]` and `missing_band` over `[1.04 * observed_cutoff_hz, 0.92 * nyquist]` to exclude the linear-phase crossover transition skirt around `observed_cutoff_hz`; minor crossover-region differences inside `(0.98 * cutoff, 1.04 * cutoff)` are outside these two summary bands.
3. **Role of the Full-Band Bypass Guard:** Zero false HF on `fullband_clean_control` relies on `_detect_fullband_presence` (Hann-windowed Welch PSD `hf_to_mid_db > -28 dB`). When forced with `bypass_fullband=False` (Section 3.2), DSP synthesis injects extraneous HF above 20 kHz, confirming why explicit full-band detection and opt-in user control are mandatory.
4. **Extreme Narrowband OOD (`e7-ood-narrowband-4k-48k`):** Extending from a 4 kHz cutoff across >2.5 missing octaves via periodic SBR + harmonic excitation yields a coarse spectral approximation only and increases timbre mismatch relative to 11-16 kHz cutoffs.
5. **Codec Roundtrip Interaction (`mp3_64k`, `opus_32k`, `ood_double_codec_8k`):** Lossy codec quantization and pre-echo present in the upper observed band (`[0.5*cutoff, cutoff]`) are transposed upward by SBR; BWE does **not** remove codec pre-echo inside the observed band.
6. **Neural BWE Comparison Pending Weights/Colab:** No AudioSR, UniverSR, BABE-2, or SAMECFM comparison numbers are reported because licensed weights and authorized Colab GPU execution are not provisioned in this offline CPU environment.
