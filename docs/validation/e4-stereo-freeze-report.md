# E4 · Stereo/Album Safety Verification & CPU Candidate Freeze

**Card:** Board #1127 · **Spec:** E0 charter + spec v1 (sections 3, 4, E4) · **Freeze status:** `frozen_for_e5` · **Sealed audio opened:** `false` (custodian-held; `corpus/sealed-v1.json` metadata inspected post-pilot for group-leakage audit only, audio never accessed or executed).

- **Frozen Candidate:** `aspade_mit__v2_windowed` (`detect_hard_clip_v2` + `declip_aspade` + unified `-1.0 dBTP` headroom)
- **Freeze SHA-256:** `ee4cdc8cec0a4f6a365b4784e6b694b64febac202f1ce31528f66a38a416f09c`
- **Reference CPU:** `x86_64` (8 logical cores, single-threaded BLAS/OpenMP `= 1`)
- **Toolchain Pins:** Python `3.12.3`, NumPy `2.5.2`, SciPy `1.17.1`, FFmpeg `125907-ga7e72069f1-20260802`, libflac `1.4.3`
- **Cross-manifest group leakage (`smoke/dev/dev-e3/sealed`):** `0` errors
- **Neural / Google Colab arm:** `not_run` (`colab_connection_not_authorized_or_configured`; excluded from CPU release claim)
- **Machine-verifiable freeze manifest:** [`docs/validation/e4-freeze-manifest.json`](e4-freeze-manifest.json)

---

## 1. Suite A — Stereo Safety Audit on Non-Sealed Corpora (`smoke-v1`, `dev-v1`, `dev-e3-v1`)

| Corpus | Defects / Ctrls | Coverage | Ctrl FP | Pol Viol | Max Rel Diff | Worst L/R Corr Drift | Worst ILD Drift (dB) | Worst Local ILD p95 (dB) | Worst IPD Mean (°) | Worst ITD Drift (samp) | Worst M/S Drift (dB) | Worst Pre-Echo (dB) | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smoke-v1` | 5 / 3 | 100.0% | 0 | 0 | 0.0e+00 | 0.0000 | 0.00 | 0.01 | 0.77 | - | 0.00 | 0.00 | `PASS` |
| `dev-v1` | 5 / 3 | 100.0% | 0 | 0 | 0.0e+00 | 0.0000 | 0.00 | 0.00 | 0.31 | - | 0.00 | 0.00 | `PASS` |
| `dev-e3-v1` | 12 / 6 | 100.0% | 0 | 0 | 0.0e+00 | 0.0004 | 0.02 | 0.08 | 0.28 | 0.002 | 0.05 | 0.00 | `PASS` |

*(Note: localized gain-jump `late_clip` cases with `defect_gain > 1` are evaluated for pre-gain invariants and zero polarity violations; as documented in E3 Section 4, their whole-track clean reference contains an unrecorded local gain step.)*

---

## 2. Suite B — Dedicated Stereo Stress Suite & Real 24-Bit FLAC Albums

### 2.1 Stereo Image, Phase/IPD, Single-Channel, Pre-Echo & Quiet-Tail Stress Cases (Real FLAC Input)

| Case | Rate / Bits | Status | ΔMasked SDR (dB) | L/R Corr Drift | ILD Drift (dB) | Local ILD p95 (dB) | IPD Mean (°) | ITD Drift (samp) | M/S Drift (dB) | Mono ΔSDR (dB) | Pre-Echo (dB) | Untouched Ch Diff | Gates |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `stereo-coherent-corr-44k` | 44.1k / 24b | `processed_candidate` | +7.11 | 0.0001 | 0.00 | 0.01 | 0.03 | 0.001 | 0.00 | +7.44 | 0.00 | - | `PASS` |
| `stereo-ild-itd-panned-48k` | 48k / 24b | `processed_candidate` | +8.60 | 0.0001 | 0.00 | 0.02 | 0.01 | 0.006 | 0.01 | +8.62 | 0.00 | - | `PASS` |
| `stereo-single-ch-left-44k` | 44.1k / 24b | `processed_candidate` | +8.61 | 0.0001 | 0.01 | 0.02 | 0.05 | 0.000 | 0.03 | +8.61 | 0.00 | 0.0e+00 | `PASS` |
| `stereo-single-ch-right-asym-16b` | 48k / 16b | `processed_candidate` | +3.12 | 0.0022 | 0.12 | 0.32 | 1.09 | 0.004 | 0.09 | +3.05 | 0.00 | 0.0e+00 | `PASS` |
| `stereo-transient-pre-echo-48k` | 48k / 24b | `processed_candidate` | +11.47 | 0.0006 | 0.04 | 0.05 | 0.46 | 0.006 | 0.06 | +13.31 | 0.00 | - | `PASS` |
| `stereo-quiet-tail-96k` | 96k / 24b | `processed_candidate` | +4.80 | 0.0006 | 0.00 | 0.05 | 0.12 | 0.001 | 0.04 | +4.69 | 0.00 | - | `PASS` |
| `ctrl-clean-stereo-44k` | 44.1k / 24b | `kept_original` | - | 0.0000 | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | +0.00 | 0.00 | - | `PASS` |
| `ctrl-clean-percussive-48k` | 48k / 24b | `kept_original` | - | 0.0000 | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | +0.00 | 0.00 | - | `PASS` |
| `ctrl-intentional-tanh-44k` | 44.1k / 16b | `kept_original` | - | 0.0003 | 0.18 | 0.80 | 3.09 | 0.002 | 0.12 | +0.00 | 0.00 | - | `PASS` |
| `ctrl-soft-limiter-48k` | 48k / 24b | `kept_original` | - | 0.0000 | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 | +0.00 | 0.00 | - | `PASS` |

### 2.2 Real Multi-Track Stereo FLAC Albums (`restore_safe.py --album-headroom` + `evaluate_album.py`)

1. **4-Track Gapless All-Clipped Album (`album_all_clipped`)** — `PASS`:
   - Policy: `unified_album_gain`, unified album gain `+0.00 dB`, `max_relative_level_shift_db = 0.00 dB`, `auto_album_delivery = True`, `needs_review = False`.
   - Unified `-1.0 dBTP` headroom: all delivered tracks `<= -1.0 + 0.05 dBTP` (`True`), `0` clipped samples (`True`).
   - Gapless Track 01 → Track 02 seam vs clean ground truth: level jump `0.0000 dB`, seam peak growth `-0.000 dB`, seam step `0.00598` vs ref `0.00598` (`passed = True`).
   - `scripts/evaluate_album.py` verdict: `comparable` (`{'ours_better': 0, 'orig_better': 0, 'comparable': 4, 'inconclusive': 0}`).
2. **3-Track Mixed Clipped + Immutable Kept Album (`album_mixed_kept`)** — `PASS`:
   - Policy: `mixed_kept_and_attenuated_incompatible`, `compatible = False`, `needs_review = True`, `auto_album_delivery = False`, `max_relative_level_shift_db = 2.57 dB`.
   - Track statuses: `{'01 - Clipped Track.flac': 'processed_candidate', '02 - Clean Continuation.flac': 'kept_original', '03 - Intentional Saturation.flac': 'kept_original'}`; kept tracks byte-for-byte SHA-256 identical to source (`True`).
   - Confirms Spec Section 3 contract: never silently attenuates/re-encodes immutable `kept_original` tracks and never silently delivers a level-shifted album.

---

## 3. Suite C — Local CPU Profiling (Cold/Warm, 5-Minute Stereo FLAC Tracks & Album)

All timings and peak RSS values were measured in **isolated fresh subprocesses** (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`) on real stereo FLAC files via `/proc/self/status` `VmHWM` (true post-`execve` resident high-water mark, zero parent/synthesis contamination; chunked 4x polyphase `true_peak_dbfs`).

> **Provenance note:** The 10 single-track 5-minute rows below (`tracks` and 5-min aggregate) were reused from freeze run `6ebc11f22a6b8ff71d62c6abe02f205ee393fdbb2395f5fee93586bd99dbc7b5` after verifying byte-identical SHA-256 hashes of all 4 core DSP modules (`rnext/dsp/clip_detector.py`, `rnext/dsp/declip_methods.py`, `rnext/dsp/contour.py`, `rnext/features.py`); Suites A, B, and the 10-minute `album_benchmark` were re-measured in full on the current revision.

| Track | Rate / Bits | Dur (s) | Masked (%) | Run | Decode (s) | Detect (s) | Repair (s) | Headroom (s) | Encode+QC (s) | Total (s) | RTF | s / audio min | Peak RSS (MiB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `track-5m-44k-16b-mod` | 44.1k / 16b | 300 | 0.004% | cold | 0.23 | 2.26 | 14.11 | 1.57 | 3.55 | 21.77 | 0.073x | 4.35 | 1217.5 |
| `track-5m-44k-16b-mod` | 44.1k / 16b | 300 | 0.004% | warm | 0.19 | 2.20 | 15.79 | 1.59 | 2.76 | 22.55 | 0.075x | 4.51 | 1217.8 |
| `track-5m-44k-24b-heavy` | 44.1k / 24b | 300 | 0.017% | cold | 0.24 | 2.54 | 36.29 | 1.64 | 3.01 | 43.73 | 0.146x | 8.75 | 1218.4 |
| `track-5m-44k-24b-heavy` | 44.1k / 24b | 300 | 0.017% | warm | 0.21 | 2.48 | 33.99 | 1.66 | 2.50 | 40.87 | 0.136x | 8.17 | 1218.4 |
| `track-5m-48k-24b-mod` | 48k / 24b | 300 | 0.005% | cold | 0.25 | 2.45 | 16.44 | 1.75 | 3.15 | 24.08 | 0.080x | 4.82 | 1278.7 |
| `track-5m-48k-24b-mod` | 48k / 24b | 300 | 0.005% | warm | 0.20 | 2.48 | 16.11 | 1.73 | 2.75 | 23.30 | 0.078x | 4.66 | 1279.3 |
| `track-5m-48k-24b-clean` | 48k / 24b | 300 | 0.000% | cold | 0.21 | 2.54 | 0.00 | 0.00 | 0.00 | 2.77 | 0.009x | 0.55 | 809.3 |
| `track-5m-48k-24b-clean` | 48k / 24b | 300 | 0.000% | warm | 0.29 | 2.83 | 0.00 | 0.00 | 0.00 | 3.13 | 0.010x | 0.63 | 809.0 |
| `track-5m-96k-24b-mod` | 96k / 24b | 300 | 0.007% | cold | 0.44 | 11.18 | 53.43 | 3.79 | 10.71 | 79.62 | 0.265x | 15.92 | 2447.9 |
| `track-5m-96k-24b-mod` | 96k / 24b | 300 | 0.007% | warm | 0.76 | 11.29 | 49.21 | 3.61 | 6.74 | 71.64 | 0.239x | 14.33 | 2450.9 |

- **5-min aggregate:** RTF p50 `0.079x`, p95 `0.253x`; `wall_s_per_audio_min` p50 `4.74`, p95 `15.21` (cold p95 `14.49`, warm p95 `13.10`); isolated Peak RSS p50 `1218.4 MiB`, p95 `2449.6 MiB`, max `2450.9 MiB`; failures `0`, defect abstentions `0`, control FP `0`.
- **Multi-track album benchmark (4 tracks, 10.0 min stereo FLAC):** wall `118.30 s` (RTF `0.197x`, `11.83 s/audio min`), policy `unified_album_gain` (`-0.20 dB`), all processed `True`, all boundaries passed `True`.

---

## 4. Frozen E5 Preregistration & Pilot Registration Block

Locked **before** opening sealed audio (`corpus/sealed-v1.json` audio remains custodian-held and unexecuted; manifest group metadata is inspected post-pilot strictly for cross-manifest leakage audit). Any edit after freeze requires a new sealed corpus version.

| Pilot Field (`corpus_v1.PILOT_FIELDS`) | Frozen Value | Basis |
|---|---:|---|
| `pilot.sample_size` | `30` | Paired power (`alpha_eff=0.016667`, 80% power, `dev_std=1.2057 dB`, effect `0.75 dB`) |
| `pilot.gate_margin_sdr_db` | `1.0` dB | Practical superiority margin vs `identity` and `gain_only` |
| `pilot.gate_margin_vs_external_db` | `0.5` dB | Locked non-inferiority margin vs `ss_pew_mit__v2_windowed` (with CPU/safety advantage) |
| `pilot.min_coverage` | `0.9` | Minimum blind defect attempt coverage (100% observed on `dev-v1` and `dev-e3-v1`) |
| `pilot.cpu_p95_s_per_audio_min` | `244.2` s/min | `1.25x` max(dev-e3 p95 `195.38`, 5-min p95 `15.21`) |
| `pilot.rss_mb_p95` | `3063.7` MiB | `1.25x` max isolated 5-min peak RSS (`2450.9 MiB`) |
| `pilot.registered_at` | `2026-11-23` | E4 freeze timestamp |

### Locked Confirmatory Hypotheses for E5 (No Post-Hoc Switching Allowed)

- **`H1_vs_identity_and_gain_only` (`superiority`):** aspade_mit__v2_windowed vs identity & gain_only — 95% cluster CI lower bound > +1.0 dB (gate_margin_sdr_db) AND Holm-Bonferroni p < 0.05.
- **`H2_vs_current_production` (`superiority`):** aspade_mit__v2_windowed vs current_snapshot_auto — 95% cluster CI lower bound > 0.0 dB AND Holm-Bonferroni p < 0.05.
- **`H3_vs_strong_external_dsp` (`non_inferiority_with_cpu_safety_advantage`):** aspade_mit__v2_windowed vs ss_pew_mit__v2_windowed — 95% cluster CI lower bound >= -0.50 dB (-gate_margin_vs_external_db) with verified faster or equal local CPU wall_s_per_audio_min p95 and zero stereo safety rubric violations.

### Blind Listening Panel & Promotion Gate

- Equal achievable loudness verified `<= 0.2 LU` after FLAC encode (`scripts/listen_pack.py`), double-blind A/B + BS.1116 artifact anchor against `identity` and `ss_pew_mit__v2_windowed`, custodian-held randomization key, 95% CIs over recording and listener clusters.
- Promotion requires passing **all** engineering gates (0 control FP, 0 polarity violations, unified `-1.0 dBTP` headroom, 0 stereo safety rubric violations, CPU p95 & RSS budgets, `>= 90%` coverage), all 3 locked confirmatory statistical hypotheses on sealed, AND perceptual confirmation on the blind panel. Otherwise verdict is `no-promote` / `inconclusive`.
