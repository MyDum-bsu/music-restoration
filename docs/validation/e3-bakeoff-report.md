# E3 · Hard-Declipping Bake-Off & Dev Decision

**Card:** Board #1125 · **Spec:** E0 charter + spec v1 (2026-11-23), Section 4 & E3 · **Sealed opened:** `false` (blocked and untouched).

> **Provenance note:** This report is a curated multi-corpus synthesis built directly from two unmodified `research/e3_bakeoff.py` runs (`corpus/dev-v1.json` and `corpus/dev-e3-v1.json`) pinned to the exact committed implementation SHA-256 hashes in [`docs/validation/e3-manifests.json`](e3-manifests.json).

## Executive Dev Decision

- **Verdict:** `select_dev_candidate_for_e4_freeze`
- **Dev selection gate basis:** `exploratory_dev_bootstrap_ci_95` (exploratory unadjusted 95% cluster bootstrap CI `> 0` plus hard zero-FP / zero-violation / zero-failure guards; Student-t 95% CIs and Holm–Bonferroni adjusted `p`-values are reported alongside; formal confirmatory gates are preregistered in E4 before opening sealed in E5)
- **Selected dev candidate:** `aspade_mit__v2_windowed` (`rnext.dsp.declip_methods.declip_aspade` + `rnext.dsp.clip_detector.detect_hard_clip_v2`)
- **Baseline:** `current_snapshot_auto` (`rnext.dsp.repair.select_restoration` + `rnext.dsp.sparse.hard_clip_mask`)
- **Secondary external DSP comparators:** `ss_pew_mit` (psychoacoustically-weighted IRL1 synthesis via Douglas–Rachford) and `reg_ar_mit` (Tikhonov-regularized Janssen AR declipper)
- **Neural / Google Colab arm:** `not_run` (`colab_connection_not_authorized_or_configured`; per spec v1 Section 5, separate Colab authorization/setup is required before GPU neural runs and does not block the local CPU bake-off)
- **Scope caveat:** Exploratory dev candidate selection only. Neither DR, peak reduction, nor dev waveform SDR proves perceptual preference or release readiness. Promotion requires E4 stereo/album safety + CPU freeze and E5 sealed holdout + blind listening panel.

### Why `aspade_mit__v2_windowed` over `reg_ar_mit` and `ss_pew_mit`

Evaluating **two** independent stereo dev corpora prevented an overfitting trap:

1. On `corpus/dev-v1.json` (`synth/v1`, 5-sinusoid tonal waveforms + sparse transients, 5 defect clusters + 3 controls), `reg_ar_mit` achieved the highest mean masked ΔSDR (`+26.46 dB` oracle, `+22.45 dB` blind) because a 5-sinusoid sum is an exact low-order autoregressive process (though with `n = 5`, Student-t 95% CI vs `current_snapshot_auto` spans `[-7.47, +50.02] dB`, Holm–Bonferroni `p = 0.8750`).
2. On `corpus/dev-e3-v1.json` (`synth/v2-stereo-music`, 24+ simultaneous harmonic partials across 5 stereo-panned voices + broadband percussive hits, 12 defect clusters + 6 controls across asymmetric, sub-full-scale, rare, dithered, single-channel, quiet-tail, late >90s, and 44.1/48/96 kHz 16/24-bit slices), **`reg_ar_mit` regressed below `current_sparse` by `-1.40 dB`** (oracle `+4.99 dB` vs `+6.39 dB`; 95% bootstrap CI `[-2.58, -0.21] dB`), demonstrating that pure AR struggles on dense polyphonic mixes.
3. **`aspade_mit` is the ONLY solver that beat `current_sparse` strongly and consistently on BOTH stereo dev corpora**, winning on **10/12** polyphonic stereo recordings in `dev-e3-v1` (trailing only on `e3-rare-trans-48k-24b` by `-0.21 dB` and `e3-late-defect-44k-16b` by `-0.10 dB`, with `+1.09 dB` oracle mean, 95% bootstrap CI `[+0.70, +1.43] dB`, Student-t CI `[+0.65, +1.53] dB`, Holm–Bonferroni `p = 0.0176`) and **5/5** recordings in `dev-v1` (`+3.05 dB` oracle mean, 95% bootstrap CI `[+0.86, +5.53] dB`), while cutting CPU wall time substantially at both p50 and p95.
4. Pairing `aspade_mit` with `detect_hard_clip_v2` raises end-to-end defect coverage from **40.0% → 100.0%** on `dev-v1` and **83.3% → 100.0%** on `dev-e3-v1`, with **0/9 false positives across all negative controls** (`clean`, `clean_percussive`, `clean_quiet_tail`, `intentional_saturation`, `soft_limiter`) and **0 polarity violations**.

---

## 1. Unified Protocol, Normalization, and Corpora

- **Gain / delay / headroom policy (identical across all methods):** `delay_samples = 0` (no per-method time shift), single stereo whole-track least-squares scalar gain `gain_to_clean` shared across both channels, epsilon `1e-20`, output headroom ceiling `-1.0 dBTP` via `true_peak_dbfs` (`rnext.dsp.declip_methods.apply_headroom_policy` for repair arms, and the same `-1.0 dBTP` true-peak attenuation for `gain_only`). Both gain-controlled and uncompensated masked SDR are stored; `gain_only` achieves exactly `+0.00 dB` masked ΔSDR, `+0.00 dB` whole-track SI-SDRi, `0` pre-gain polarity violations, and `0.0` pre-gain reliable diff.
- **Pre-gain invariant accounting:** Reliable-sample diff and polarity inequalities are evaluated strictly on pre-gain candidate PCM. Arms that only expose post-gain delivered PCM (`current_snapshot_auto`) record `null` (`pre_gain_candidate_unavailable`) rather than evaluating artifacts on gain-scaled output.
- **All-input vs conditional accounting:** Every failure (`error`, `qc_fallback_original`) and abstention (`kept_original`) is counted as identity (`0.0 dB` delta) in all-input averages, CIs, and risk–coverage curves; conditional means are reported alongside coverage.
- **Memory & AURC reporting caveats:** `Peak RSS (MB)` reflects process-level `ru_maxrss` across the multi-arm benchmark process (not isolated per-method peak RSS; per-call `rss_delta_mb_p95` is stored in `e3-manifests.json`). AURC integrates selective regression risk over each detector's achieved coverage interval (`[0, 0.833]` for `v1_current` vs `[0, 1.0]` for `v2_windowed` on `dev-e3-v1`), so AURC values are only directly comparable between arms at matched coverage.
- **Zero sealed leakage:** Cross-manifest group leakage (`take`, `fingerprint`, `session`, `artist`, `album`, `stem_family`) was verified empty across `[smoke-v1.json, dev-v1.json, dev-e3-v1.json, sealed-v1.json]`. `sealed-v1.json` remains custodian-held and unexecuted.
- **Mono SQAM smoke reference (not a stereo substitute):** Historical mono SQAM holdout outputs (`docs/validation/clipping-holdout.csv`, 7 mono recordings at 15 dB input SDR) showed published mono A-SPADE and SS-PEW leading our prototype by median `+1.10 dB` and `+3.17 dB`, respectively. Those published mono outputs serve only as smoke context; all E3 decisions rest on same-input stereo runs below.
- **Reference CPU:** `x86_64` (8 logical cores), single-threaded BLAS/OpenMP (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), Python `3.12.3`, NumPy `2.5.2`, SciPy `1.17.1`, FFmpeg `125907-ga7e72069f1-20260802`, libflac `1.4.3`.

---

## 2. Stereo Bake-Off Results — Extended Musical Dev Corpus (`corpus/dev-e3-v1.json`, n = 18: 12 defect clusters + 6 controls)

### 2.1 Oracle-Mask Comparison (All-Input Defect Accounting)

| Method | Mean Masked SDR (dB) | Mean ΔMasked SDR (dB) | Median ΔMasked SDR (dB) | Worst ΔMasked (dB) | Mean SI-SDRi (dB) | Polarity Viol | Wall p50 (s/min) | Wall p95 (s/min) | Proc Max RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `identity` | 14.91 | +0.00 | +0.00 | +0.00 | +0.00 | 0 | 0.02 | 0.04 | 903.0 |
| `gain_only` | 14.91 | +0.00 | +0.00 | +0.00 | +0.00 | 0 | 0.57 | 1.15 | 903.0 |
| `current_sparse` | 21.30 | +6.39 | +7.29 | -2.56 | +6.68 | 0 | 112.34 | 844.86 | 903.0 |
| **`aspade_mit`** | **22.39** | **+7.48** | **+8.34** | **-2.67** | **+7.77** | **0** | **13.14** | **254.72** | **903.0** |
| `ss_pew_mit` | 21.80 | +6.88 | +7.94 | -2.28 | +7.13 | 0 | 14.69 | 260.39 | 903.0 |
| `reg_ar_mit` | 19.90 | +4.99 | +5.69 | -1.66 | +5.15 | 0 | 5.82 | 87.87 | 903.0 |

### 2.2 Blind Detector Comparison (`dev-e3-v1`)

| Detector | Sample Micro P | Sample Micro R | Sample Micro F1 | Macro Defect R | Case Recall | Control FP | Control FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v1_current` | 0.9998 | 0.8211 | 0.9017 | 0.8265 | 10/12 (83.3%) | 0 | 0.0% |
| **`v2_windowed`** | **0.9998** | **0.9993** | **0.9996** | **0.9990** | **12/12 (100.0%)** | **0** | **0.0%** |

`v1_current` missed `e3-rare-trans-48k-24b` (rare transient clip) and `e3-late-defect-44k-16b` (late clip after 90s below earlier unclipped peak). `v2_windowed` achieved `>= 0.991` sample recall on every single defect slice and `0/6` false positives across clean tonal, clean percussive, clean quiet-tail, intentional `tanh` saturation (drive 4.0 and 5.5), and soft limiter controls.

### 2.3 End-to-End Blind Pipeline Comparison (`dev-e3-v1`, Failures/Abstentions = Identity)

| Blind Arm | Coverage | Fail | Control FP | All-Input Mean ΔMasked (dB) | Cond Mean ΔMasked (dB) | All-Input Mean SI-SDRi (dB) | Worst ΔMasked (dB) | AURC (Reg Risk)* | Wall p50 / p95 (s/min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `current_snapshot_auto` | 10/12 (83.3%) | 0 | 0 | +6.35 | +7.62 | +6.47 | +0.00 | 0.0000 | 103.38 / 667.52 |
| `aspade_mit__v1_current` | 10/12 (83.3%) | 0 | 0 | +7.35 | +8.82 | +7.47 | +0.00 | 0.0000 | 14.53 / 290.83 |
| `ss_pew_mit__v1_current` | 10/12 (83.3%) | 0 | 0 | +6.74 | +8.09 | +6.83 | +0.00 | 0.0000 | 15.18 / 226.22 |
| `reg_ar_mit__v1_current` | 10/12 (83.3%) | 0 | 0 | +4.81 | +5.78 | +4.87 | +0.00 | 0.0000 | 5.98 / 94.75 |
| `current_sparse__v2_windowed` | 12/12 (100.0%) | 0 | 0 | +6.30 | +6.30 | +6.58 | -2.56 | 0.0073 | 102.71 / 475.47 |
| **`aspade_mit__v2_windowed`** | **12/12 (100.0%)** | **0** | **0** | **+7.32** | **+7.32** | **+7.61** | **-2.67** | **0.0073** | **12.82 / 195.38** |
| `ss_pew_mit__v2_windowed` | 12/12 (100.0%) | 0 | 0 | +6.70 | +6.70 | +6.94 | -2.28 | 0.0073 | 14.59 / 218.16 |
| `reg_ar_mit__v2_windowed` | 12/12 (100.0%) | 0 | 0 | +4.92 | +4.92 | +5.09 | -1.66 | 0.0073 | 6.56 / 91.29 |

*\*AURC integrates over each detector's achieved coverage interval (`<= 0.833` for `v1_current`, `1.0` for `v2_windowed`); `v1` avoids the `-2.67 dB` late-clip case solely by missing it completely.*

### 2.4 Paired 95% Cluster Confidence Intervals (`dev-e3-v1`, n = 12 Independent Recording Clusters)

| Planned Comparison (Masked ΔSDR) | Mean Diff (dB) | Median Diff (dB) | Bootstrap 95% CI | Student-t 95% CI | Worst Cluster (dB) | Sign-Flip p | Holm–Bonf p |
|---|---:|---:|---|---|---|---:|---:|
| `oracle: aspade_mit vs identity` | +7.48 | +8.34 | `[+4.91, +9.58]` | `[+4.70, +10.26]` | `t-e3-12` (-2.67) | 0.0015 | **0.0176** |
| `oracle: aspade_mit vs gain_only` | +7.48 | +8.34 | `[+4.91, +9.58]` | `[+4.70, +10.26]` | `t-e3-12` (-2.67) | 0.0015 | **0.0176** |
| `oracle: aspade_mit vs current_sparse` | **+1.09** | **+1.28** | **`[+0.70, +1.43]`** | **`[+0.65, +1.53]`** | `t-e3-06` (-0.21) | 0.0020 | **0.0176** |
| `blind: aspade_mit__v2_windowed vs current_snapshot_auto` | **+0.97** | **+1.26** | **`[+0.23, +1.46]`** | **`[+0.21, +1.74]`** | `t-e3-12` (-2.67) | **0.0225** | 0.1348 |
| `oracle: ss_pew_mit vs current_sparse` | +0.49 | +0.39 | `[+0.01, +0.98]` | `[-0.07, +1.06]` | `t-e3-10` (-1.03) | 0.0820 | 0.2661 |
| `blind: ss_pew_mit__v2_windowed vs current_snapshot_auto` | +0.35 | +0.51 | `[-0.31, +0.94]` | `[-0.39, +1.08]` | `t-e3-12` (-2.28) | 0.3164 | 0.6328 |
| `oracle: reg_ar_mit vs current_sparse` | -1.40 | -1.57 | `[-2.58, -0.21]` | `[-2.83, +0.02]` | `t-e3-02` (-4.28) | 0.0532 | 0.2661 |
| `blind: reg_ar_mit__v2_windowed vs current_snapshot_auto` | -1.43 | -1.63 | `[-2.63, -0.17]` | `[-2.89, +0.03]` | `t-e3-05` (-4.01) | 0.0596 | 0.2661 |

---

## 3. Stereo Bake-Off Results — Official E2 Dev Corpus (`corpus/dev-v1.json`, n = 8: 5 defect clusters + 3 controls)

### 3.1 Oracle-Mask & Blind Summary (`dev-v1`, Proc Max RSS `944.9 MB`)

| Arm | Coverage | Control FP | Mean ΔMasked SDR (dB) | Median ΔMasked SDR (dB) | Mean SI-SDRi (dB) | Worst ΔMasked (dB) | Wall p50 (s/min) | Wall p95 (s/min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle `identity` | — | 0 | +0.00 | +0.00 | +0.00 | +0.00 | 0.02 | 0.21 |
| Oracle `gain_only` | — | 0 | +0.00 | +0.00 | +0.00 | +0.00 | 0.99 | 2.78 |
| Oracle `current_sparse` | 5/5 | 0 | +7.54 | +10.18 | +6.87 | -3.24 | 222.42 | 949.53 |
| Oracle `aspade_mit` | 5/5 | 0 | +10.59 | +15.01 | +9.92 | -3.24 | 6.23 | 13.56 |
| Oracle `ss_pew_mit` | 5/5 | 0 | +18.48 | +16.71 | +17.79 | -3.24 | 8.99 | 13.34 |
| Oracle `reg_ar_mit` | 5/5 | 0 | +26.46 | +26.46 | +25.93 | -3.15 | 2.76 | 9.22 |
| Blind `current_snapshot_auto` (`v1`) | 2/5 (40%) | 0 | +1.17 | +0.00 | +0.91 | -3.16 | 6.80 | 154.82 |
| Blind `aspade_mit__v2_windowed` | **5/5 (100%)** | **0** | **+10.33** | **+14.79** | **+9.70** | -3.24 | 8.55 | **15.21** |
| Blind `ss_pew_mit__v2_windowed` | 5/5 (100%) | 0 | +15.72 | +13.45 | +15.09 | -3.24 | 9.50 | 13.42 |
| Blind `reg_ar_mit__v2_windowed` | 5/5 (100%) | 0 | +22.45 | +19.52 | +21.97 | -3.14 | 3.22 | 10.29 |

- **Detector improvement on `dev-v1`:** `v1_current` micro-recall `0.2343` (macro `0.1636`, case recall `2/5 = 40%`, 0/3 control FP) → **`v2_windowed` micro-recall `0.9988`** (macro `0.9199`, case recall `5/5 = 100%`, micro-precision `1.0000`, 0/3 control FP).
- **Paired 95% cluster CIs (`dev-v1`, n = 5):**
  - `oracle: aspade_mit vs current_sparse`: `+3.05 dB`, bootstrap CI `[+0.86, +5.53] dB`, Student-t CI `[-0.80, +6.91] dB`, Holm–Bonferroni `p = 0.8750`.
  - `blind: aspade_mit__v2_windowed vs current_snapshot_auto`: `+9.16 dB`, bootstrap CI `[+3.64, +15.00] dB`, Student-t CI `[-0.01, +18.33] dB`, Holm–Bonferroni `p = 1.0000`.

---

## 4. Negative Results, Worst Cases, and Protocol Diagnostics

Honest reporting of all regressions and failure modes (included unmodified in every table and CI above):

1. **Localized gain-jump `late_clip` regression across all 4 solvers:**
   - Exact oracle ΔMasked SDR values:
     - `dev-clip-eval-c-late90-44k` (`defect_gain = 8.0`): `current_sparse -3.24 dB`, `aspade_mit -3.24 dB`, `ss_pew_mit -3.24 dB`, `reg_ar_mit -3.15 dB`.
     - `e3-late-defect-44k-16b` (`defect_gain = 2.5`): `current_sparse -2.56 dB`, `aspade_mit -2.67 dB`, `ss_pew_mit -2.28 dB`, `reg_ar_mit -1.66 dB`.
   - **Root cause:** In `research/corpus_synth.py::apply_operator`, `late_clip` multiplies ONLY the post-90s segment by `defect_gain` *before* clipping, while leaving `[0..90s]` at `1.0x` scale and comparing against the unboosted `1.0x` `clean` waveform under a single whole-track scalar gain (`gain_to_clean ≈ 1.0`). Consequently, every consistent declipper reconstructs the true peaks of `defect_gain * clean[90s:]` — which moves the repaired peaks *further* from `1.0x * clean[90s:]`.
   - **Action for E4/E5:** Keep `dev-v1` untouched for hash compatibility, and ensure any future sealed or E4 localized-clipping test compares against the pre-clip local signal or induces clipping via a sub-peak threshold without an unrecorded local gain step.
2. **`reg_ar_mit` underperformance on polyphonic mixes (`dev-e3-v1`):**
   - While `reg_ar_mit` dominates sparse 5-sinusoid signals (`dev-v1`, up to `+60.62 dB` on 96 kHz), it trails `aspade_mit` by `-2.49 dB` and `current_sparse` by `-1.40 dB` on 24+-partial polyphonic stereo mixes (`worst vs current_sparse`: `-4.28 dB` on heavy -9 dBFS clipping `e3-dense-heavy-48k-24b`). Therefore `reg_ar_mit` is NOT selected as the general-purpose stereo candidate.
3. **`v1_current` blind detector failure on rare, dithered, and late-onset clips:**
   - Global-maximum anchoring (`values.max()`) misses late clips whenever an earlier unclipped peak exceeds the clip threshold, and misses 16-bit TPDF-dithered/rare plateaus because `background = max(1.0, shoulder / 48)` destroys SNR when the true shoulder density is `< 0.1` samples/bin.
4. **Rare transient single-frame loss (`e3-rare-trans-48k-24b`):**
   - On extremely sparse broadband percussive hits (`< 0.01%` masked samples), `aspade_mit` (`+1.09 dB`) trails `current_sparse` (`+1.30 dB`) by `-0.21 dB` and `reg_ar_mit` (`+2.14 dB`) by `-1.05 dB` because short impulsive hits have broad STFT spectra.
5. **Neural / Google Colab arm:** `not_run` (`colab_connection_not_authorized_or_configured`). No GPU runtime or neural claims are made.

---

## 5. Baseline & Candidate Manifests and E4 Handoff

Exact SHA-256 implementation pins, corpus digests, toolchain pins, and full JSON summary tables are committed in [`docs/validation/e3-manifests.json`](e3-manifests.json).

- **Baseline (`current_snapshot_auto`):** `rnext.dsp.repair.select_restoration` + `rnext.dsp.sparse.hard_clip_mask` (preserved unmodified on branch).
- **Candidate (`aspade_mit__v2_windowed`):**
  - Detector: `rnext.dsp.clip_detector.detect_hard_clip_v2(window_s=2.0, hop_s=1.0, min_confidence=0.5)`
  - Solver: `rnext.dsp.declip_methods.declip_aspade(frame_44k=2048, redundancy=2, step_k=4, max_iter=120, eps=1e-3)`
  - Headroom: `rnext.dsp.declip_methods.apply_headroom_policy(ceiling_db=-1.0, max_attenuation_db=-24.0)`
- **Next step (E4):** Stereo phase/IPD, M/S, mono-sum, onset/pre-echo, gapless/album-level safety checks, 5-minute track CPU p95/RSS profiling, and pilot parameter freeze prior to E5 sealed release.
