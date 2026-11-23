# E6 · Archival De-Click & Hiss: DSP vs Learned Research Report

**Card:** Board #1131 · **Spec:** E0 charter + spec v1 (Section E6) · **Dataset:** `dev-e6-archival-2026-11-23` (`20` stereo cases) · **Learned Arm:** `blocked_external_weights_or_colab`

## 1. Scope, Safety Boundaries & Provenance

- **Sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`):** `sealed_corpora_opened = False` (strictly untouched).
- **Frozen E4 Preserve hard-declip pipeline:** `18/18` SHA-256 pins verified untouched (`all_untouched = True`).
- **Preserve promotion / Colab automation:** `False` (isolated research study only; does not extend Preserve stereo).
- **Manifest SHA-256 (`corpus/dev-e6-archival-v1.json`):** `43be9dbc1ced833b734f8a5b1339c1d8bb68123723b061213a6873f09a67a48a`
- **Corpus Builder SHA-256 (`research/build_e6_archival_corpus.py`):** `cfe13070e747a452682298dd3c6792bef0a0661735520916cc2c9f4cc650d6aa`
- **E6 Harness SHA-256 (`research/e6_declick_hiss.py`):** `653eaf6d2329176c1b62ca4ff28aa79f9f7237d64977bbe7d5467b29f5bee266`
- **Residual & Immutability Contract:** `140/140` runs passed `restored + residual == input` (`max_err = 1.11e-16`) with `0` input mutations.

---

## 2. DSP Arm Comparison by Stratum

### 2.1 Clicks & Crackles (`clicks_only`, n=5, incl. low-level crackle) + Clean Attack Protection (`clean_control`, n=5)

> **Note on Clean Controls:** Ultra-clean controls (`-82 dBFS` floor) trigger the spectral hiss `clean_bypass_dbfs = -70.0` gate (`0 (no edit)`). The `e6-ctl-clean-roomtone-tail-48k` control (`-58 dBFS` studio room-tone) does **not** bypass, exposing active Wiener/MMSE trade-offs (see Section 3.3 and Section 6).

| Arm | Mean ΔSNR (dB) | Mean SI-SDRi (dB) | Click F1 | Clean Attack FP Rate | Clean Attack Residual (dB) | Wall (s/min) |
|---|---:|---:|---:|---:|---:|---:|
| `identity` | +0.00 | +0.00 | 0.000 | 0.0000% | 0 (no edit) | 0.03 |
| `ar_declick` | +16.74 | +16.74 | 0.709 | 0.0000% | 0 (no edit) | 3.74 |
| `median_impulse_declick` | +9.40 | +9.38 | 0.456 | 3.8910% | -19.8 | 0.30 |
| `wiener_hiss` | +0.00 | +0.00 | - | 3.0542% | -48.4 | 3.05 |
| `mmse_stsa_hiss` | +0.00 | +0.00 | - | 2.8900% | -48.7 | 3.11 |
| `ar_declick_wiener_hiss` | +16.74 | +16.74 | 0.709 | 0.0000% | -48.4 | 7.27 |
| `ar_declick_mmse_stsa` | +16.74 | +16.74 | 0.709 | 0.0000% | -48.7 | 7.56 |

### 2.2 Stationary & Modulated Hiss (`hiss_only`, n=4) + Reverb Tail Preservation

| Arm | Mean ΔSNR (dB) | Mean SI-SDRi (dB) | Hiss Tail Drift (dB) | Clean Tail Drift (-82 dBFS) | Room-Tone Tail Drift (-58 dBFS) | Room-Tone Tail SNR Out (dB) | Wall (s/min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `identity` | +0.00 | +0.00 | 20.19 | 0.00 | 0.00 | ∞ (no edit) | 0.03 |
| `ar_declick` | +0.00 | +0.00 | 20.19 | 0.00 | 0.00 | ∞ (no edit) | 3.29 |
| `median_impulse_declick` | +0.00 | +0.00 | 20.19 | 0.00 | 0.00 | ∞ (no edit) | 0.26 |
| `wiener_hiss` | +8.59 | +8.58 | 11.28 | 0.00 | 2.36 | 6.80 | 3.90 |
| `mmse_stsa_hiss` | +6.45 | +6.45 | 13.60 | 0.00 | 2.14 | 7.61 | 5.30 |
| `ar_declick_wiener_hiss` | +8.59 | +8.58 | 11.28 | 0.00 | 2.36 | 6.80 | 7.05 |
| `ar_declick_mmse_stsa` | +6.45 | +6.45 | 13.60 | 0.00 | 2.14 | 7.61 | 7.79 |

### 2.3 Combined Archival Transfers (`combined_archival`, n=3) & OOD Stress Cases (`ood`, n=3)

| Arm | Archival Mean ΔSNR (dB) | Archival SI-SDRi (dB) | Archival Tail Drift (dB) | OOD Mean ΔSNR (dB) | OOD SI-SDRi (dB) | OOD Attack FP Rate |
|---|---:|---:|---:|---:|---:|---:|
| `identity` | +0.00 | +0.00 | 25.09 | +0.00 | +0.00 | 0.0000% |
| `ar_declick` | +0.26 | +0.26 | 25.09 | +1.81 | +1.81 | 0.0032% |
| `median_impulse_declick` | -0.16 | -0.18 | 25.09 | -3.75 | -3.79 | 16.0640% |
| `wiener_hiss` | +5.81 | +5.80 | 15.61 | +0.66 | +0.74 | 0.0000% |
| `mmse_stsa_hiss` | +5.00 | +4.95 | 17.93 | +0.86 | +0.88 | 0.0000% |
| `ar_declick_wiener_hiss` | +6.19 | +6.15 | 15.73 | +2.47 | +2.55 | 0.0032% |
| `ar_declick_mmse_stsa` | +5.07 | +5.02 | 18.08 | +2.67 | +2.69 | 0.0032% |

---

## 3. Sensitivity Diagnostics & Active-Algorithm Trade-Offs

### 3.1 Quiet Crackle Sensitivity (`e6-clicks-low-level-crackle-44k`, amp `0.03-0.08`)

| Arm | ΔSNR (dB) | Click Precision | Click Recall | Click F1 | Detected / True Samples |
|---|---:|---:|---:|---:|---:|
| `ar_declick` | +3.90 | 1.000 | 0.187 | 0.315 | 53 / 283 |
| `median_impulse_declick` | +5.35 | 0.971 | 0.233 | 0.376 | 68 / 283 |

### 3.2 OOD Wide Scratch Fragmentation (`e6-ood-wide-scratch-44k`, width `20-34` samples)

| Arm | ΔSNR (dB) | Click Precision | Click Recall | Click F1 | Detected / True Samples |
|---|---:|---:|---:|---:|---:|
| `ar_declick` | +3.31 | 1.000 | 0.148 | 0.258 | 78 / 526 |
| `median_impulse_declick` | +0.57 | 0.500 | 0.038 | 0.071 | 40 / 526 |

### 3.3 Active Wiener / MMSE-STSA Without Bypass vs Room-Tone Clean Control

| Case / Mode | Arm | Bypassed Ch | Attack FP (`abs(r) > 1e-3`) | Attack Residual (dB) | Quiet Tail Drift (dB) | Tail SNR Out (dB) |
|---|---|---:|---:|---:|---:|---:|
| `e6-ctl-clean-percussion-48k` (bypass disabled) | `wiener_hiss` | 0/2 | 92.59% | -23.0 | - | - |
| `e6-ctl-clean-percussion-48k` (bypass disabled) | `mmse_stsa_hiss` | 0/2 | 90.16% | -24.3 | - | - |
| `e6-ctl-clean-reverb-tail-48k` (bypass disabled) | `wiener_hiss` | 0/2 | - | - | 0.21 | 21.44 |
| `e6-ctl-clean-reverb-tail-48k` (bypass disabled) | `mmse_stsa_hiss` | 0/2 | - | - | 0.17 | 23.71 |
| `e6-ctl-clean-roomtone-tail-48k` (`-58 dBFS` room-tone, default) | `wiener_hiss` | 0/2 | 9.16% | -48.4 | 2.36 | 6.80 |
| `e6-ctl-clean-roomtone-tail-48k` (`-58 dBFS` room-tone, default) | `mmse_stsa_hiss` | 0/2 | 8.67% | -48.7 | 2.14 | 7.61 |

---

## 4. Learned Arm Audit (Historical U-Net / RESTORE) & Colab Contract

- **Status:** `blocked_external_weights_or_colab` (`executed = False`)
- **Blocked Reasons:** `no_local_historical_unet_or_restore_weights, no_local_cuda_gpu, colab_connection_not_authorized_or_configured`
- **Environment Audit:** `torch = 2.14.1+cpu`, `cuda_available = False`, `local_archival_weights_found = 0`, `colab_authorized = False`.
- **Policy Compliance:** No learned metrics fabricated; no weights downloaded or executed without license/checkpoint SHA-256 verification; no Colab session launched.
- **Handoff Interface:** Future authorized Colab run must evaluate on `corpus/dev-e6-archival-v1.json` (`rights.colab == True` only, never sealed corpora), export `restored` + `residual = input - restored` with pinned checkpoint SHA-256 and GPU/CUDA/PyTorch metadata, and pass local verification via `research/e6_declick_hiss.py`.

## 5. Unpaired Real Archival Transfers Protocol

- **Status:** `not_provisioned_in_repo` (`clean_reference_fabricated = False`).
- No rights-cleared real archival vinyl/tape audio files are bundled in the repository. When rights-cleared real transfers are provided locally, evaluation uses loudness-matched (`<= 0.2 LU`) blind A/B preference + isolated residual (`removed = input - restored`) auditioning for attack/cymbal/reverb leakage — never synthetic paired SDR.

## 6. Limitations & Honest Method Boundaries

1. **Quiet Crackle Sensitivity Floor:** `ar_declick` uses `min_abs_residual = 0.035` and a strict transient guard to maintain `0.0000%` false positives on clean sharp percussion/plucked attacks. Consequently, on high-amplitude pops (`0.22-0.72`) it achieves `F1 ~ 0.81` and `~+20 dB ΔSNR`, whereas on quiet vinyl/shellac crackle (`e6-clicks-low-level-crackle-44k`, amp `0.03-0.08`) recall drops to `0.187` (`ΔSNR = +3.90 dB`). Lowering the residual threshold would trade away clean-attack protection.
2. **Role of `clean_bypass_dbfs = -70.0` in Spectral Hiss Reduction:** Zero drift (`0.00 dB`) and zero residual (`0 (no edit)`) on the ultra-clean `-82 dBFS` controls are achieved by the high-frequency clean-floor bypass gate, **not** by Wiener/MMSE-STSA selectivity alone. When bypass is disabled (`clean_bypass_dbfs = -120.0`) or when a clean recording carries realistic `-58 dBFS` room-tone (`e6-ctl-clean-roomtone-tail-48k`), active Wiener/MMSE-STSA modifies both percussive attacks (`clean_attack_residual_energy_db` ≈ `-48.4 dB`) and quiet reverb tails (`quiet_tail_drift_db` ≈ `2.36 dB`).
3. **Partial Fragmentation on OOD Wide Scratches:** On `e6-ood-wide-scratch-44k` (20-34 sample scratches exceeding `max_click_samples = 20`), `ar_declick` does **not** achieve full abstention: the bidirectional AR residual fragments wide scratches into shorter sub-runs (`<= 20` samples) and partially interpolates `78/526` samples (`recall = 0.148`, `ΔSNR = +3.31 dB`), leaving residual scratch artifacts. Wide physical scratches require dedicated multi-scale/gap-inpainting detection rather than single-scale AR run-length gating.
4. **Metric Semantics Across Arm Families:** `transient_fp_rate` counts masked samples inside `attack_mask & ~click_mask` for de-clickers (`ar_declick`, `median_impulse_declick`, and cascade stage 1), whereas for standalone broadband STFT hiss reducers (`wiener_hiss`, `mmse_stsa_hiss`) it counts attack samples with `|residual| > 1e-3`. Cross-family comparison on clean attacks should therefore use `clean_attack_residual_energy_db` alongside `transient_fp_rate`.
5. **Non-Stationary Mains Hum & Learned Comparison:** Broadband spectral hiss reducers do not remove tonal 50/100/150 Hz mains hum (`e6-ood-nonstationary-hum-hiss-48k`), which requires a dedicated harmonic notch stage; and comparison against learned models (Historical U-Net / RESTORE) remains blocked until licensed weights and authorized Colab GPU execution are provisioned.
