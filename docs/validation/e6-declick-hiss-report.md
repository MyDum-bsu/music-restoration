# E6 · Archival De-Click & Hiss: DSP vs Learned Research Report

**Card:** Board #1131 · **Spec:** E0 charter + spec v1 (Section E6) · **Dataset:** `dev-e6-archival-2026-11-23` (`18` stereo cases) · **Learned Arm:** `blocked_external_weights_or_colab`

## 1. Scope, Safety Boundaries & Provenance

- **Sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`):** `sealed_corpora_opened = False` (strictly untouched).
- **Frozen E4 Preserve hard-declip pipeline:** `18/18` SHA-256 pins verified untouched (`all_untouched = True`).
- **Preserve promotion / Colab automation:** `False` (isolated research study only; does not extend Preserve stereo).
- **Manifest SHA-256 (`corpus/dev-e6-archival-v1.json`):** `2194a576fc01ff0b6c0e9398f6817f758cb2fdd80aa0776ec320c0c8a8721201`
- **Corpus Builder SHA-256 (`research/build_e6_archival_corpus.py`):** `6ead3777e1a9da837942df59e0e72d4157173e0e62b8967c58298acb930d804b`
- **E6 Harness SHA-256 (`research/e6_declick_hiss.py`):** `cc9c2e59864c6d835b4f4277b3865cc8ceaeeeae2924b793029cace8fe850bf1`
- **Residual & Immutability Contract:** `126/126` runs passed `restored + residual == input` (`max_err = 1.11e-16`) with `0` input mutations.

---

## 2. DSP Arm Comparison by Stratum

### 2.1 Clicks & Crackles (`clicks_only`, n=4) + Clean Attack Protection (`clean_control` percussive/plucked)

| Arm | Mean ΔSNR (dB) | Mean SI-SDRi (dB) | Click F1 | Clean Attack FP Rate | Clean Attack Residual (dB) | Wall (s/min) |
|---|---:|---:|---:|---:|---:|---:|
| `identity` | +0.00 | +0.00 | 0.000 | 0.0000% | -228.6 | 0.03 |
| `ar_declick` | +19.95 | +19.95 | 0.808 | 0.0000% | -228.6 | 3.97 |
| `median_impulse_declick` | +9.63 | +9.61 | 0.476 | 3.1618% | -22.3 | 0.30 |
| `wiener_hiss` | +0.00 | +0.00 | - | 0.0000% | -228.6 | 3.30 |
| `mmse_stsa_hiss` | +0.00 | +0.00 | - | 0.0000% | -228.6 | 2.91 |
| `ar_declick_wiener_hiss` | +19.95 | +19.95 | 0.808 | 0.0000% | -228.6 | 7.02 |
| `ar_declick_mmse_stsa` | +19.95 | +19.95 | 0.808 | 0.0000% | -228.6 | 7.28 |

### 2.2 Stationary & Modulated Hiss (`hiss_only`, n=4) + Reverb Tail Preservation

| Arm | Mean ΔSNR (dB) | Mean SI-SDRi (dB) | Hiss Tail Drift (dB) | Clean Tail Drift (dB) | Clean Tail SNR Loss (dB) | Wall (s/min) |
|---|---:|---:|---:|---:|---:|---:|
| `identity` | +0.00 | +0.00 | 20.19 | 0.00 | 0.00 | 0.03 |
| `ar_declick` | +0.00 | +0.00 | 20.19 | 0.00 | 0.00 | 3.97 |
| `median_impulse_declick` | +0.00 | +0.00 | 20.19 | 0.00 | 0.00 | 0.33 |
| `wiener_hiss` | +8.59 | +8.58 | 11.28 | 0.00 | 0.00 | 4.51 |
| `mmse_stsa_hiss` | +6.45 | +6.45 | 13.60 | 0.00 | 0.00 | 7.12 |
| `ar_declick_wiener_hiss` | +8.59 | +8.58 | 11.28 | 0.00 | 0.00 | 7.85 |
| `ar_declick_mmse_stsa` | +6.45 | +6.45 | 13.60 | 0.00 | 0.00 | 9.19 |

### 2.3 Combined Archival Transfers (`combined_archival`, n=3) & OOD Stress Cases (`ood`, n=3)

| Arm | Archival Mean ΔSNR (dB) | Archival SI-SDRi (dB) | Archival Tail Drift (dB) | OOD Mean ΔSNR (dB) | OOD SI-SDRi (dB) | OOD Attack FP Rate |
|---|---:|---:|---:|---:|---:|---:|
| `identity` | +0.00 | +0.00 | 25.09 | +0.00 | +0.00 | 0.0000% |
| `ar_declick` | +0.26 | +0.26 | 25.09 | +1.81 | +1.81 | 0.0032% |
| `median_impulse_declick` | -0.20 | -0.23 | 25.09 | -3.88 | -3.92 | 16.0640% |
| `wiener_hiss` | +5.81 | +5.80 | 15.61 | +0.66 | +0.74 | 0.0000% |
| `mmse_stsa_hiss` | +5.00 | +4.95 | 17.93 | +0.86 | +0.88 | 0.0000% |
| `ar_declick_wiener_hiss` | +6.19 | +6.15 | 15.73 | +2.47 | +2.55 | 0.0032% |
| `ar_declick_mmse_stsa` | +5.07 | +5.02 | 18.08 | +2.67 | +2.69 | 0.0032% |

---

## 3. Learned Arm Audit (Historical U-Net / RESTORE) & Colab Contract

- **Status:** `blocked_external_weights_or_colab` (`executed = False`)
- **Blocked Reasons:** `no_local_historical_unet_or_restore_weights, no_local_cuda_gpu, colab_connection_not_authorized_or_configured`
- **Environment Audit:** `torch = 2.14.1+cpu`, `cuda_available = False`, `local_archival_weights_found = 0`, `colab_authorized = False`.
- **Policy Compliance:** No learned metrics fabricated; no weights downloaded or executed without license/checkpoint SHA-256 verification; no Colab session launched.
- **Handoff Interface:** Future authorized Colab run must evaluate on `corpus/dev-e6-archival-v1.json` (`rights.colab == True` only, never sealed corpora), export `restored` + `residual = input - restored` with pinned checkpoint SHA-256 and GPU/CUDA/PyTorch metadata, and pass local verification via `research/e6_declick_hiss.py`.

## 4. Unpaired Real Archival Transfers Protocol

- **Status:** `not_provisioned_in_repo` (`clean_reference_fabricated = False`).
- No rights-cleared real archival vinyl/tape audio files are bundled in the repository. When rights-cleared real transfers are provided locally, evaluation uses loudness-matched (`<= 0.2 LU`) blind A/B preference + isolated residual (`removed = input - restored`) auditioning for attack/cymbal/reverb leakage — never synthetic paired SDR.

## 5. Key Research Takeaways

1. **AR De-Click + Transient Guard vs Naive Median Impulse:** Bidirectional AR prediction (`min(|e_f|, |e_b|)`) with pre/post energy-jump and impulsiveness gating removes short clicks/crackles with exact bit-preservation outside the click mask while keeping false positives on clean sharp percussive/plucked attacks near zero, whereas naive median/high-pass impulse detection triggers significantly more false positives on sharp musical attacks.
2. **Spectral Hiss Reduction (Wiener vs Ephraim-Malah MMSE-STSA):** Both Wiener and MMSE-STSA with minimum-statistics tracking, clean-floor bypass (`< -70 dBFS`), and a `-11 dB` gain floor improve SNR on stationary and modulated hiss while preserving clean reverb tails (`0.00 dB` drift on clean controls).
3. **Two-Stage Cascade (`ar_declick -> hiss_reducer`):** Running AR de-click before spectral hiss reduction prevents impulsive clicks from smearing across STFT frames and biasing noise-floor estimation, delivering the strongest overall ΔSNR / SI-SDRi on combined vinyl/tape/78 rpm archival profiles while exposing an exact auditionable `residual = input - restored`.
4. **OOD Limits:** Wide physical scratches (`20-34` samples) exceed the conservative click-width guard (`max_click_ms = 0.45 ms`) by design so the de-clicker abstains rather than hallucinating or destroying musical transients; non-stationary mains hum (50/100/150 Hz) requires a dedicated harmonic notch stage rather than broadband hiss suppression.
