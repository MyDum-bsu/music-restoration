# E8 · Voice & Stems Feasibility Study — Separate Evaluation Report

**Card:** Board #1135 · **Spec:** E0 charter + spec v1 (Section E8) · **Contour A (Voice):** `dev-e8-voice-2026-11-23` (`15` cases) · **Contour B (Stems):** `dev-e8-stems-2026-11-23` (`9` cases)

## 1. Scope, Safety Boundaries & Provenance

- **Sealed corpora (`corpus/sealed-v1.json`, `corpus/sealed-e5-v1.json`):** `sealed_corpora_opened = False` (strictly untouched).
- **Frozen E4 Preserve hard-declip pipeline:** `18/18` SHA-256 pins verified untouched (`all_untouched = True`).
- **Preserve promotion / production dependencies / Colab automation:** `False` (isolated CPU feasibility study only).
- **Contour A Manifest SHA-256 (`corpus/dev-e8-voice-v1.json`):** `fbc22f9a2a4f6937d890485a8b78ea67fa3d8700a9658dd8b527d47188d51eec`
- **Contour A Evaluator SHA-256 (`research/e8_voice_eval.py`):** `876e3985bda5b1cbd48013c71accafe02607ecd925a3f81680f308269bf58104`
- **Contour B Manifest SHA-256 (`corpus/dev-e8-stems-v1.json`):** `b6e3ce4c623a355c0e9f5bb2ec6d0090547778442592428eb83812700ae63c07`
- **Contour B Evaluator SHA-256 (`research/e8_stems_eval.py`):** `1c0fd80b587118f8e824cf5d88f0a8ed915a1cc109beead180163fed5c2251ec`
- **Contour B Exact Additive Mix Invariant:** `max(|sum(true_stems) - mix|) = 0.00e+00`, `max(|sum(est_stems) - mix|) = 3.33e-16`.

> **Strict Non-Generalization Warning:** Contour A (Voice) and Contour B (Stems) are **two independent feasibility contours**. Results from speech enhancement or 4-stem source separation **MUST NOT** be generalized to full-mix stereo music mastering (`Preserve` hard-declipping).

---

## 2. Contour A — Voice Restoration Feasibility

### 2.1 CPU Baselines by Stratum (Synthetic Formant + RIR + Noise Dev Corpus)

| Stratum (n) | Arm | SI-SDR In (dB) | SI-SDR Out (dB) | SI-SDRi (dB) | STOI-Proxy Out | ΔSTOI-Proxy | Formant Drift Out (Hz) | ΔLSD (dB) | Syllable Env Corr | Wall (s/min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `reverb_only` (3) | `identity` | -1.89 | -1.89 | +0.00 | 0.891 | +0.000 | 113.1 | +0.00 | 0.731 | 0.00 |
| `reverb_only` (3) | `wpe_dereverb_cpu` | -1.89 | -1.24 | +0.65 | 0.894 | +0.004 | 105.1 | -1.25 | 0.781 | 3.00 |
| `reverb_only` (3) | `spectral_speech_enhancer_cpu` | -1.89 | -1.89 | -0.00 | 0.887 | -0.003 | 115.3 | -0.51 | 0.731 | 0.43 |
| `noise_only` (3) | `identity` | 9.99 | 9.99 | +0.00 | 0.953 | +0.000 | 57.3 | +0.00 | 0.995 | 0.00 |
| `noise_only` (3) | `wpe_dereverb_cpu` | 9.99 | 8.94 | -1.05 | 0.945 | -0.008 | 61.1 | -0.35 | 0.988 | 2.96 |
| `noise_only` (3) | `spectral_speech_enhancer_cpu` | 9.99 | 12.23 | +2.24 | 0.951 | -0.002 | 47.5 | -3.81 | 0.997 | 0.53 |
| `reverb_plus_noise` (3) | `identity` | -1.19 | -1.19 | +0.00 | 0.880 | +0.000 | 146.2 | +0.00 | 0.818 | 0.00 |
| `reverb_plus_noise` (3) | `wpe_dereverb_cpu` | -1.19 | -0.81 | +0.38 | 0.881 | +0.001 | 138.3 | -0.81 | 0.845 | 2.78 |
| `reverb_plus_noise` (3) | `spectral_speech_enhancer_cpu` | -1.19 | -1.05 | +0.14 | 0.870 | -0.010 | 143.6 | -3.51 | 0.817 | 0.42 |
| `clean_voice_control` (3) | `identity` | 60.00 | 60.00 | +0.00 | 1.000 | +0.000 | 0.0 | +0.00 | 1.000 | 0.00 |
| `clean_voice_control` (3) | `wpe_dereverb_cpu` | 60.00 | 60.00 | +0.00 | 1.000 | +0.000 | 0.0 | +0.00 | 1.000 | 0.03 |
| `clean_voice_control` (3) | `spectral_speech_enhancer_cpu` | 60.00 | 60.00 | +0.00 | 1.000 | +0.000 | 0.0 | +0.00 | 1.000 | 0.03 |
| `ood_voice` (3) | `identity` | -3.28 | -3.28 | +0.00 | 0.861 | +0.000 | 177.9 | +0.00 | 0.704 | 0.00 |
| `ood_voice` (3) | `wpe_dereverb_cpu` | -3.28 | -2.82 | +0.46 | 0.864 | +0.003 | 175.1 | -0.62 | 0.773 | 2.76 |
| `ood_voice` (3) | `spectral_speech_enhancer_cpu` | -3.28 | -3.05 | +0.24 | 0.844 | -0.017 | 177.1 | -3.09 | 0.701 | 0.37 |

### 2.2 External Voice Datasets, Neural Models & Licensed Metrics Audit

| Component | Category | Status | License / Version Gate | Colab / Provisioning Note |
|---|---|---|---|---|
| `daps` | Dataset | `not_provisioned_on_host` | CC-BY-NC-4.0 / research terms — verify license before download or Colab upload | Download externally with license verification; keep strictly out of sealed music corpora |
| `voicebank_demand` | Dataset | `not_provisioned_on_host` | CC-BY-4.0 — verify split integrity (no speaker overlap between train/test) | Provision on authorized Colab session or approved dev storage with SHA-256 manifest |
| `deepfilternet3` | Neural Model | `blocked_external_weights_or_colab` | DeepFilterNet 0.5.6 (DeepFilterNet3 48 kHz) (MIT / Apache-2.0 dual-licensed) | T4/L4 GPU (~1-2 GB VRAM, ~1-3 s/min audio; also CPU-capable after weight audit) |
| `sgmse_plus` | Neural Model | `blocked_external_weights_or_colab` | SGMSE+ (Richter et al. 2023, score-based diffusion) (MIT (verify checkpoint release terms)) | T4/L4 GPU (~4-6 GB VRAM, ~15-40 s/min audio for 30-step reverse SDE) |
| `flow_matching_speech` | Neural Model | `blocked_external_weights_or_colab` | FlowSE / conditional flow-matching speech restoration (pin commit + SHA-256) (Verify repository & checkpoint SPDX before download) | T4/L4 GPU (~4-8 GB VRAM, ~8-20 s/min audio for 5-10 ODE steps) |
| `pesq` | Perceptual Metric | `license_required_for_pesq` | Proprietary / patent-encumbered ITU-T reference implementation; requires license check before install | Local CPU proxy: `stoi_proxy_envelope_corr + formant_drift_hz + syllable_env_corr` |
| `dnsmos` | Perceptual Metric | `blocked_external_weights_or_colab` | Microsoft ONNX weights (CC-BY-4.0 / MSR terms) not provisioned locally | Local CPU proxy: `si_sdri_db + spectral_lsd_db + stoi_proxy_envelope_corr` |

### 2.3 Go/No-Go #1 — Voice Profile

- **Production Verdict:** `no_go` (`preserve_stereo_music_generalization = forbidden`).
- **Colab Research Stage Verdict:** `conditional_go` (`cpu_harness_verified = True`).
- **Rationale:** CPU WPE + spectral speech enhancement baselines validate the evaluation harness with positive SI-SDRi on reverberant/noisy synthetic speech and 0.0 error on clean voice negative controls, but production adoption is NO-GO because: (1) real speech corpora (DAPS, VoiceBank+DEMAND) are not provisioned on host, (2) neural speech models (DeepFilterNet3, SGMSE+, Flow) and DNSMOS are blocked pending authorized Colab GPU + checkpoint audit, (3) PESQ requires explicit license clearance, and (4) speech enhancement must never be generalized to stereo music Preserve.
- **Decision Inputs Required for Colab Voice Stage:**
  - Provision licensed DAPS and VoiceBank+DEMAND dev splits with speaker-disjoint manifests and SHA-256 pins.
  - Authorize Google Colab GPU session (separate ticket) with pinned DeepFilterNet3 (MIT/Apache-2.0) and SGMSE+ (MIT) checkpoints.
  - Clear ITU-T P.862 license for PESQ or standardize on open-licensed DNSMOS P.835 + Whisper WER / phoneme-drop hallucination checks.
  - Require >= +3.0 dB SI-SDRi and positive STOI/DNSMOS gain over CPU WPE+Wiener on DAPS/VoiceBank with 0 regressions on clean anechoic controls.
  - Keep any future voice profile strictly isolated from Preserve stereo music mastering (separate opt-in voice route only after product approval).

---

## 3. Contour B — 4-Stem Separation Feasibility

### 3.1 CPU Baselines vs Oracle IRM Upper Bound by Stratum

| Stratum (n) | Arm | Oracle | Vocals SI-SDR (dB) | Drums SI-SDR (dB) | Bass SI-SDR (dB) | Other SI-SDR (dB) | Mean Active SI-SDR (dB) | Mean Leakage (dB) | Mix Recon Max Err | Wall (s/min) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `standard_4stem_mix` (4) | `identity` | `False` | 1.42 | -9.82 | -8.38 | -6.25 | -5.76 | 5.73 | 0.00e+00 | 0.46 |
| `standard_4stem_mix` (4) | `hpss_band_separator_cpu` | `False` | 9.66 | -2.95 | 2.99 | 8.78 | 4.62 | -7.72 | 3.33e-16 | 5.77 |
| `standard_4stem_mix` (4) | `oracle_irm_upper_bound` | `True` | 17.02 | 11.96 | 11.96 | 12.72 | 13.41 | -16.71 | 3.33e-16 | 3.14 |
| `sparse_clean_stem_control` (2) | `identity` | `False` | 60.00 | -1.49 | 0.73 | - | 29.81 | -45.00 | 0.00e+00 | 0.25 |
| `sparse_clean_stem_control` (2) | `hpss_band_separator_cpu` | `False` | 15.56 | 1.81 | 3.74 | - | 9.17 | -50.53 | 2.22e-16 | 4.59 |
| `sparse_clean_stem_control` (2) | `oracle_irm_upper_bound` | `True` | 60.00 | 9.63 | 10.60 | - | 35.06 | -51.65 | 2.78e-16 | 3.03 |
| `ood_stems` (3) | `identity` | `False` | 0.07 | -8.60 | -8.97 | -5.16 | -5.66 | 5.65 | 0.00e+00 | 0.35 |
| `ood_stems` (3) | `hpss_band_separator_cpu` | `False` | 6.08 | -1.42 | 2.93 | -14.90 | -1.83 | -1.62 | 3.33e-16 | 6.29 |
| `ood_stems` (3) | `oracle_irm_upper_bound` | `True` | 14.84 | 12.16 | 11.12 | 11.64 | 12.44 | -15.80 | 3.33e-16 | 3.86 |

### 3.2 External Stem Datasets & Neural Models Audit (HTDemucs / SCNet / BS-RoFormer)

| Component | Category | Status | License / Version Pin | Colab VRAM & Runtime Budget |
|---|---|---|---|---|
| `musdb18_hq` | Dataset | `not_provisioned_on_host` | Non-commercial research license — verify terms before download or Colab upload | Provision approved dev subset on Colab with SHA-256 manifest; never mix with sealed music holdout |
| `htdemucs_v4` | Neural Model | `blocked_external_weights_or_colab` | demucs 4.0.1 (`htdemucs` / `htdemucs_ft` 4-stem bag of 4 models) (MIT) | T4/L4 GPU (~2-4 GB VRAM, ~2-5 s/min audio for `htdemucs`, ~8-18 s/min for `htdemucs_ft`) |
| `scnet` | Neural Model | `blocked_external_weights_or_colab` | SCNet (Tong et al. 2024, Sparse Compression Network — pin commit + SHA-256) (MIT (verify checkpoint release terms)) | T4/L4 GPU (~4-6 GB VRAM, ~4-10 s/min audio) |
| `bs_roformer` | Neural Model | `blocked_external_weights_or_colab` | BS-RoFormer (Lu et al. 2023, Band-Split RoFormer — pin checkpoint SHA-256) (MIT/Apache-2.0 code; MUST verify specific community checkpoint license before download) | T4/L4 GPU (~6-10 GB VRAM, ~10-25 s/min audio with overlap-add) |

### 3.3 Go/No-Go #2 — Stems Restoration

- **Production Verdict:** `no_go` (`preserve_stereo_music_generalization = forbidden`).
- **Colab Research Stage Verdict:** `conditional_go` (`cpu_harness_verified = True`).
- **Rationale:** The 4-stem CPU harness verifies exact additive mix integrity (`mix == sum(stems)`), shows deterministic HPSS+band separation improves over identity while trailing the Oracle IRM upper bound, and exposes OOD collapse when vocal/other spectra clash or stereo cues vanish (mono collapse). Production stem restoration is NO-GO because: (1) neural stem separators (HTDemucs, SCNet, BS-RoFormer) are blocked pending authorized Colab GPU + checkpoint license verification, (2) MUSDB18-HQ is not provisioned on host, and (3) stem separation is outside the Preserve stereo release.
- **Non-Generalization Warning:** STRICT BOUNDARY: Neither Contour A (Voice) nor Contour B (Stems) results may be generalized to stereo music mastering (`Preserve` hard-declipping). Stem separation and re-summing introduces inter-stem bleed and phase/masking artifacts incompatible with Preserve's untouched-outside-mask stereo fidelity contract.
- **Decision Inputs Required for Colab Stems Stage:**
  - Provision licensed MUSDB18-HQ dev split (non-commercial research check) with SHA-256 manifest.
  - Authorize Google Colab T4/L4 GPU session (separate ticket) with pinned HTDemucs v4 (MIT), SCNet (MIT), and license-verified BS-RoFormer checkpoints.
  - Measure per-stem SI-SDR, inter-stem leakage (dB), and exact mix reconstruction error (`sum(est_stems) - mix`) on both dev-e8-stems-v1 and MUSDB18-HQ.
  - Audit stereo phase/IPD preservation and bleed-through artifacts before any opt-in Enhance stem-repair prototype is proposed.
  - Do NOT route stereo mastering (`Preserve`) through stem separation or generalize stem SI-SDR to full-mix hard-declipping.

---

## 4. Limitations & Honest Method Boundaries

1. **Synthetic Dev Corpora Only:** Both `dev-e8-voice-v1` (formant + RIR + noise) and `dev-e8-stems-v1` (4-stem additive stereo mix) are deterministic synthetic benchmarks designed to validate metrics, invariants, and CPU baselines offline. Real speech corpora (`DAPS`, `VoiceBank+DEMAND`) and real multitrack music (`MUSDB18-HQ`) are `not_provisioned_on_host`.
2. **Neural GPU Arms Blocked Locally:** `DeepFilterNet3`, `SGMSE+`, flow-matching speech restoration, `DNSMOS`, `HTDemucs`, `SCNet`, and `BS-RoFormer` are all honestly reported as `blocked_external_weights_or_colab` with zero fabricated numbers. All GPU inference/benchmarks require separate Google Colab authorization.
3. **Licensed Perceptual Metrics:** Official ITU-T P.862 `PESQ` is marked `license_required_for_pesq`; local voice evaluation uses `stoi_proxy_envelope_corr`, `formant_drift_hz`, `spectral_lsd_db`, and `syllable_env_corr` as transparent CPU proxies.
4. **Oracle IRM Is an Upper Bound, Not a Deployable Separator:** `oracle_irm_upper_bound` accesses ground-truth clean stems to establish the magnitude-masking ceiling on `dev-e8-stems-v1`; real separators have no access to clean stems.
5. **OOD Failure Modes Exposed Honestly:** In Contour A, heavy hall reverb at 2 dB SNR and non-stationary babble degrade CPU spectral enhancement; in Contour B, spectral clash (`e8-stems-ood-vocal-other-clash-44k`) and mono collapse (`e8-stems-ood-mono-collapsed-mix-44k`) sharply reduce `hpss_band_separator_cpu` vocal/other separation because spatial Mid/Side cues vanish.
6. **Zero Generalization to Preserve Stereo Mastering:** Neither voice dereverberation/denoising nor 4-stem source separation validates or improves stereo hard-declipping (`Preserve`). Frozen E4 files remain `18/18` untouched.
