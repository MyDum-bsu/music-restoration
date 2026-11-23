# E5 · Sealed Holdout Evaluation, Blind Listening Panel & Release Decision

**Card:** Board #1129 · **Spec:** E0 charter + spec v1 (Section 4 & E5) · **Release Verdict:** `no_promote`

## 1. Freeze Integrity, Custodian Authorization & Corpus Provenance

- **Frozen Candidate:** `aspade_mit__v2_windowed` (ZERO DSP edits; ZERO tuning on sealed)
- **Verified E4 `freeze_sha256`:** `7c4847d4d572321cb93b4d126dfde009e832890d6b652700047287fb6dff0e55` (`verified = True`)
- **18 Frozen Implementation SHA-256 Pins:** `18/18` exact match before unseal
- **Custodian Authorization Date:** `2026-11-23` · **Token SHA-256 prefix:** `f157263b95abd7d8`
- **Authorization Reason:** Owner interactive custodian authorization (2026-11-23) for single-use E5 evaluation of frozen E4 candidate aspade_mit__v2_windowed (freeze_sha256=7c4847d4d572321cb93b4d126dfde009e832890d6b652700047287fb6dff0e55)
- **Evaluated Sealed Manifest:** `sealed-e5-2026-11-23` (`sealed-e5-v1.json`, SHA-256 `147e56a53f2fa705...`, `n = 30` independent defect clusters + `10` negative controls)
- **Cross-manifest group leakage (`smoke/dev/dev-e3/sealed-v1/sealed-e5`):** `0` errors
- **Neural / Google Colab arm:** `not_run` (`colab_connection_not_authorized_or_configured`; excluded from CPU release claim)

> **Honest Custodian Audit of `corpus/sealed-v1.json`:** status = `not_provisioned_on_host` (`0/4` files pinned, `0` present on host, `pilot_registered = False`). The E2 stub `corpus/sealed-v1.json` (`sealed-real-01`, `sealed-real-02`) remains untouched so E2/E4 gate tests remain valid; the powered confirmatory evaluation below runs once on the post-freeze independent 40-cluster sealed holdout `sealed-e5-v1.json` matching E4 `pilot.sample_size = 30`.

---

## 2. Hard Engineering Gates (Frozen E4 Preregistration)

| Engineering Gate | Observed | Frozen Limit | Verdict |
|---|---:|---:|---|
| Negative-control False Positives | `0` | `<= 0` | `PASS` |
| Polarity Violations (pre-gain) | `0` | `<= 0` | `PASS` |
| Reliable Sample Max Diff (pre-gain) | `0.0e+00` | `<= 0.0` | `PASS` |
| Unified `-1.0 dBTP` Ceiling Violations (`+0.05 dB` tol) | `0` | `0` | `PASS` |
| E4 Stereo Safety Rubric Violations | `1` | `<= 0` | `FAIL` |
| Blind Defect Coverage | `100.0%` | `>= 90.0%` | `PASS` |
| Local CPU p95 (`wall_s / audio_min`) | `143.29` | `<= 244.2` | `PASS` |
| Isolated Subprocess Peak RSS p95 (MiB) | `466.6` | `<= 3063.7` | `PASS` |

---

## 3. Sealed Holdout Arm Comparison (All-Input Defect Accounting, Failures/Abstentions = 0.0 dB)

| Arm | Coverage | Fail | Ctrl FP | All-Input Mean ΔMasked (dB) | Median ΔMasked (dB) | Worst ΔMasked (dB) | Mean SI-SDRi (dB) | Pol Viol | Stereo Gate Viol | Wall p50 / p95 (s/min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `identity` | 0/30 (0.0%) | 0 | 0 | +0.00 | +0.00 | +0.00 | +0.00 | 0 | 6 | 0.01 / 0.01 |
| `gain_only` | 0/30 (0.0%) | 0 | 0 | +0.00 | +0.00 | +0.00 | +0.00 | 0 | 14 | 0.32 / 0.65 |
| `current_snapshot_auto` | 27/30 (90.0%) | 0 | 0 | +5.75 | +5.93 | +0.00 | +5.89 | - | 40 | 162.33 / 381.14 |
| `aspade_mit__v2_windowed` | 30/30 (100.0%) | 0 | 0 | +6.89 | +7.46 | +1.65 | +7.01 | 0 | 1 | 15.18 / 143.29 |
| `ss_pew_mit__v2_windowed` | 30/30 (100.0%) | 0 | 0 | +6.30 | +6.35 | +1.77 | +6.39 | 0 | 1 | 16.88 / 165.13 |

### Blind Detector Performance on Sealed Holdout

| Detector | Sample Micro P | Sample Micro R | Sample Micro F1 | Macro Defect R | Case Recall | Control FP |
|---|---:|---:|---:|---:|---:|---:|
| `v1_current` | 0.9998 | 0.9942 | 0.9970 | 0.8104 | 27/30 (90.0%) | 0/10 |
| `v2_windowed` | 0.9998 | 0.9947 | 0.9972 | 0.9790 | 30/30 (100.0%) | 0/10 |

---

## 4. Locked Confirmatory Hypotheses (Holm-Bonferroni Step-Down, m = 3)

| Hypothesis | Type | n | Mean Diff (dB) | Bootstrap 95% CI | Student-t 95% CI | Worst Cluster (dB) | Sign-Flip p | Holm-Bonf p | Verdict |
|---|---|---:|---:|---|---|---|---:|---:|---|
| `H1_vs_identity_and_gain_only` | `superiority` | 30 | +6.89 | [+5.78, +7.98] | [+5.72, +8.05] | `t-e5-sealed-19` (+1.65) | 0.0001 | 0.0003 | `PASS` |
| `H2_vs_current_production` | `superiority` | 30 | +1.13 | [+0.02, +1.90] | [+0.11, +2.16] | `t-e5-sealed-15` (-11.95) | 0.0189 | 0.0189 | `PASS` |
| `H3_vs_strong_external_dsp` | `non_inferiority_with_cpu_safety_advantage` | 30 | +0.58 | [+0.27, +0.91] | [+0.24, +0.92] | `t-e5-sealed-26` (-1.70) | 0.0007 | 0.0014 | `FAIL` |

- **H3 CPU & Safety check:** `aspade_mit__v2_windowed` p95 `143.29 s/min` vs `ss_pew_mit__v2_windowed` p95 `165.13 s/min` (`cpu_faster_or_equal = True`, stereo safety violations = `1`).

---

## 5. Frozen Candidate Stereo Safety Worst-Case Envelope & Worst 5 Defect Cases

- **Worst stereo metrics across all sealed defects:** L/R corr drift `0.0035`, ILD drift `0.10 dB`, Local ILD p95 `0.35 dB`, Phase coh drift `0.0053`, IPD mean `1.92°`, Masked IPD mean `2.07°`, ITD drift `0.009 samp`, M/S drift `0.24 dB`, Pre-echo rise `0.00 dB`, Quiet-tail drift `0.00 dB`, Band-shape RMS `1.87 dB`.

| Worst Sealed Case (`aspade_mit__v2_windowed`) | Profile | Rate / Bits | Status | ΔMasked SDR (dB) | SI-SDRi (dB) | Stereo Gate |
|---|---|---|---|---:|---:|---|
| `e5s-19-rare-trans-48k-24b` | `rare_transient_clip` | 48k / 24b | `processed_candidate` | +1.65 | +1.65 | `PASS` |
| `e5s-15-asym-neg-48k-24b` | `asymmetric_sub_full_scale` | 48k / 24b | `processed_candidate` | +2.34 | +2.62 | `FAIL` |
| `e5s-18-rare-trans-44k-24b` | `rare_transient_clip` | 44.1k / 24b | `processed_candidate` | +2.63 | +2.63 | `PASS` |
| `e5s-29-late-onset-48k-24b` | `late_onset_clip` | 48k / 24b | `processed_candidate` | +2.79 | +2.79 | `PASS` |
| `e5s-11-mild-subfs-48k-16b` | `mild_sub_full_scale` | 48k / 16b | `processed_candidate` | +2.93 | +2.94 | `PASS` |

---

## 6. Blind Listening Panel & Release Decision

- **Objective Sealed Technical Gates (`H1`+`H2`+`H3` + all 8 engineering gates):** `FAIL`
- **Blind Listening Panel Status:** `pending_human_listening`
- **Overall E5 Release Verdict:** `no_promote`

Per E0 Spec Section 4 (Go/No-Go), promotion requires **simultaneous** confirmation of: (1) all hard engineering & stereo safety gates, (2) locked confirmatory CIs on the sealed holdout, (3) **blind human listening panel perceptual confirmation with zero significant harm**, and (4) CPU/RSS/coverage budgets. Until the owner completes the loudness-matched blind listening panel and scores it via `python3 research/e5_blind_panel.py score --panel-dir <panel_dir>`, the release verdict remains strictly `inconclusive_pending_blind_panel` (or `no_promote` if any objective gate failed) — never a premature `promote` claim.
