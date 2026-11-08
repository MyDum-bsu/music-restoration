# Defect-specific hard-clipping repair

## Choose the right problem

Hard clipping creates repeated amplitude plateaus. Soft limiting, intentional distortion, analog saturation and codec loss are different problems. A high DR does not rule out clipping. A low DR alone does not establish a defect. The detector provides evidence of a repeated extreme-level pattern, not proof that the artist wanted it removed.

Auto detection handles separate positive/negative PCM levels and up to three adjacent bins with excess histogram mass (dither/quantization). Repeated runs support a level; isolated samples at the same supported level join the mask. Clean isolated maxima stay unsupported. This can miss time-varying levels, larger dither/noise clouds, asymmetric processing away from extrema and very sparse clips. An explicit threshold override changes the mask; it does not establish correct source ancestry or unwanted clipping.

## Repair engines

`periodic_float_ar` fits floating-point predictors from reliable context and checks held-aside context rows plus forward/backward agreement. It is selected only with ≥90% confident mask coverage and ≥80% changed mask samples. Its confidence thresholds are engineering heuristics, not calibrated probabilities. Unsupported gaps remain unchanged; failure to reach coverage selects the sparse engine.

`parseval_stft_hard_projection` alternates a clipping-consistent sample projection and structured hard thresholding in a Parseval STFT. It fixes unmasked samples, permits reconstructed peaks above the observed clip level, and works on overlapping bounded-duration blocks for long tracks. Defaults: 4096-sample window at 44.1 kHz (scaled approximately at other rates), 160 iterations maximum, 12-second blocks with half overlap. This is an independent analysis-sparsity implementation, not the published A-SPADE code or exact configuration. `iteration_budget` means the iteration limit was reached, not convergence. Nonfinite/unrepresentable predictions are rejected.

After repair a single attenuation factor, shared by the channels, provides −1 dBFS true-peak headroom when needed. A proposal requiring more than 24 dB of added attenuation falls back to the source. This is a delivery guard, not a reconstruction confidence estimate. The report records pre-gain peak and output gain. A processed FLAC is strictly decoded; sample count/rate/channels, headroom and agreement with the accepted float waveform are checked. Integer dither/rounding may change samples by up to the stated 24-bit encoding tolerance. Sample consistency is checked before common output gain, not by comparing the scaled result directly to the source.

Use one sequential track at a time. Sparse reconstruction is much slower than the old contour pass and consumes FFT workspace per block. Native mono and stereo delivery preserve the channel count; multichannel conversion remains rejected by restore_safe. FFT workspace is bounded by the block, but decoded input/output arrays still grow with the track duration.

## Evidence and limits

The study uses the [authors' public dataset and precomputed references](https://rajmic.github.io/declipping2020/) from [Záviška et al., declipping survey](https://arxiv.org/abs/2007.07663). Three instrument recordings selected the window size; parameters were then frozen before evaluating seven other instruments. At input SDR 15 dB the sparse prototype improved scale-aligned SDR on all seven: median +17.36 dB, minimum +6.38 dB. These are waveform reconstruction results on artificially hard-clipped mono audio, not listening preferences or a victory over the project's named human restorers.

On those cases published precomputed A-SPADE leads our sparse prototype by a median paired 1.10 dB and SS-PEW by 3.17 dB. Their programs were not executed here; the comparison uses the provided audio, identical frame lengths and independently computed scale-aligned error. The external code is GPL-3.0/MATLAB and was not copied into this project's implementation.

Integration reuses that corpus and is not a second independent holdout. The expanded corpus checks input SDR 7/15/20 dB and ten clean controls. Two 20-second stereo excerpts from a local rock/electronic collection also improve reconstruction of added hard clipping; their pre-clipping waveforms are local references, not studio ground truth. Tests cover PCM quantization/dither, periodic tones, native mono/stereo encoding and block seams. Other defects, production sample rates and listening remain separate validation work. The early single-engine sparse prototype slightly regressed on a dithered sine; the confidence-gated AR branch addresses that measured failure. Keep the failure and subsequent result in the experiment record rather than claiming all sparse reconstruction is harmless.

For reproducible research use `research/bench_declipping.py --manifest <manifest.json> --out <empty-directory>`. Every exact source/reference path needs its SHA-256; `expected_input_sdr_db` rejects mismatched degradation levels before evaluation. Results record implementation hashes, failures, native format, scalar gain alignment and elapsed time. Do not choose reference editions or severity with an ambiguous directory glob.

The historical integer AR routine clamps to int16 and cannot reconstruct true peaks beyond full scale. It worsened error on the three calibration recordings even after oracle threshold normalization. It remains available for legacy reproduction, not as the default repair stage in new dynamics candidates.
