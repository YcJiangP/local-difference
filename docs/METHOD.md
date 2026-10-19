# Implementation details

## Frozen representations

Each experiment uses one of WavLM Base, multilingual Whisper Small, or wav2vec 2.0 Base. Encoders run in evaluation mode without gradients.

- Audio: 16-kHz mono PCM.
- Independent encoding contexts: 10 seconds, with a final partial context retained.
- Pooling bins: 2 seconds, with a final partial bin retained.
- Hidden features: 12 Transformer-layer outputs, 768 coordinates per layer.
- Whisper: 80-bin log-Mel features, 25-ms window, 10-ms hop, 30-second right padding per context. Padded encoder frames are excluded from pooling.
- Waveform normalization follows the checkpoint processor: disabled for WavLM and enabled for wav2vec 2.0.

Frame times are 0.0125 + 0.02 f seconds for the waveform encoders and 0.02 f seconds for Whisper. Bins with no assigned frame use the nearest frame within the same context; their original frame count stays zero.

## Neural readout

Normalization is fitted on original fitting recordings only. First and second moments are averaged over bins within each recording, then equally over participants and tasks. The per-coordinate standard deviation is floored at 1e-3. Standardization occurs **before** uniform averaging over layers.

The temporal readout uses projection, GELU, a depthwise convolution of width 3, residual linear mixing, layer normalization, and dropout. It processes the full bin sequence of each task recording, including transitions between encoder contexts.

Duration-weighted pooling produces a task representation. Three task-specific logits receive equal weight in the participant logit. The training objective is participant BCE plus 0.5 times mean task BCE. Three seed-specific probabilities are averaged before conversion to log-odds.

The binary reference implementation parameterizes a task's mean classifier with two additive weight vectors. The release retains both parameter tensors and their initialization order. Although they define a single linear map at inference, merging the parameters before training would change the optimizer trajectory.

| Encoder | Hidden width | Dropout | Weight decay |
|---|---:|---:|---:|
| WavLM Base | 32 | 0.2 | 0.05 |
| Whisper Small | 64 | 0.2 | 0.01 |
| wav2vec 2.0 Base | 32 | 0.2 | 0.05 |

AdamW uses learning rate 1e-3, batch size 16, gradient clipping norm 1, and 40 epochs. Seeds are 42, 123, and 2026. Maximum original-validation AUC selects the checkpoint; the earlier epoch wins a tie. There is no refitting on validation participants.

## Local differences and static statistics

Delta and S use the raw layer mean, followed by descriptor-level standardization. WavLM's layer mean is rounded through float32 before descriptor computation to preserve the reference cache convention.

Delta uses adjacent bins only when both contain genuine encoder frames and belong to the same context. Pair weights are the shorter bin duration. Each task contributes 768 RMS coordinates; CTD/PFT/SFT concatenation gives 2304 coordinates.

S is the 768-dimensional duration-weighted CTD mean. The statistical controls concatenate the three task means (2304 coordinates), or all task means followed by all task standard deviations (4608 coordinates). Mean pooling retains fallback-bin values. Standard-deviation controls use genuine-frame bins only, matching the archived implementation.

## Paired linear heads

For standardized original/noisy features x and x', each binary head minimizes

$$
\frac{1}{2n}\sum_i a_i[\operatorname{BCE}(y_i,z_i)+\operatorname{BCE}(y_i,z'_i)]
+\frac{\|w\|^2}{2Cn}
+\frac{\beta}{2n}\sum_i a_i(z'_i-z_i)^2.
$$

Here a_i = n / (2 n_yi), C = 1e-3, and beta = 1. The intercept is unpenalized. Standardization is fitted to the original fitting view and shared by both views. L-BFGS-B uses at most 1500 iterations, ftol 1e-13, gtol 1e-8, and maxls 40.

CTD-only noise is added before frozen feature extraction. N uses original recordings; only Delta and S receive paired training. Both heads see the same perturbed CTD recording.

## Fusion and three classes

Binary fusion uses Delta as the reference logit, with neural and static corrections bounded by 2 and 1. Neural probabilities are clipped to [1e-6, 1 - 1e-6] before the logit transform.

For K = 3, neural ensemble probabilities are represented as

$$u_N = \frac{K}{K-1}\operatorname{center}(\log p_N).$$

The linear heads produce centered class scores u_Delta and u_S. Consistency and L2 penalties are averaged across classes. The fused probabilities are

$$
p=\operatorname{softmax}\left(\frac{K-1}{K}
[u_\Delta+2\tanh(u_N/4)+\tanh(u_S)]\right).
$$

The logarithm clips neural class probabilities at 1e-12. Class weights are n / (K n_class), and decisions use argmax.

## Controls

The branch-removal outputs reuse the trained components and fixed fusion terms. Statistical controls use the same encoder, three tasks, fitting split, CTD-only augmentation, beta, and descriptor-head C.

Shuffle controls permute full-duration genuine bins within each context while retaining partial/fallback bins and context membership. Each original/noisy pair shares a permutation. The five seeds are 11, 29, 47, 71, and 101; the label-independent hash namespace preserves the reference permutations.

Consistency controls refit Delta and S at beta 0 and 1, holding data, standardization, and L2 settings fixed. Normalized drift divides test RMS logit change by the standard deviation of original fitting logits, using divisor n_fit. Probability drift compares each model with its own original-recording predictions.
