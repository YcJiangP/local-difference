# Exploratory event extension

The event extension is separate from the original N + Delta + S framework. It adds statistical moments and speech/non-speech event information to the descriptor classifier while preserving the same N/S fusion form.

## Event definitions

Input intervals are speech segments measured in seconds within a recording. They are obtained from a frozen voice activity detector, not from transcripts. Speech bouts are not word counts, and non-speech gaps are not manually validated clinical pauses.

Each task contributes:

| Group | Coordinates | Features |
|---|---:|---|
| Summary | 21 | Log recording duration; speech fraction; bout rate; leading/trailing non-speech fractions; empty indicator; six log-duration summaries for speech and internal gaps; three gap-threshold fractions |
| Temporal | 18 | Four temporal-quartile speech fractions; four onset rates; linear/quadratic occupancy contrasts; linear/quadratic speech and gap duration progression; lag-one correlations and adjacent log-duration changes |

Across CTD/PFT/SFT, this yields 63 summary and 54 temporal coordinates. The no-temporal ablation removes only the 54 temporal coordinates.

Synthetic example:

```bash
python -m speech_ci event-features --segments examples/synthetic_events.json --output artifacts/event_features.npz
```

The Python interfaces are `speech_ci.events.extract(segments, total_seconds)` and `speech_ci.event_heads.BlockHeadPath`.

## Frozen VAD provenance

The development implementation used [Silero VAD](https://github.com/snakers4/silero-vad) at commit `1e261b036686cd0017d500ee96acd1c4ba572a9d`.

- Sampling rate: 16000 Hz.
- Speech / negative thresholds: 0.5 / 0.35.
- Minimum speech / silence durations: 100 / 100 ms.
- Speech padding: 0 ms.
- Model path upstream: `src/silero_vad/data/silero_vad.jit`.
- Model SHA-256: `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`.

Download and use upstream assets under their own license. This release does not bundle the VAD model or third-party implementation.

## Classifier and nested selection

Feature blocks are standardized on original fitting participants, divided by the square root of their block dimensionality, and concatenated. The paired classifier uses beta 1 and an unpenalized intercept.

The shared normalized C grid is:

```text
0.04608, 0.4608, 2.304, 4.608, 46.08, 460.8
```

For a single d-dimensional block, normalized C = d times the original unnormalized C. Thus 2.304 for the three-task mean and 4.608 for mean-plus-SD reproduce the original C = 0.001 regularization.

For each standalone/fusion role, the inner criterion is mean (AUC + Macro-F1) / 2. Ties within 1e-6 prefer smaller C. Each family receives the same grid. Thresholds are fixed at 0.5.

The exploratory protocol uses two repetitions of five outer folds and three inner folds, stratified by HC/MCI/dementia. Neural selection must be performed inside each outer training partition, and the same fold-specific N/S components must be reused across descriptor families. Reusing a model trained on all TRAIN320 would leak outer-fold participants.

Model families:

- Statistical controls: mean and mean-plus-SD, standalone and fused.
- Original Delta descriptor, with fixed and inner-selected regularization.
- Mean-plus-SD + Delta + event summaries.
- Mean-plus-SD + Delta + event summaries + temporal event descriptors.
- Event-order permutation controls with seeds 11, 29, and 47.

The principal encoder specified in the development protocol was WavLM; other encoders were secondary. Displaying a favorable secondary result does not change that protocol.

## What this release reproduces

The event functions and block-normalized solver are included and validated independently. The archived nested-CV orchestration and its fold-specific neural artifacts are not included as a turnkey reproduction workflow. This boundary is explicit because the main fixed-split pipeline evaluates a different experiment.
