# Data preparation

## Authorized local data

Request PROCESS-2 through the [official dataset page](https://huggingface.co/datasets/CognoSpeak/PROCESS-2) and follow the applicable data-use agreement. The repository provides algorithms and synthetic examples; it does not redistribute any portion of the clinical dataset or its derived participant features.

Expected local layout:

```text
data/PROCESS-2/
  meta-info.csv
  SUBJECT_ID/
    SUBJECT_ID__CTD.wav
    SUBJECT_ID__PFT.wav
    SUBJECT_ID__SFT.wav
```

`SUBJECT_ID` is the identifier appearing in the metadata. Required metadata columns are `IDs`, `diagnosis`, and `Split`. Diagnosis values are `HC`, `MCI`, and `Dementia`; split values are `train` and `test`.

Audio must be mono, 16-kHz, uncompressed 16-bit PCM WAV. The loader rejects mismatched formats rather than silently resampling them.

## Participant partition

| Partition | HC | MCI | Dementia | Total |
|---|---:|---:|---:|---:|
| Fitting | 128 | 96 | 32 | 256 |
| Validation | 32 | 24 | 8 | 64 |
| Test | 40 | 30 | 10 | 80 |

The fitting/validation split uses seed 61 after sorting official training participants by identifier, with stratification by the original three diagnostic categories. All recordings from one participant stay in the same partition.

For the binary task, MCI and dementia map to positive class 1; HC maps to 0. For the three-class task, the order is HC/MCI/dementia.

## Feature cache

```text
artifacts/cache/ENCODER/
  original/SUBJECT_ID/CTD.npz
  original/SUBJECT_ID/PFT.npz
  original/SUBJECT_ID/SFT.npz
  fit20/SUBJECT_ID/CTD.npz
  eval20/SUBJECT_ID/CTD.npz
  eval20/SUBJECT_ID/PFT.npz
  eval20/SUBJECT_ID/SFT.npz
```

Each recording cache contains:

| Key | Shape | Meaning |
|---|---|---|
| `features` | bins × 12 × 768 | Raw per-layer bin representations |
| `starts` / `ends` | bins | Recording-relative bin boundaries in seconds |
| `token_counts` | bins | Genuine encoder-frame count before fallback |

Companion JSON files record waveform, encoder-weight, and feature-file hashes. Feature caches are data-derived artifacts and remain local under the dataset agreement.

## Fitting and evaluation views

- Original features are computed without synthetic corruption.
- Fitting pairs use one deterministic CTD-only 20-dB white-noise view.
- Evaluation uses a separate deterministic seed namespace.
- CTD-only and all-task evaluations share exactly the same CTD perturbation.
- Noise is scaled by the original waveform RMS and added without waveform clipping.

The neural branch uses original fitting inputs. The Delta and S classifiers use paired original/noisy inputs. Fitting standardizers never use validation or test recordings.
