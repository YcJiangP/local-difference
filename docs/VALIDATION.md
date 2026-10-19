# Release validation

The release was checked against the reference implementation on 19 October 2026. These checks validate packaging and numerical fidelity; they are separate from the scientific evaluation of the method.

## Completed checks

| Check | Result |
|---|---|
| Dataset-free unit tests | 18 passed |
| Package build and isolated installation | Offline wheel build, isolated installation, CLI help, and installed-package tests passed |
| Binary synthetic workflow | Feature caches, three neural seeds, paired heads, serialization, and three evaluation conditions completed |
| Three-class synthetic workflow | Three neural seeds, class-balanced heads, centered-score fusion, and evaluation completed |
| Matched-control synthetic workflow | Mean/Mean+SD, both consistency settings, and all five shuffle controls completed |
| Neural initialization and gradients | Exact agreement for nine binary and three multiclass model initializations |
| Linear objectives and gradients | Exact agreement for binary and three-class objectives |
| Descriptor definitions | Exact mean, genuine-bin SD, and local-difference agreement |
| Permutation protocol | Exact agreement for all five reference bin-shuffle seeds |
| Participant partition | Exact reconstruction of the reference 256/64 fitting/validation identities |
| Binary trained checkpoints | Exact seed probabilities and fused predictions on four training participants per encoder |
| Three-class trained checkpoints | Exact seed probabilities and fused predictions on four training participants |
| Frozen feature extraction | Exact agreement on a 12.7-second synthetic waveform for all three encoders |

The reference comparisons reported a maximum absolute discrepancy of 0.0 in this environment after preserving the reference float32 multiclass ensemble averaging. The binary ensemble retains float64 averaging. The waveform check exercises a complete and a partial encoding context; unit tests cover empty-bin handling separately.

The aggregate audit is in [reference_parity.json](reference_parity.json). It contains no participant identifiers, clinical labels, audio, or participant-level predictions.

## Environment

- Python 3.10.19
- PyTorch 2.9.1+cu126; the reported numerical comparisons ran on CPU
- NumPy 2.2.6
- SciPy 1.13.1
- scikit-learn 1.6.1
- Transformers 4.35.2

## Scope

The release checks did not retrain all manuscript models for 40 epochs or re-extract every clinical recording. The short end-to-end workflows use synthetic data. Numerical agreement on the checked cases is not a guarantee of bitwise equality on every device or dependency build.

The GitHub Actions workflow is supplied for future pushes; a local pass does not imply that a hosted GitHub run has already occurred.
