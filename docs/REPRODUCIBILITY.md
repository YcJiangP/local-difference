# Reproduction scope

## Included workflows

| Experiment | Release entry point | Scope |
|---|---|---|
| Binary original/CTD20/ALL20 results | split, extract, train, evaluate | Main fixed 256/64/80 participant partition |
| Branch removal | evaluate | All fixed-component configurations |
| Mean and mean-plus-SD controls | controls | Fixed-C controls on the main participant partition |
| Within-context permutation | controls | All five independently fitted descriptor controls |
| Consistency and drift | controls | beta 0 versus 1 on identical paired fitting data |
| Three-class classification | train --classes 3; evaluate | HC/MCI/dementia, centered-score fusion |
| Event descriptors and classifier | event-features; Python event modules | Exploratory extension components |

The public source package is independent of the original workstation layout. It requires authorized local audio and public pretrained encoder files; it does not require archived development folders or published participant-level prediction files.

## Fixed split versus nested development

The main pipeline implements the manuscript's fixed partition: 256 fitting, 64 validation, and 80 test participants.

The event table uses a different protocol: TRAIN320 only, two repetitions of five outer folds, and three inner folds. Its scores are averages of within-fold metrics. The fixed-split controls command must not be presented as reproducing that table.

This release includes the exact event feature definitions and block-normalized linear solver, but does not include a one-command replay of the archived nested event study. That study also depended on separately generated, fold-specific neural selection curves and predictions. Clinical feature caches, fold-specific predictions, and trained weights are not distributed. See [the event protocol](EVENT_EXTENSION.md) for the full procedure and its additional requirements.

## Selection and historical exposure

The main fusion coefficients and consistency weight were inherited from prior TRAIN-only HC-versus-MCI development; the earlier encoder set included HuBERT. The current Whisper experiment reuses the fixed settings.

The cohort and historical test results were examined during development. Fixed-model replay or refactoring validation does not create a new independent validation cohort. Conditional paired-bootstrap intervals do not include the full uncertainty of retraining or repeated research selection.

## Numerical portability

The validated environment uses Python 3.10.19, PyTorch 2.9.1+cu126, NumPy 2.2.6, SciPy 1.13.1, scikit-learn 1.6.1, and Transformers 4.35.2. CPU installation is supported for the smoke test and readout computations. Frozen feature extraction and training can show device-dependent numerical differences.

Encoder downloads support an immutable upstream revision. Extraction records the actual encoder-file hash. The source release preserves normalization order, seed-specific probability averaging, reference noise/permutation namespaces, and the additive neural-head parameterization.

New head checkpoints use NPZ arrays rather than pickled estimators. Neural checkpoints store state dictionaries and are loaded with weights_only=True.

## Evaluation outputs

Metrics are computed at the participant level. Binary AUC uses the positive CI probability; threshold-dependent metrics use 0.5. Three-class AUC is macro one-versus-rest AUC.

The controls command retains all moment controls, all shuffle seeds, and both consistency settings, including adverse results. Five-shuffle SD describes permutation variability and is not a participant-level confidence interval. The optional paired-bootstrap function in metrics.py is a conditional binary-class-stratified utility for already aligned predictions.

## Release validation

Packaging validation is recorded in [VALIDATION.md](VALIDATION.md). Unit tests and synthetic smoke runs do not replace full data-based experimental replication. The release validation distinguishes exact numerical comparisons from end-to-end execution checks.
