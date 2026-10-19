"""Fixed configuration of the manuscript's main experiments."""

TASKS = ("CTD", "PFT", "SFT")
CLASSES = ("HC", "MCI", "Dementia")
ENCODERS = {
    "wavlm": "microsoft/wavlm-base",
    "whisper_small_multi": "openai/whisper-small",
    "wav2vec2": "facebook/wav2vec2-base",
}
NEURAL_CONFIGS = {
    "wavlm": {"hidden": 32, "dropout": 0.2, "weight_decay": 0.05},
    "whisper_small_multi": {"hidden": 64, "dropout": 0.2, "weight_decay": 0.01},
    "wav2vec2": {"hidden": 32, "dropout": 0.2, "weight_decay": 0.05},
}
SEEDS = (42, 123, 2026)
SAMPLE_RATE = 16000
CONTEXT_SAMPLES = 160000
BIN_SAMPLES = 32000
LAYERS = 12
DIMENSION = 768
SPLIT_SEED = 61
EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
CLASSIFIER_C = 1e-3
CONSISTENCY_BETA = 1.0
THRESHOLD = 0.5
SHUFFLE_SEEDS = (11, 29, 47, 71, 101)
