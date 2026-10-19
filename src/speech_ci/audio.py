"""Waveform loading and deterministic white-noise views."""

import hashlib
import wave
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE


def load_waveform(path):
    with wave.open(str(Path(path)), "rb") as stream:
        if (
            stream.getframerate() != SAMPLE_RATE or stream.getnchannels() != 1
            or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE"
        ):
            raise ValueError("Expected uncompressed, 16-kHz, mono, 16-bit PCM WAV audio.")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def add_noise(audio, pid, task, view):
    if view == "original":
        return audio
    if view == "fit20":
        if task != "CTD":
            raise ValueError("The manuscript fitting augmentation changes CTD only.")
        namespace = f"noise_robustness_20260924:{pid}:train_white20"
    elif view == "eval20":
        namespace = (
            f"noise_robustness_20260924:{pid}:eval_white20"
            if task == "CTD" else f"all_task_noise20_20261006:{pid}:{task}"
        )
    else:
        raise ValueError("Unknown noise view.")
    # These namespaces preserve the manuscript's fixed noise realizations.
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "little")
    noise = np.random.default_rng(seed).normal(size=len(audio))
    noise -= noise.mean()
    rms = np.sqrt(np.mean(audio.astype(float) ** 2))
    if rms <= 1e-10:
        raise ValueError("SNR is undefined for a silent waveform.")
    noise *= rms / (10 * np.sqrt(np.mean(noise * noise)))
    return (audio + noise).astype(np.float32)
