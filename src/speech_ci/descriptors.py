"""Duration-aware descriptors and fitting-only neural normalization."""

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .config import TASKS
from .io import load_npz, save_npz


def layer_mean(sequence, encoder):
    values = sequence["features"].mean(axis=1, dtype=np.float64)
    if encoder == "wavlm":
        values = values.astype(np.float32).astype(np.float64)
    return values


def valid_pairs(sequence):
    starts = sequence["starts"]
    context = np.floor((starts + 1e-7) / 10).astype(int)
    present = sequence["token_counts"] > 0
    keep = (
        present[:-1] & present[1:]
        & (context[:-1] == context[1:])
        & np.isclose(np.diff(starts), 2, atol=1e-7, rtol=0)
    )
    pairs = np.flatnonzero(keep)
    if not len(pairs):
        raise ValueError("Each task recording must contain a valid within-context adjacent pair.")
    return pairs


def describe(sequence, encoder):
    values = layer_mean(sequence, encoder)
    duration = sequence["ends"] - sequence["starts"]
    pairs = valid_pairs(sequence)
    delta = np.sqrt(np.average(
        (values[pairs + 1] - values[pairs]) ** 2, axis=0,
        weights=np.minimum(duration[pairs], duration[pairs + 1]),
    ))
    mean = np.average(values, axis=0, weights=duration)
    genuine = sequence["token_counts"] > 0
    valid_mean = np.average(values[genuine], axis=0, weights=duration[genuine])
    std = np.sqrt(np.average(
        (values[genuine] - valid_mean) ** 2, axis=0, weights=duration[genuine]
    ))
    return {"delta": delta, "mean": mean, "std": std}


def participant_descriptors(sequences, encoder):
    rows = [describe(sequences[task], encoder) for task in TASKS]
    mean = np.concatenate([row["mean"] for row in rows])
    std = np.concatenate([row["std"] for row in rows])
    return {
        "delta": np.concatenate([row["delta"] for row in rows]),
        "static": rows[0]["mean"],
        "mean": mean,
        "mean_std": np.concatenate([mean, std]),
    }


def shuffle_bins(sequence, pid, task, seed):
    """Permute eligible full bins, preserving partial bins, gaps, and metadata."""
    out = {key: value.copy() for key, value in sequence.items()}
    context = np.floor((sequence["starts"] + 1e-7) / 10).astype(int)
    duration = sequence["ends"] - sequence["starts"]
    for group in np.unique(context):
        indices = np.flatnonzero(
            (context == group) & np.isclose(duration, 2, atol=1e-7, rtol=0)
            & (sequence["token_counts"] > 0)
        )
        key = f"temporal_controls_20261008:{seed}:{pid}:{task}:{int(group)}"
        random_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        permutation = np.random.default_rng(random_seed).permutation(indices)
        out["features"][indices] = sequence["features"][permutation]
    return out


@dataclass
class NeuralNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, store, people):
        first, second = [], []
        for person in people:
            for task in TASKS:
                values = store.get(person["id"], task)["features"].astype(np.float64)
                first.append(values.mean(axis=0))
                second.append((values ** 2).mean(axis=0))
        mean = np.mean(first, axis=0)
        scale = np.sqrt(np.maximum(np.mean(second, axis=0) - mean ** 2, 1e-6))
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, sequence):
        values = torch.from_numpy((sequence["features"] - self.mean) / self.scale)
        result = (values * torch.full((12, 1), 1 / 12)).sum(dim=1)
        return result.numpy().astype(np.float32)

    def save(self, path):
        save_npz(path, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path):
        arrays = load_npz(path)
        return cls(arrays["mean"], arrays["scale"])
