"""Participant-level partitions and feature-cache access."""

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .config import CLASSES, ENCODERS, SPLIT_SEED, TASKS
from .io import load_npz, read_json, safe_identifier, sha256, write_json


def make_split(data_root, output):
    """Reconstruct the fixed 256/64/80 split from authorized local metadata."""
    data_root = Path(data_root)
    metadata = data_root / "meta-info.csv"
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    required = {"IDs", "diagnosis", "Split"}
    if not records or not required.issubset(records[0]):
        raise ValueError("Metadata requires IDs, diagnosis, and Split columns.")
    people = []
    for row in records:
        pid = safe_identifier(row["IDs"])
        if row["diagnosis"] not in CLASSES:
            raise ValueError("Unknown diagnosis in metadata.")
        split = row["Split"].strip().lower()
        if split not in ("train", "test"):
            raise ValueError("Official split must be train or test.")
        people.append({"id": pid, "diagnosis": CLASSES.index(row["diagnosis"]), "split": split})
    people.sort(key=lambda row: row["id"])
    if len({row["id"] for row in people}) != len(people):
        raise ValueError("Duplicate participant identifiers.")
    train = [row for row in people if row["split"] == "train"]
    test = [row for row in people if row["split"] == "test"]
    if len(train) != 320 or len(test) != 80:
        raise ValueError("The paper protocol requires TRAIN320 and TEST80.")
    diagnosis = np.array([row["diagnosis"] for row in train])
    fit, validation = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=SPLIT_SEED, stratify=diagnosis
    )
    fit, validation = np.sort(fit), np.sort(validation)
    expected = ((fit, [128, 96, 32]), (validation, [32, 24, 8]))
    for indices, counts in expected:
        if np.bincount(diagnosis[indices], minlength=3).tolist() != counts:
            raise ValueError("Unexpected fitting or validation class counts.")
    if np.bincount([row["diagnosis"] for row in test], minlength=3).tolist() != [40, 30, 10]:
        raise ValueError("Unexpected test class counts.")
    result = {
        "schema_version": 1,
        "split_seed": SPLIT_SEED,
        "class_order": list(CLASSES),
        "metadata_sha256": sha256(metadata),
        "fit": [train[j] for j in fit],
        "validation": [train[j] for j in validation],
        "test": test,
    }
    write_json(output, result)
    return result


def partition(path, name):
    split = read_json(path)
    if name == "train":
        return sorted(split["fit"] + split["validation"], key=lambda row: row["id"])
    if name == "all":
        return sorted(split["fit"] + split["validation"] + split["test"], key=lambda row: row["id"])
    return split[name]


def validate_split(split):
    groups = [set(row["id"] for row in split[key]) for key in ("fit", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Participants must not cross fitting, validation, and test partitions.")
    for key in ("fit", "validation", "test"):
        rows = split[key]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate participant in partition.")
        if {row["diagnosis"] for row in rows} != {0, 1, 2}:
            raise ValueError("Every partition must contain HC, MCI, and dementia.")


class FeatureStore:
    """Read raw layer features; no fitted normalization is stored in the cache."""

    def __init__(self, root, encoder):
        if encoder not in ENCODERS:
            raise ValueError("Unknown encoder identifier.")
        self.root = Path(root) / encoder

    def path(self, pid, task, view="original"):
        if task not in TASKS or view not in ("original", "fit20", "eval20"):
            raise ValueError("Unknown task or feature view.")
        return self.root / view / safe_identifier(pid) / (task + ".npz")

    def get(self, pid, task, view="original"):
        path = self.path(pid, task, view)
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature cache: {path}. Run the extract command first.")
        sequence = load_npz(path)
        validate_sequence(sequence)
        return sequence


def validate_sequence(sequence):
    features = sequence["features"]
    starts, ends, counts = (sequence[key] for key in ("starts", "ends", "token_counts"))
    if features.ndim != 3 or features.shape[1:] != (12, 768) or len(features) == 0:
        raise ValueError("Features must have shape [bins, 12, 768].")
    if any(len(value) != len(features) for value in (starts, ends, counts)):
        raise ValueError("Feature metadata length mismatch.")
    if not all(np.isfinite(value).all() for value in (features, starts, ends, counts)):
        raise ValueError("Non-finite feature values.")
    if np.any(ends <= starts) or np.any(counts < 0):
        raise ValueError("Invalid bin duration or frame count.")
    if not np.isclose(starts[0], 0) or not np.allclose(starts[1:], ends[:-1]):
        raise ValueError("Bins must provide consecutive recording coverage.")
