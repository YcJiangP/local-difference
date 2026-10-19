"""Dataset-free end-to-end validation using synthetic features."""

from pathlib import Path

import numpy as np

from .config import TASKS
from .data import FeatureStore
from .io import save_npz, write_json
from .pipeline import evaluate_run, train_run


def run_smoke(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("The smoke-test output directory must be empty.")
    output.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(817)
    split = {"schema_version": 1, "synthetic": True, "fit": [], "validation": [], "test": []}
    counter = 0
    store = FeatureStore(output / "cache", "wavlm")
    for name, count in (("fit", 18), ("validation", 9), ("test", 9)):
        for j in range(count):
            diagnosis = j % 3
            pid = f"synthetic_{counter:03d}"
            counter += 1
            split[name].append({"id": pid, "diagnosis": diagnosis})
            for task_index, task in enumerate(TASKS):
                bins = 5 + (j + task_index) % 3
                values = generator.normal(size=(bins, 12, 768)).astype(np.float32)
                values[:, :, :8] += 0.25 * diagnosis
                base = {
                    "features": values, "starts": 2.0 * np.arange(bins),
                    "ends": 2.0 * (np.arange(bins) + 1),
                    "token_counts": np.full(bins, 100, dtype=np.int32),
                }
                save_npz(store.path(pid, task), **base)
                if name == "fit" and task == "CTD":
                    noisy = dict(base, features=values + 0.1 * generator.normal(size=values.shape).astype(np.float32))
                    save_npz(store.path(pid, task, "fit20"), **noisy)
                if name == "test":
                    noisy = dict(base, features=values + 0.1 * generator.normal(size=values.shape).astype(np.float32))
                    save_npz(store.path(pid, task, "eval20"), **noisy)
    split_path = output / "split.json"
    write_json(split_path, split)
    summary = {"synthetic_only": True, "epochs_per_seed": 2, "pipelines": {}}
    for classes in (2, 3):
        run = output / f"run_{classes}class"
        train_run(split_path, output / "cache", "wavlm", run, classes=classes, epochs=2)
        result = evaluate_run(
            split_path, output / "cache", run, output / f"evaluation_{classes}class.json"
        )
        for condition in result["conditions"].values():
            probability = np.asarray(condition["probabilities"]["Full"])
            expected = (9,) if classes == 2 else (9, 3)
            if probability.shape != expected or not np.isfinite(probability).all():
                raise AssertionError("Invalid synthetic predictions.")
        summary["pipelines"][str(classes)] = "passed"
    write_json(output / "smoke_report.json", summary)
    return summary
