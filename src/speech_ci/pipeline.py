"""Training, evaluation, and matched binary controls."""

import copy
import os
from pathlib import Path
import random

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .config import (
    BATCH_SIZE, CLASSIFIER_C, CONSISTENCY_BETA, EPOCHS, LEARNING_RATE,
    NEURAL_CONFIGS, SEEDS, SHUFFLE_SEEDS, TASKS,
)
from .data import FeatureStore, validate_split
from .descriptors import NeuralNormalizer, participant_descriptors, shuffle_bins
from .fusion import binary_fusion, multiclass_fusion, logit
from .heads import PairedHead
from .io import read_json, sha256, write_json
from .metrics import classification_metrics, normalized_logit_drift, probability_drift
from .models import BinaryReadout, MulticlassReadout, batch, class_weights, neural_loss


def configure_runtime(threads=2):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assemble(store, people, normalizer=None, condition="original", shuffle_seed=None):
    if condition not in ("original", "fit20", "CTD20", "ALL20"):
        raise ValueError("Unknown evaluation or fitting condition.")
    result = {"ids": [], "diagnosis": [], "items": [], "delta": [], "static": [], "mean": [], "mean_std": []}
    encoder = store.root.name
    for person in people:
        pid = person["id"]
        sequences = {}
        for task in TASKS:
            changed = condition == "ALL20" or (condition in ("fit20", "CTD20") and task == "CTD")
            view = ("fit20" if condition == "fit20" else "eval20") if changed else "original"
            sequences[task] = store.get(pid, task, view)
        result["ids"].append(pid)
        result["diagnosis"].append(person["diagnosis"])
        if normalizer is not None:
            result["items"].append({
                "x": [normalizer.transform(sequences[task]) for task in TASKS],
                "duration": [(sequences[task]["ends"] - sequences[task]["starts"]).astype(np.float32) for task in TASKS],
            })
        if shuffle_seed is not None:
            sequences = {task: shuffle_bins(sequences[task], pid, task, shuffle_seed) for task in TASKS}
        descriptors = participant_descriptors(sequences, encoder)
        for key, values in descriptors.items():
            result[key].append(values)
    for key in ("diagnosis", "delta", "static", "mean", "mean_std"):
        result[key] = np.asarray(result[key])
    return result


def labels(pack, classes):
    return (pack["diagnosis"] != 0).astype(int) if classes == 2 else pack["diagnosis"]


@torch.inference_mode()
def predict_neural(model, items, classes=2, device="cpu"):
    model.eval()
    outputs = []
    for start in range(0, len(items), BATCH_SIZE):
        x, duration = batch(items[start:start + BATCH_SIZE], device)
        logits = model(x, duration)["logits"]
        probability = torch.sigmoid(logits) if classes == 2 else torch.softmax(logits, dim=-1)
        outputs.append(probability.cpu().numpy())
    result = np.concatenate(outputs)
    return result.astype(float) if classes == 2 else result


def new_model(config, classes):
    model_class = BinaryReadout if classes == 2 else MulticlassReadout
    return model_class(hidden=config["hidden"], dropout=config["dropout"])


def train_run(split_path, cache_root, encoder, output, classes=2, epochs=EPOCHS, device="cpu"):
    if classes not in (2, 3) or epochs < 1:
        raise ValueError("Use two or three classes and a positive epoch count.")
    split = read_json(split_path)
    validate_split(split)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new, empty run directory; trained artifacts are never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(NEURAL_CONFIGS[encoder])
    manifest = {
        "schema_version": 1, "encoder": encoder, "classes": classes,
        "seeds": list(SEEDS), "epochs": epochs, "neural_config": config,
        "C": CLASSIFIER_C, "beta": CONSISTENCY_BETA, "threshold": 0.5,
        "split_sha256": sha256(split_path), "selection": "maximum validation AUC; earliest tie",
        "completed": False, "device": device,
    }
    write_json(output / "run.json", manifest)
    store = FeatureStore(cache_root, encoder)
    normalizer = NeuralNormalizer.fit(store, split["fit"])
    normalizer.save(output / "normalizer.npz")
    fitting = assemble(store, split["fit"], normalizer)
    validation = assemble(store, split["validation"], normalizer)
    augmented = assemble(store, split["fit"], condition="fit20")
    y_fit, y_val = labels(fitting, classes), labels(validation, classes)
    if classes == 2:
        weight = torch.tensor(float((y_fit == 0).sum() / (y_fit == 1).sum()), device=device)
    else:
        weight = torch.tensor(class_weights(y_fit), dtype=torch.float32, device=device)
    with threadpool_limits(limits=2):
        for seed in SEEDS:
            set_seed(seed)
            model = new_model(config, classes).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=config["weight_decay"]
            )
            generator = np.random.default_rng(seed)
            best_score, history = -np.inf, []
            for epoch in range(1, epochs + 1):
                model.train()
                indices = generator.permutation(len(y_fit))
                for start in range(0, len(indices), BATCH_SIZE):
                    chosen = indices[start:start + BATCH_SIZE]
                    x, duration = batch([fitting["items"][j] for j in chosen], device)
                    target = torch.tensor(
                        y_fit[chosen], dtype=torch.float32 if classes == 2 else torch.long, device=device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss = neural_loss(model(x, duration), target, weight, classes)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite neural objective.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                probabilities = predict_neural(model, validation["items"], classes, device)
                score = float(
                    roc_auc_score(y_val, probabilities) if classes == 2 else
                    roc_auc_score(y_val, probabilities, multi_class="ovr", average="macro")
                )
                if score > best_score:
                    best_score = score
                    checkpoint = {
                        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                        "config": config, "classes": classes, "seed": seed, "epoch": epoch,
                    }
                    torch.save(checkpoint, output / f"seed_{seed}.pt")
                history.append({"epoch": epoch, "validation_auc": score})
                print(f"{encoder} seed={seed} epoch={epoch}/{epochs} validation_auc={score:.4f}", flush=True)
            write_json(output / f"history_{seed}.json", history)
        for name in ("delta", "static"):
            head = PairedHead(c=CLASSIFIER_C, beta=CONSISTENCY_BETA, classes=classes).fit(
                fitting[name], augmented[name], y_fit
            )
            head.save(output / f"{name}.npz")
            write_json(output / f"{name}_diagnostics.json", head.diagnostics)
    manifest["completed"] = True
    manifest["artifacts"] = {
        path.name: sha256(path) for path in output.iterdir() if path.name != "run.json" and path.is_file()
    }
    write_json(output / "run.json", manifest)
    return manifest


def load_run(run_dir, split_path, device="cpu"):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "run.json")
    if not manifest["completed"] or sha256(split_path) != manifest["split_sha256"]:
        raise ValueError("Incomplete run or mismatched participant split.")
    for name, expected in manifest["artifacts"].items():
        if sha256(run_dir / name) != expected:
            raise ValueError(f"Changed run artifact: {name}")
    models = []
    for seed in manifest["seeds"]:
        saved = torch.load(run_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
        model = new_model(saved["config"], manifest["classes"]).to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
        models.append(model.eval())
    heads = {key: PairedHead.load(run_dir / f"{key}.npz") for key in ("delta", "static")}
    normalizer = NeuralNormalizer.load(run_dir / "normalizer.npz")
    return manifest, models, heads, normalizer


def evaluate_run(split_path, cache_root, run_dir, output, conditions=("original", "CTD20", "ALL20"), device="cpu"):
    split = read_json(split_path)
    validate_split(split)
    manifest, models, heads, normalizer = load_run(run_dir, split_path, device)
    store = FeatureStore(cache_root, manifest["encoder"])
    result = {"encoder": manifest["encoder"], "classes": manifest["classes"], "conditions": {}}
    for condition in conditions:
        packed = assemble(store, split["test"], normalizer, condition)
        neural = np.mean(
            [predict_neural(model, packed["items"], manifest["classes"], device) for model in models], axis=0
        )
        delta = heads["delta"].decision_function(packed["delta"])
        static = heads["static"].decision_function(packed["static"])
        function = binary_fusion if manifest["classes"] == 2 else multiclass_fusion
        probabilities = function(neural, delta, static)
        target = labels(packed, manifest["classes"])
        metrics = {
            name: classification_metrics(target, value, manifest["classes"])
            for name, value in probabilities.items()
        }
        for value in metrics.values():
            value["net_correct_gain_over_N"] = value["correct"] - metrics["N"]["correct"]
        result["conditions"][condition] = {
            "ids": packed["ids"], "labels": target.tolist(), "metrics": metrics,
            "probabilities": {name: value.tolist() for name, value in probabilities.items()},
        }
    write_json(output, result)
    return result


@threadpool_limits.wrap(limits=2)
def run_controls(split_path, cache_root, run_dir, output, device="cpu"):
    """Refit matched descriptor controls without selecting on evaluation outcomes."""
    split = read_json(split_path)
    manifest, models, _, normalizer = load_run(run_dir, split_path, device)
    if manifest["classes"] != 2:
        raise ValueError("The descriptor-control protocol is binary.")
    store = FeatureStore(cache_root, manifest["encoder"])
    fitting = assemble(store, split["fit"])
    augmented = assemble(store, split["fit"], condition="fit20")
    target_fit = labels(fitting, 2)
    evaluation = {
        condition: assemble(store, split["test"], normalizer, condition)
        for condition in ("original", "CTD20", "ALL20")
    }
    neural = {
        condition: np.mean([predict_neural(model, packed["items"], device=device) for model in models], axis=0)
        for condition, packed in evaluation.items()
    }
    result = {
        "encoder": manifest["encoder"], "C": CLASSIFIER_C, "threshold": 0.5,
        "statistical_controls": {}, "consistency": {}, "shuffle": {},
        "ids": evaluation["original"]["ids"],
        "labels": labels(evaluation["original"], 2).tolist(),
    }
    static = PairedHead(c=CLASSIFIER_C, beta=CONSISTENCY_BETA).fit(
        fitting["static"], augmented["static"], target_fit
    )
    for family in ("mean", "mean_std"):
        head = PairedHead(c=CLASSIFIER_C, beta=CONSISTENCY_BETA).fit(
            fitting[family], augmented[family], target_fit
        )
        result["statistical_controls"][family] = {}
        for condition, packed in evaluation.items():
            scores = head.decision_function(packed[family])
            fused = binary_fusion(neural[condition], scores, static.decision_function(packed["static"]))["Full"]
            target = labels(packed, 2)
            result["statistical_controls"][family][condition] = {
                "standalone": classification_metrics(target, expit(scores)),
                "fusion": classification_metrics(target, fused),
                "probabilities": {"standalone": expit(scores).tolist(), "fusion": fused.tolist()},
            }
    for beta in (0.0, 1.0):
        heads = {
            key: PairedHead(c=CLASSIFIER_C, beta=beta).fit(fitting[key], augmented[key], target_fit)
            for key in ("delta", "static")
        }
        predictions = {}
        result["consistency"][str(beta)] = {}
        for condition, packed in evaluation.items():
            scores = {key: head.decision_function(packed[key]) for key, head in heads.items()}
            probability = binary_fusion(neural[condition], scores["delta"], scores["static"])["Full"]
            predictions[condition] = probability
            values = classification_metrics(labels(packed, 2), probability)
            values["probability_drift"] = probability_drift(predictions["original"], probability)
            values["probabilities"] = probability.tolist()
            if condition == "CTD20":
                values["normalized_head_drift"] = {
                    key: normalized_logit_drift(
                        head.decision_function(fitting[key]),
                        head.decision_function(evaluation["original"][key]),
                        scores[key],
                    ) for key, head in heads.items()
                }
            result["consistency"][str(beta)][condition] = values
    for seed in (None, *SHUFFLE_SEEDS):
        train = fitting if seed is None else assemble(store, split["fit"], shuffle_seed=seed)
        noisy = augmented if seed is None else assemble(store, split["fit"], condition="fit20", shuffle_seed=seed)
        test = evaluation["original"] if seed is None else assemble(store, split["test"], shuffle_seed=seed)
        head = PairedHead(c=CLASSIFIER_C, beta=CONSISTENCY_BETA).fit(
            train["delta"], noisy["delta"], target_fit
        )
        probability = head.predict_proba(test["delta"])[:, 1]
        values = classification_metrics(labels(test, 2), probability)
        values["probabilities"] = probability.tolist()
        result["shuffle"]["ordered" if seed is None else str(seed)] = values
    result["shuffle_gain_summary"] = {}
    for metric in ("auc", "macro_f1"):
        gains = np.asarray([
            result["shuffle"]["ordered"][metric] - result["shuffle"][str(seed)][metric]
            for seed in SHUFFLE_SEEDS
        ])
        result["shuffle_gain_summary"][metric] = {
            "mean": float(gains.mean()), "sample_sd": float(gains.std(ddof=1)),
        }
    write_json(output, result)
    return result
