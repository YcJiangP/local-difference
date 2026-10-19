"""Command-line entry points."""

import argparse
import json
from pathlib import Path

from .config import ENCODERS, EPOCHS
from .data import make_split
from .encoding import download_encoder, extract_dataset
from .io import read_json, save_npz, write_json
from .pipeline import configure_runtime, evaluate_run, run_controls, train_run


def parser():
    root = argparse.ArgumentParser(
        prog="speech-ci", description="Frozen speech readouts for cognitive impairment research."
    )
    commands = root.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="Generate the fixed participant split from local metadata.")
    split.add_argument("--data-root", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    download = commands.add_parser("download", help="Download one public pretrained encoder.")
    download.add_argument("--encoder", choices=ENCODERS, required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--revision", help="Optional immutable Hugging Face revision.")
    extract = commands.add_parser("extract", help="Encode authorized local recordings into raw layer caches.")
    extract.add_argument("--data-root", type=Path, required=True)
    extract.add_argument("--split-file", type=Path, required=True)
    extract.add_argument("--cache-root", type=Path, required=True)
    extract.add_argument("--encoder", choices=ENCODERS, required=True)
    extract.add_argument("--model-path", type=Path, required=True)
    extract.add_argument("--split", choices=("fit", "validation", "train", "test", "all"), default="train")
    extract.add_argument("--view", choices=("original", "fit20", "eval20"), default="original")
    extract.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    extract.add_argument("--limit", type=int, help="Limit participants for an extraction check.")
    train = commands.add_parser("train", help="Fit neural readouts and paired linear heads.")
    train.add_argument("--split-file", type=Path, required=True)
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--encoder", choices=ENCODERS, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--classes", type=int, choices=(2, 3), default=2)
    train.add_argument("--epochs", type=int, default=EPOCHS, help="Paper setting: 40. Other values are development runs.")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for name, description in (
        ("evaluate", "Evaluate fixed trained artifacts on the held-out participants."),
        ("controls", "Run matched moment, shuffle, and consistency controls."),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--split-file", type=Path, required=True)
        command.add_argument("--cache-root", type=Path, required=True)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if name == "evaluate":
            command.add_argument(
                "--conditions", nargs="+", choices=("original", "CTD20", "ALL20"),
                default=("original", "CTD20", "ALL20"),
            )
    smoke = commands.add_parser("smoke", help="Run binary and multiclass workflows on synthetic features.")
    smoke.add_argument("--output", type=Path, required=True)
    events = commands.add_parser("event-features", help="Compute event descriptors from local VAD intervals.")
    events.add_argument("--segments", type=Path, required=True)
    events.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    configure_runtime()
    limit = getattr(arguments, "limit", None)
    if getattr(arguments, "epochs", 1) < 1 or (limit is not None and limit < 1):
        raise SystemExit("Epochs and participant limits must be positive.")
    if arguments.command == "split":
        result = make_split(arguments.data_root, arguments.output)
        print(json.dumps({key: len(result[key]) for key in ("fit", "validation", "test")}))
    elif arguments.command == "download":
        print(download_encoder(arguments.encoder, arguments.output, arguments.revision))
    elif arguments.command == "extract":
        result = extract_dataset(
            arguments.data_root, arguments.split_file, arguments.cache_root,
            arguments.encoder, arguments.model_path, arguments.split,
            arguments.view, arguments.device, arguments.limit,
        )
        print(json.dumps(result))
    elif arguments.command == "train":
        result = train_run(
            arguments.split_file, arguments.cache_root, arguments.encoder,
            arguments.output, arguments.classes, arguments.epochs, arguments.device,
        )
        print(json.dumps({"completed": result["completed"], "epochs": result["epochs"]}))
    elif arguments.command == "evaluate":
        result = evaluate_run(
            arguments.split_file, arguments.cache_root, arguments.run,
            arguments.output, arguments.conditions, arguments.device,
        )
        print(json.dumps({key: value["metrics"]["Full"] for key, value in result["conditions"].items()}, indent=2))
    elif arguments.command == "controls":
        run_controls(
            arguments.split_file, arguments.cache_root, arguments.run,
            arguments.output, arguments.device,
        )
        print(f"Control results saved to {arguments.output}")
    elif arguments.command == "smoke":
        from .smoke import run_smoke
        print(json.dumps(run_smoke(arguments.output), indent=2))
    else:
        import numpy as np
        from .config import TASKS
        from .events import extract
        records = read_json(arguments.segments)
        values = [extract(records[task]["segments"], records[task]["duration_seconds"]) for task in TASKS]
        save_npz(
            arguments.output,
            summary=np.concatenate([value[0] for value in values]),
            temporal=np.concatenate([value[1] for value in values]),
        )
        print("Saved 63 summary features and 54 temporal features.")
