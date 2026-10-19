"""Frozen speech encoding with context-local bin assignment."""

from pathlib import Path

import numpy as np
import torch

from .config import BIN_SAMPLES, CONTEXT_SAMPLES, ENCODERS, SAMPLE_RATE, TASKS
from .audio import add_noise, load_waveform
from .data import FeatureStore, partition, validate_sequence
from .io import sha256, save_npz, write_json


def download_encoder(encoder, destination, revision=None):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        ENCODERS[encoder], revision=revision, local_dir=str(destination),
        allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"],
    )


def load_encoder(encoder, model_path, device="cpu"):
    from transformers import (
        AutoConfig, WavLMModel, Wav2Vec2Model, Wav2Vec2FeatureExtractor, WhisperFeatureExtractor,
    )
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    folder = Path(model_path)
    config = AutoConfig.from_pretrained(str(folder), local_files_only=True)
    bin_path = folder / "pytorch_model.bin"
    if bin_path.is_file():
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
    elif (folder / "model.safetensors").is_file():
        from safetensors.torch import load_file
        state = load_file(str(folder / "model.safetensors"))
    else:
        raise FileNotFoundError("Model directory needs pytorch_model.bin or model.safetensors.")
    if encoder == "whisper_small_multi":
        model = WhisperEncoder(config)
        state = {key[len("model.encoder."):]: value for key, value in state.items() if key.startswith("model.encoder.")}
        processor = WhisperFeatureExtractor.from_pretrained(str(folder), local_files_only=True)
        if config.d_model != 768 or config.encoder_layers != 12 or processor.n_fft != 400 or processor.hop_length != 160:
            raise ValueError("Unexpected Whisper encoder or feature-extractor configuration.")
    else:
        if config.hidden_size != 768 or config.num_hidden_layers != 12:
            raise ValueError("Expected a 12-layer, 768-dimensional base encoder.")
        model = (WavLMModel if encoder == "wavlm" else Wav2Vec2Model)(config)
        if encoder == "wav2vec2":
            state = {key[len("wav2vec2."):]: value for key, value in state.items() if key.startswith("wav2vec2.")}
        expected = set(model.state_dict())
        converted = {}
        for key, value in state.items():
            target = key
            for old, new in ((".weight_g", ".parametrizations.weight.original0"), (".weight_v", ".parametrizations.weight.original1")):
                if key.endswith(old) and key not in expected and key[:-len(old)] + new in expected:
                    target = key[:-len(old)] + new
                elif key.endswith(new) and key not in expected and key[:-len(new)] + old in expected:
                    target = key[:-len(new)] + old
            converted[target] = value
        state = converted
        processor = Wav2Vec2FeatureExtractor.from_pretrained(str(folder), local_files_only=True)
        if processor.do_normalize != (encoder != "wavlm"):
            raise ValueError("Waveform normalization differs from the manuscript configuration.")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, processor


@torch.inference_mode()
def extract_recording(audio, encoder, model, processor, device="cpu"):
    if audio.ndim != 1 or len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError("Expected a nonempty, finite, mono waveform.")
    rows = {key: [] for key in ("features", "starts", "ends", "token_counts")}
    for offset in range(0, len(audio), CONTEXT_SAMPLES):
        chunk = audio[offset:offset + CONTEXT_SAMPLES]
        if encoder == "whisper_small_multi":
            inputs = processor(
                [chunk], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                padding="max_length", max_length=480000, truncation=True, do_normalize=False,
            )
            output = model(inputs.input_features.to(device), output_hidden_states=True)
            limit = (len(chunk) + 319) // 320
            hidden = torch.stack([value[:, :limit] for value in output.hidden_states[1:]], dim=2)
            hidden = hidden[0].float().cpu().numpy()
            centers = np.arange(limit) * 0.02
        else:
            inputs = processor(
                np.pad(chunk, (0, max(0, 1600 - len(chunk)))),
                sampling_rate=SAMPLE_RATE, return_tensors="pt",
            )
            output = model(inputs.input_values.to(device), output_hidden_states=True)
            hidden = torch.stack(output.hidden_states[1:], dim=2)[0].float().cpu().numpy()
            centers = 0.0125 + np.arange(len(hidden)) * 0.02
        for start in range(0, len(chunk), BIN_SAMPLES):
            end = min(start + BIN_SAMPLES, len(chunk))
            chosen = np.flatnonzero((centers >= start / SAMPLE_RATE) & (centers < end / SAMPLE_RATE))
            count = len(chosen)
            if not count:
                chosen = [int(np.argmin(abs(centers - (start + end) / (2 * SAMPLE_RATE))))]
            if encoder == "whisper_small_multi":
                pooled = hidden[chosen].mean(0)
            else:
                pooled = np.stack([hidden[chosen, layer, :].mean(0) for layer in range(12)])
            rows["features"].append(pooled)
            rows["starts"].append((offset + start) / SAMPLE_RATE)
            rows["ends"].append((offset + end) / SAMPLE_RATE)
            rows["token_counts"].append(count)
    result = {
        key: np.asarray(values, dtype=np.float32 if key == "features" else np.int32 if key == "token_counts" else np.float64)
        for key, values in rows.items()
    }
    validate_sequence(result)
    return result


def extract_dataset(data_root, split_path, cache_root, encoder, model_path, split="train", view="original", device="cpu", limit=None):
    if view == "fit20" and split != "fit":
        raise ValueError("Use --split fit for the manuscript fitting-noise view.")
    people = partition(split_path, split)
    if limit is not None:
        people = people[:limit]
    store = FeatureStore(cache_root, encoder)
    model, processor = load_encoder(encoder, model_path, device)
    model_file = Path(model_path) / "pytorch_model.bin"
    if not model_file.exists():
        model_file = Path(model_path) / "model.safetensors"
    model_hash = sha256(model_file)
    records = []
    for person in people:
        pid = person["id"]
        for task in (("CTD",) if view == "fit20" else TASKS):
            target = store.path(pid, task, view)
            source = Path(data_root) / pid / f"{pid}__{task}.wav"
            audio = add_noise(load_waveform(source), pid, task, view)
            sequence = extract_recording(audio, encoder, model, processor, device)
            save_npz(target, **sequence)
            metadata = {
                "schema_version": 1, "encoder": encoder, "participant": pid, "task": task,
                "view": view, "model_sha256": model_hash, "audio_sha256": sha256(source),
                "feature_sha256": sha256(target), "shape": list(sequence["features"].shape),
            }
            write_json(target.with_suffix(".json"), metadata)
            records.append({"participant": pid, "task": task, "view": view})
            print(f"Encoded {encoder}: {pid}/{task} ({view})", flush=True)
    return {"recordings": len(records), "encoder": encoder, "view": view}
