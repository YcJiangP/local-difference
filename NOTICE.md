# Third-party resources

The source package does not contain upstream model weights, clinical recordings, transcripts, clinical metadata, participant-level feature caches, or archived predictions.

| Resource | Upstream source |
|---|---|
| PROCESS-2 | https://huggingface.co/datasets/CognoSpeak/PROCESS-2 |
| WavLM | https://huggingface.co/microsoft/wavlm-base |
| Whisper | https://huggingface.co/openai/whisper-small |
| wav2vec 2.0 | https://huggingface.co/facebook/wav2vec2-base |
| Silero VAD | https://github.com/snakers4/silero-vad |

Use the access terms and license provided by each upstream project. This repository does not grant rights to redistribute those resources. The event module operates on supplied intervals and does not vendor Silero source code or weights.

The package was extracted and reorganized from the manuscript's research implementation. Historical search scripts, machine-specific paths, and experiment artifacts are not runtime dependencies.
