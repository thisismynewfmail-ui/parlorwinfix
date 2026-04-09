# Parlor

On-device, real-time multimodal AI. Have natural voice and vision conversations with an AI that runs entirely on your machine.

Parlor uses [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) for understanding speech and vision, and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) for text-to-speech. You talk, show your camera, and it talks back, all locally.

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **Research preview.** This is an early experiment. Expect rough edges and bugs.

# Why?

I'm [self-hosting a totally free voice AI](https://www.fikrikarim.com/bule-ai-initial-release/) on my home server to help people learn speaking English. It has hundreds of monthly active users, and I've been thinking about how to keep it free while making it sustainable.

The obvious answer: run everything on-device, eliminating any server cost. Six months ago I needed an RTX 5090 to run just the voice models in real-time.

Google just released a super capable small model that I can run on my M3 Pro in real-time, with vision too! Sure you can't do agentic coding with this, but it is a game-changer for people learning a new language. Imagine a few years from now that people can run this locally on their phones. They can point their camera at objects and talk about them. And this model is multi-lingual, so people can always fallback to their native language if they want. This is essentially what OpenAI demoed a few years ago.

## How it works

```
Browser (mic + camera)
    │
    │  WebSocket (audio PCM + JPEG frames)
    ▼
FastAPI server
    ├── LLM: Gemma 4 E2B via LiteRT-LM (Mac / Linux, GPU)
    │        Gemma 3 4B via Transformers + faster-whisper (Windows)
    └── TTS: Kokoro via MLX (Mac) or ONNX (Linux / Windows)
    │
    │  WebSocket (streamed audio chunks)
    ▼
Browser (playback + transcript)
```

- **Voice Activity Detection** in the browser ([Silero VAD](https://github.com/ricky0123/vad)). Hands-free, no push-to-talk.
- **Barge-in.** Interrupt the AI mid-sentence by speaking.
- **Sentence-level TTS streaming.** Audio starts playing before the full response is generated.

## Requirements

- Python 3.12+
- One of:
  - macOS with Apple Silicon
  - Linux with a supported GPU
  - Windows 10/11 x64 (CUDA GPU strongly recommended)
- ~3 GB free RAM for the model (more on Windows, see below)

## Quick start

```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
# or on Windows (PowerShell):
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

cd src
uv sync
uv run server.py
```

Open [http://localhost:8000](http://localhost:8000), grant camera and microphone access, and start talking.

Models are downloaded automatically on first run (~2.6 GB for Gemma 4 E2B on Mac/Linux, ~8 GB for Gemma 3 4B on Windows, plus TTS models).

## Windows notes

`litert-lm` (the Google AI Edge runtime used on Mac / Linux) does not ship
Windows wheels, so on Windows Parlor falls back to a HuggingFace Transformers
pipeline:

- **LLM**: `google/gemma-3-4b-it` via `transformers` (text + vision)
- **Speech-to-text**: `faster-whisper` (`base.en` by default)
- **TTS**: `kokoro-onnx` (same as Linux)

Extra setup on Windows:

1. **Accept the Gemma license** on HuggingFace for
   [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)
   and run `huggingface-cli login` (or set `HF_TOKEN`) so the weights can be
   downloaded.
2. **GPU strongly recommended.** A CUDA-capable GPU + a CUDA build of PyTorch
   makes the difference between "slightly slow" and "unusable". Install the
   CUDA wheel from <https://pytorch.org/get-started/locally/> after `uv sync`
   if you want GPU acceleration:

   ```powershell
   uv pip install --index-url https://download.pytorch.org/whl/cu124 torch
   ```
3. Everything else (model download, server startup, browser UI) is identical
   to Mac / Linux.

## Configuration

| Variable        | Default                        | Description                                                              |
| --------------- | ------------------------------ | ------------------------------------------------------------------------ |
| `MODEL_PATH`    | auto-download from HuggingFace | Mac / Linux: path to a local `gemma-4-E2B-it.litertlm` file              |
| `HF_MODEL_ID`   | `google/gemma-3-4b-it`         | Windows only: HuggingFace repo ID for the Transformers LLM backend       |
| `WHISPER_MODEL` | `base.en`                      | Windows only: `faster-whisper` model size (`tiny.en`, `base.en`, `small.en`, …) |
| `PORT`          | `8000`                         | Server port                                                              |

## Performance (Apple M3 Pro)

| Stage                            | Time          |
| -------------------------------- | ------------- |
| Speech + vision understanding    | ~1.8-2.2s     |
| Response generation (~25 tokens) | ~0.3s         |
| Text-to-speech (1-3 sentences)   | ~0.3-0.7s     |
| **Total end-to-end**             | **~2.5-3.0s** |

Decode speed: ~83 tokens/sec on GPU (Apple M3 Pro).

## Project structure

```
src/
├── server.py              # FastAPI WebSocket server (LLM + TTS glue)
├── llm.py                 # Platform-aware LLM (LiteRT on Mac/Linux, Transformers on Windows)
├── tts.py                 # Platform-aware TTS (MLX on Mac, ONNX on Linux/Windows)
├── index.html             # Frontend UI (VAD, camera, audio playback)
├── pyproject.toml         # Dependencies
└── benchmarks/
    ├── bench.py           # End-to-end WebSocket benchmark
    └── benchmark_tts.py   # TTS backend comparison
```

## Acknowledgments

- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) by Google AI Edge
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) for browser voice activity detection

## License

[Apache 2.0](LICENSE)
