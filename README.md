# EPUB Audiobook Reader

Turn any EPUB into an audiobook with AI voice cloning. Powered by [Faster Qwen3-TTS](https://github.com/andimarafioti/faster-qwen3-tts).

## Features

- **Voice Cloning** — clone any voice from a short audio sample
- **Custom Voices** — use built-in Qwen3-TTS speaker voices
- **Real-time Streaming** — audio plays as it's generated, no waiting
- **Chapter Navigation** — browse and select chapters from the sidebar
- **EPUB Library** — books are saved locally for quick access
- **Full Audiobook Export** — generate and download a complete WAV file
- **Voice Manager** — record reference audio directly in the browser or upload files
- **Dark / Light Mode** — with system preference detection
- **Bilingual** — automatic Chinese / English language detection and chunking

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~4GB VRAM for the 0.6B model, ~8GB for the 1.7B model

## Installation

```bash
pip install epub-audiobook-reader
```

Or install from source:

```bash
git clone https://github.com/williamcotton/epub-audiobook-reader.git
cd epub-audiobook-reader
pip install -e .
```

## Quick Start

```bash
# Start with the default 1.7B model (auto-downloads from HuggingFace)
epub-audiobook-reader

# Use the smaller 0.6B model
epub-audiobook-reader --model Qwen/Qwen3-TTS-12Hz-0.6B-Base

# Start without preloading a model (select in the UI)
epub-audiobook-reader --no-preload

# Enable HTTPS for microphone recording support
epub-audiobook-reader --ssl
```

Then open http://localhost:7861 in your browser.

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Model to preload at startup |
| `--port` | `7861` | Server port |
| `--host` | `0.0.0.0` | Server host |
| `--no-preload` | off | Skip model loading at startup |
| `--ssl` | off | Enable HTTPS with auto-generated self-signed certificate |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QWEN3_TTS_ROOT` | — | Local directory containing downloaded models |
| `PORT` | `7861` | Server port (overridden by `--port`) |
| `MODEL_CACHE_SIZE` | `2` | Maximum number of models to keep loaded |
| `ACTIVE_MODELS` | all | Comma-separated list of models to show in the UI |
| `ASSET_DIR` | `/tmp/faster-qwen3-tts-assets` | Directory for downloaded reference audio assets |

## Available Models

| Model | Type | Size | Description |
|---|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Voice Clone | ~0.6B | Smaller, faster model |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice Clone | ~1.7B | Higher quality |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Custom Voice | ~0.6B | Built-in speakers, smaller |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Custom Voice | ~1.7B | Built-in speakers, higher quality |

## How It Works

1. **Upload an EPUB** — the book is parsed into chapters and text segments
2. **Select a voice** — choose a built-in preset, upload reference audio, or record your own
3. **Play** — click any chapter to start streaming audio in real-time
4. **Export** — optionally generate a full audiobook WAV file for offline listening

## License

MIT
