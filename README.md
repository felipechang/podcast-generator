# Podcast Generator

A small FastAPI service that turns source material into a two-host podcast: an [Ollama](https://ollama.com/) model
writes tagged dialogue, and [Chatterbox](https://github.com/resemble-ai/chatterbox) (multilingual TTS) synthesizes each
line with per-speaker voice cloning from reference WAV files, then concatenates everything into one WAV response.

## Requirements

- **Python** 3.11+
- **Ollama** running locally (or reachable at `OLLAMA_BASE_URL`) with your chosen model pulled (default in config is
  `glm-4.7-flash:latest`)
- **Reference audio**: two WAV files (paths set via environment variables) used as voice prompts for cloning
- **TTS stack**: install the optional `tts` extra so PyTorch and `chatterbox-tts` are available; a CUDA, MPS (Apple), or
  CPU device is selected automatically at startup. For Apple Silicon acceleration in Docker, ensure your Docker Desktop settings allow GPU passthrough.

## Installation

```bash
# Core API dependencies
pip install -e .

# LLM + TTS (PyTorch, chatterbox-tts, etc.)
pip install -e ".[tts]"

# Optional: linters
pip install -e ".[dev]"
```

Create a `.env` file in the project root or set variables in your environment. The app loads `.env` from the working
directory when present (see `podcast_generator/config.py`).

## Configuration

| Variable                              | Description                                                                        |
|---------------------------------------|------------------------------------------------------------------------------------|
| `OLLAMA_BASE_URL`                     | Ollama API base (default `http://127.0.0.1:11434`)                                 |
| `OLLAMA_MODEL`                        | Model name                                                                         |
| `OLLAMA_TIMEOUT_S`                    | HTTP timeout for Ollama (default `600`)                                            |
| `SPEAKER_1_NAME` / `SPEAKER_2_NAME`   | Labels the LLM must use in tags (defaults `Ana`, `Carlos`)                         |
| `SPEAKER_1_VOICE` / `SPEAKER_2_VOICE` | Absolute or relative paths to existing WAV files                                   |
| `TTS_DEFAULT_LANGUAGE`                | Chatterbox language id (default `es`; aliases like `spanish` → `es` are supported) |

**Voices and host names:** Put your reference WAV files in the `voices/` directory (or another path the process can read) and set `SPEAKER_1_VOICE` / `SPEAKER_2_VOICE` to those files—for example `voices/YourHost.wav`. Update `SPEAKER_1_NAME` and `SPEAKER_2_NAME` so the LLM uses the same labels in `[Name]` dialogue tags (see [Script format](#script-format) below). In Docker, mounted files under `/voices/` work the same way; see `.env.example` for path examples.

Chatterbox-specific environment variables (see `podcast_generator/chatterbox.py`):

| Variable              | Description                                                          |
|-----------------------|----------------------------------------------------------------------|
| `TTS_MODEL_ID`        | Hugging Face model id (default `ResembleAI/chatterbox-multilingual`) |
| `TTS_NUM_THREADS`     | CPU thread count; `0` means use CPU count                            |
| `TTS_INTEROP_THREADS` | PyTorch interop threads (default `1`)                                |
| `TTS_WARMUP`          | `true`/`false` — run a short warmup after load                       |

## Running the server

```bash
python -m podcast_generator
```

By default, this binds to `0.0.0.0:8000` (see `podcast_generator/__main__.py`). You can also run Uvicorn directly:

```bash
uvicorn podcast_generator.main:app --host 0.0.0.0 --port 8000
```

On startup the app loads Chatterbox in a background thread. If that fails, the API stays up but TTS routes return errors
until the model loads.

## API

### `GET /health`

Returns whether Ollama is reachable, whether Chatterbox loaded, whether both voice paths are set, plus `ollama_model`
and `device`.

#### Example request

### Health check
GET http://127.0.0.1:8000/health

### `POST /podcast/generate`

**Body (JSON):** `{ "content": "…your source material…", "assistant_prompt": "…optional…" }` — `assistant_prompt`
defaults to empty and is sent as the assistant turn before your source in the Ollama chat.

**Response:** `{"task_id": "…"}` — returns a unique task ID immediately.

Flow: LLM script → parse `[SpeakerName]` segments → TTS each segment with the matching reference WAV → concatenate WAVs. Generation happens in the background to avoid HTTP timeouts.

#### Example request

### Generate podcast
POST http://127.0.0.1:8000/podcast/generate
Content-Type: application/json

{
  "content": "Brief notes about quantum computing for a general audience.",
  "assistant_prompt": "Make it sound like a friendly conversation."
}

### `GET /podcast/task/{task_id}/status`

Returns the current status of the background task.

**Response:** `{"task_id": "…", "status": "…", "error": null}`

Possible statuses: `pending`, `generating_script`, `parsing_script`, `synthesizing_audio`, `merging_audio`, `completed`, `failed`.

#### Example request

### Check task status
GET http://127.0.0.1:8000/podcast/task/{{task_id}}/status

### `GET /podcast/task/{task_id}`

Retrieves the generated podcast audio if completed, or current status if not.

**Response:** `audio/wav` if completed; otherwise returns a JSON with the current status (same as `/status` endpoint).

#### Example request

### Download podcast
GET http://127.0.0.1:8000/podcast/task/{{task_id}}

### `POST /podcast/preview-script`

Same JSON body (including optional `assistant_prompt`). Runs only the LLM and returns:

```json
{
  "transcript": "...",
  "segment_count": 12
}
```

Useful for checking prompts and model output without running TTS.

#### Example request

### Preview podcast script
POST http://127.0.0.1:8000/podcast/preview-script
Content-Type: application/json

{
  "content": "Brief notes about quantum computing for a general audience.",
  "assistant_prompt": "Make it sound like a friendly conversation."
}

### Dynamic variables example
POST http://127.0.0.1:8000/podcast/preview-script
Content-Type: application/json

{
  "content": "Brief notes about quantum computing for a general audience.",
  "assistant_prompt": "The episode ID is {{$random.uuid}} and it was generated at {{$timestamp}}."
}

### Example

```bash
curl -sS -X POST http://127.0.0.1:8000/podcast/preview-script \
  -H "Content-Type: application/json" \
  -d '{"content":"Brief notes about quantum computing for a general audience."}'

curl -sS -X POST http://127.0.0.1:8000/podcast/generate \
  -H "Content-Type: application/json" \
  -d '{"content":"Same content as above."}'

# Then use the task_id from the response:
curl -sS http://127.0.0.1:8000/podcast/task/<task_id> -o episode.wav
```

## Script format

The LLM is instructed to produce dialogue where **every utterance starts with** a tag using your configured names, for
example `[Ana]` and `[Carlos]`, on its own token before the spoken text. The parser splits on these tags;
mismatched or missing tags yield HTTP 422 with a clear message.

## Troubleshooting

### `OSError: libgomp.so.1 cannot open shared object file: No such file or directory`

This is a common issue with `resemble-perth` (a dependency of `chatterbox-tts`) which requires OpenMP.

- **On Docker/Linux**: If you're building the container yourself, ensure `libgomp1` is installed. Our `Dockerfile` includes this by default.
- **On macOS (native installation)**: You may need to install the OpenMP library using Homebrew:
  ```bash
  brew install libomp
  ```
  And then point the environment variable `DYLD_LIBRARY_PATH` or similar if the library is not found automatically. Usually, just installing `libomp` is enough for Python libraries that use it.