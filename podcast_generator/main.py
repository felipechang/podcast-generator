import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Dict, Any

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from podcast_generator import chatterbox
from podcast_generator.audio_join import concat_wav_bytes
from podcast_generator.config import Settings, get_settings
from podcast_generator.llm import generate_podcast_script
from podcast_generator.parse_script import parse_tagged_script
from podcast_generator.speakers import ensure_voice_files, resolve_voice_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    try:
        await asyncio.to_thread(chatterbox.start_chatterbox, settings.tts_default_language)
    except Exception:
        logger.exception("Chatterbox failed to start; TTS endpoints will fail until it loads")
    yield


app = FastAPI(title="Podcast Generator", version="0.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Source material for the podcast dialogue")
    assistant_prompt: str = Field(
        default="",
        description="Optional assistant message inserted in the chat before the user/source block",
    )


def _settings() -> Settings:
    return get_settings()


@app.get("/health")
async def health(settings: Annotated[Settings, Depends(_settings)]):
    ollama_ok = False
    try:
        base = settings.ollama_base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base}/api/tags")
            ollama_ok = r.is_success
    except Exception:
        pass

    tts_ok = chatterbox.tts_model is not None
    voices_ok = bool(settings.speaker_1_voice and settings.speaker_2_voice)
    ok = ollama_ok and tts_ok and voices_ok

    return {
        "status": "ok" if ok else "degraded",
        "ollama_reachable": ollama_ok,
        "chatterbox_loaded": tts_ok,
        "voices_configured": voices_ok,
        "ollama_model": settings.ollama_model,
        "device": chatterbox.device,
    }


class GenerateResponse(BaseModel):
    task_id: str


class PreviewResponse(BaseModel):
    transcript: str
    segment_count: int


class TaskStatus(BaseModel):
    task_id: str
    status: str
    error: str | None = None
    transcript: str | None = None
    segment_count: int | None = None


# In-memory storage for background tasks
tasks: Dict[str, Dict[str, Any]] = {}


async def process_preview_task(task_id: str, body: GenerateRequest, settings: Settings):
    try:
        tasks[task_id]["status"] = "generating_script"
        transcript = await asyncio.to_thread(
            generate_podcast_script,
            body.content,
            settings,
            assistant_prompt=body.assistant_prompt,
        )

        tasks[task_id]["status"] = "parsing_script"
        segments = parse_tagged_script(transcript)
        if not segments:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = "LLM output had no [speaker] segments; check prompts and model output."
            return

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = {
            "transcript": transcript,
            "segment_count": len(segments),
        }
        logger.info(
            "Preview Task %s completed: segments=%s",
            task_id,
            len(segments),
        )
    except Exception as e:
        logger.exception("Preview Task %s failed", task_id)
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


async def process_podcast_task(task_id: str, body: GenerateRequest, settings: Settings):
    try:
        tasks[task_id]["status"] = "generating_script"
        transcript = await asyncio.to_thread(
            generate_podcast_script,
            body.content,
            settings,
            assistant_prompt=body.assistant_prompt,
        )

        tasks[task_id]["status"] = "parsing_script"
        segments = parse_tagged_script(transcript)
        if not segments:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = "LLM output had no [speaker] segments; check prompts and model output."
            return

        tasks[task_id]["status"] = "synthesizing_audio"
        wav_parts: list[bytes] = []
        for i, seg in enumerate(segments):
            try:
                voice_path = resolve_voice_path(seg.speaker_key, settings)
                wav_bytes = await asyncio.to_thread(
                    chatterbox.synthesize_clone_wav,
                    seg.text,
                    str(voice_path),
                    settings.tts_default_language,
                    settings.tts_default_language,
                )
                wav_parts.append(wav_bytes)
                logger.info("Task %s: Synthesized segment %d/%d", task_id, i + 1, len(segments))
            except Exception as e:
                logger.exception("Task %s: TTS failed for segment", task_id)
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = f"TTS failed: {e}"
                return

        tasks[task_id]["status"] = "merging_audio"
        merged, sr = await asyncio.to_thread(concat_wav_bytes, wav_parts)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = merged
        logger.info(
            "Task %s completed: segments=%s sample_rate=%s bytes=%s",
            task_id,
            len(segments),
            sr,
            len(merged),
        )
    except Exception as e:
        logger.exception("Task %s failed", task_id)
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


@app.post("/podcast/generate", response_model=GenerateResponse)
async def generate_podcast(
        body: GenerateRequest,
        settings: Annotated[Settings, Depends(_settings)],
):
    if chatterbox.tts_model is None:
        raise HTTPException(status_code=503, detail="Chatterbox model is not loaded")

    try:
        ensure_voice_files(settings)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "result": None, "error": None}

    # Start the background task
    asyncio.create_task(process_podcast_task(task_id, body, settings))

    return GenerateResponse(task_id=task_id)


@app.get("/podcast/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] == "completed":
        result = task["result"]
        if isinstance(result, bytes):
            return Response(content=result, media_type="audio/wav")
        else:
            # For preview tasks, result is a dict.
            # We can return it as JSON.
            return TaskStatus(
                task_id=task_id,
                status=task["status"],
                transcript=result.get("transcript"),
                segment_count=result.get("segment_count")
            )
    elif task["status"] == "failed":
        return TaskStatus(task_id=task_id, status=task["status"], error=task["error"])
    else:
        return TaskStatus(task_id=task_id, status=task["status"])


@app.get("/podcast/task/{task_id}/status", response_model=TaskStatus)
async def get_task_status_only(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    status = task["status"]
    error = task.get("error")
    transcript = None
    segment_count = None

    if status == "completed" and isinstance(task.get("result"), dict):
        result = task["result"]
        transcript = result.get("transcript")
        segment_count = result.get("segment_count")

    return TaskStatus(
        task_id=task_id,
        status=status,
        error=error,
        transcript=transcript,
        segment_count=segment_count,
    )


@app.post("/podcast/preview-script", response_model=GenerateResponse)
async def preview_script(
        body: GenerateRequest,
        settings: Annotated[Settings, Depends(_settings)],
):
    """LLM only — returns task_id to poll for transcript."""
    if not settings.ollama_model.strip():
        raise HTTPException(status_code=503, detail="OLLAMA_MODEL is not set")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending", "result": None, "error": None}

    # Start the background task
    asyncio.create_task(process_preview_task(task_id, body, settings))

    return GenerateResponse(task_id=task_id)
