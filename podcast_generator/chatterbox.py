"""
In-process Chatterbox multilingual TTS (the same process as the API).
Mirrors the TTS setup from DESIGN.md. Heavy imports are deferred until startup.
"""

from __future__ import annotations

import io
import logging
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("TTS_MODEL_ID", "ResembleAI/chatterbox-multilingual")
TTS_NUM_THREADS = int(os.environ.get("TTS_NUM_THREADS", "0"))
TTS_INTEROP_THREADS = int(os.environ.get("TTS_INTEROP_THREADS", "1"))
TTS_WARMUP = os.environ.get("TTS_WARMUP", "true").lower() in {"1", "true", "yes", "on"}

device: str | None = None
tts_model: Any = None


def _patch_torch_load(torch: Any) -> None:
    original = torch.load

    def patched(f, map_location=None, **kwargs):
        if map_location is None:
            map_location = "cpu"
        return original(f, map_location=map_location, **kwargs)

    torch.load = patched


def configure_torch_runtime(torch: Any) -> None:
    torch.set_grad_enabled(False)
    if device == "cpu":
        cpu_threads = TTS_NUM_THREADS if TTS_NUM_THREADS > 0 else (os.cpu_count() or 1)
        cpu_threads = max(1, cpu_threads)
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, TTS_INTEROP_THREADS))
        except RuntimeError:
            pass
        logger.info(
            "CPU threading: num_threads=%s interop_threads=%s",
            cpu_threads,
            max(1, TTS_INTEROP_THREADS),
        )


def load_model(torch: Any) -> None:
    global tts_model
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as e:
        raise ImportError(
            "chatterbox-tts is not installed. Install with: pip install -e \".[tts]\""
        ) from e

    assert device is not None
    logger.info("Loading Chatterbox: %s on %s", MODEL_NAME, device)
    tts_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    logger.info("Chatterbox model loaded")


def run_warmup(torch: Any, language_id: str) -> None:
    if not TTS_WARMUP or tts_model is None:
        return
    try:
        with torch.inference_mode():
            _ = tts_model.generate("Warmup.", language_id=language_id)
        logger.info("Chatterbox warmup complete")
    except Exception as e:
        logger.warning("Chatterbox warmup skipped: %s", e)


def _normalize_language(language: str | None, default_language: str) -> str:
    aliases = {
        "spanish": "es",
        "espanol": "es",
        "español": "es",
        "english": "en",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "chinese": "zh",
    }

    def one(s: str | None) -> str:
        x = (s or "").strip().lower()
        if not x or x == "auto":
            return ""
        return aliases.get(x, x)

    lang = one(language)
    if lang:
        return lang
    return one(default_language) or "es"


def start_chatterbox(default_language: str) -> None:
    global device, tts_model

    try:
        import torch
    except ImportError:
        logger.exception(
            "PyTorch is required for Chatterbox. Install the `tts` extra: pip install -e \".[tts]\""
        )
        return

    _patch_torch_load(torch)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*flash-attn.*")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info("Chatterbox device: %s", device)
    configure_torch_runtime(torch)
    load_model(torch)
    lang = _normalize_language(default_language, default_language)
    run_warmup(torch, lang)


def synthesize_clone_wav(text: str, audio_prompt_path: str, language: str, default_language: str) -> bytes:
    import numpy as np
    import soundfile as sf
    import torch

    if tts_model is None:
        raise RuntimeError("Chatterbox model is not loaded")

    language_id = _normalize_language(language, default_language)
    kwargs: dict = {"text": text, "language_id": language_id}
    if audio_prompt_path:
        kwargs["audio_prompt_path"] = audio_prompt_path

    with torch.inference_mode():
        wav = tts_model.generate(**kwargs)

    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()
    if getattr(wav, "ndim", 1) > 1:
        wav = wav[0]

    sr = int(getattr(tts_model, "sr", 24000))
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
