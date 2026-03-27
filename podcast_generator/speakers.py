from pathlib import Path

from podcast_generator.config import Settings


def resolve_voice_path(speaker_tag: str, settings: Settings) -> Path:
    """Map LLM tag text to reference WAV path; names compared case-insensitively."""
    tag = speaker_tag.strip()
    n1 = settings.speaker_1_name.strip()
    n2 = settings.speaker_2_name.strip()
    lower = tag.lower()
    if lower == n1.lower():
        return Path(settings.speaker_1_voice)
    if lower == n2.lower():
        return Path(settings.speaker_2_voice)
    raise ValueError(f"Unknown speaker tag [{speaker_tag!r}]; expected [{n1}] or [{n2}]")


def ensure_voice_files(settings: Settings) -> None:
    for label, path in (
            ("SPEAKER_1_VOICE", settings.speaker_1_voice),
            ("SPEAKER_2_VOICE", settings.speaker_2_voice),
    ):
        p = Path(path)
        if not path:
            raise ValueError(f"{label} is not set")
        if not p.is_file():
            raise ValueError(f"{label} must point to an existing WAV file: {p}")
