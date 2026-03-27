import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaggedSegment:
    speaker_key: str
    text: str


_TAG = re.compile(r"\[([^\]]+)\]")


def parse_tagged_script(raw: str) -> list[TaggedSegment]:
    """Split LLM output: each `[SpeakerName]` starts a segment until the next tag or EOF."""
    raw = raw.strip()
    if not raw:
        return []

    matches = list(_TAG.finditer(raw))
    if not matches:
        return []

    segments: list[TaggedSegment] = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        text = raw[start:end].strip()
        if name and text:
            segments.append(TaggedSegment(speaker_key=name, text=text))
    return segments
