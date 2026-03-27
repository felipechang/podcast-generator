import httpx

from podcast_generator.config import Settings, get_settings


def build_system_prompt(settings: Settings) -> str:
    s1, s2 = settings.speaker_1_name.strip(), settings.speaker_2_name.strip()
    return "\n\n".join(
        [
            "You write a podcast dialogue between two hosts.",
            f"Every line of dialogue MUST start with a tag on its own token: [{s1}] or [{s2}] using these exact names.",
            "Use one tag before each utterance. Do not use other speaker labels.",
            "Stay faithful to the source content; be conversational and concise.",
        ]
    )

def generate_podcast_script(
    content: str,
    settings: Settings | None = None,
    *,
    assistant_prompt: str = "",
) -> str:
    settings = settings or get_settings()
    if not settings.ollama_model.strip():
        raise ValueError("OLLAMA_MODEL is not set")

    base = settings.ollama_base_url.rstrip("/")
    url = f"{base}/api/chat"
    system = build_system_prompt(settings)

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "assistant", "content": assistant_prompt.strip()},
            {"role": "user", "content": content.strip()},
        ],
        "stream": False,
        "options": {"temperature": settings.ollama_temperature},
    }

    timeout = httpx.Timeout(settings.ollama_timeout_s)
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()

    msg = data.get("message") or {}
    text = msg.get("content")
    if not text:
        raise RuntimeError("Ollama returned empty content")
    return text.strip()
