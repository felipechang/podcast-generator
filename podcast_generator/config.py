from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    speaker_1_name: str = Field(default="Speaker1", alias="SPEAKER_1_NAME")
    speaker_1_voice: str = Field(default="", alias="SPEAKER_1_VOICE")
    speaker_2_name: str = Field(default="Speaker2", alias="SPEAKER_2_NAME")
    speaker_2_voice: str = Field(default="", alias="SPEAKER_2_VOICE")

    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="glm-4.7-flash:latest", alias="OLLAMA_MODEL")
    ollama_temperature: float = Field(default=0.7, alias="OLLAMA_TEMPERATURE")
    ollama_timeout_s: float = Field(default=600.0, alias="OLLAMA_TIMEOUT_S")

    tts_default_language: str = Field(default="es", alias="TTS_DEFAULT_LANGUAGE")
    task_expiration_seconds: int = Field(default=3600, alias="TASK_EXPIRATION_SECONDS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
