"""Audio transcription for incoming voice messages.

WhatsApp voice notes arrive as ``audio/ogg; codecs=opus``. Chat models such as
gpt-4.1 cannot take that as an inline part -- the request stalls until the
caller's timeout instead of failing fast. Transcribing it first and handing the
model plain text is both reliable and cheap.

Fork addition (not upstream). Controlled by environment:
  AUDIO_TRANSCRIPTION            "false" disables it (default: enabled)
  AUDIO_TRANSCRIPTION_MODEL      default "gpt-4o-mini-transcribe"
  AUDIO_TRANSCRIPTION_LANGUAGE   optional ISO-639-1 hint, e.g. "pt"
  AUDIO_TRANSCRIPTION_PROMPT     optional vocabulary hint for brand/model names
  AUDIO_TRANSCRIPTION_TIMEOUT    seconds, default 40
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_MODEL = "gpt-4o-mini-transcribe"
FALLBACK_MODEL = "whisper-1"

# OpenAI accepts: flac mp3 mp4 mpeg mpga m4a ogg wav webm. WhatsApp's opus is
# an ogg container, so it goes up as .ogg.
_EXTENSIONS = {
    "ogg": "ogg",
    "opus": "ogg",
    "vorbis": "ogg",
    "mpeg": "mp3",
    "mp3": "mp3",
    "mp4": "m4a",
    "m4a": "m4a",
    "x-m4a": "m4a",
    "aac": "m4a",
    "wav": "wav",
    "wave": "wav",
    "x-wav": "wav",
    "flac": "flac",
    "x-flac": "flac",
    "webm": "webm",
}


def is_enabled() -> bool:
    return os.getenv("AUDIO_TRANSCRIPTION", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def audio_extension(content_type: Optional[str], filename: Optional[str]) -> str:
    """Pick the file extension OpenAI needs to recognise the container."""
    if content_type:
        subtype = content_type.split(";")[0].strip().lower().rpartition("/")[2]
        if subtype in _EXTENSIONS:
            return _EXTENSIONS[subtype]
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        if ext in _EXTENSIONS:
            return _EXTENSIONS[ext]
        if ext in set(_EXTENSIONS.values()):
            return ext
    return "ogg"


async def transcribe_audio(
    api_key: str,
    data: bytes,
    content_type: Optional[str] = None,
    filename: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[str]:
    """Return the transcript, or None when it could not be produced."""
    if not api_key or not data:
        return None

    model = os.getenv("AUDIO_TRANSCRIPTION_MODEL", DEFAULT_MODEL)
    timeout = float(os.getenv("AUDIO_TRANSCRIPTION_TIMEOUT", "40"))
    upload_name = f"audio.{audio_extension(content_type, filename)}"
    mime = (content_type or "audio/ogg").split(";")[0].strip() or "audio/ogg"

    form = {"model": model, "response_format": "json"}
    if language := os.getenv("AUDIO_TRANSCRIPTION_LANGUAGE", "").strip():
        form["language"] = language
    if prompt := os.getenv("AUDIO_TRANSCRIPTION_PROMPT", "").strip():
        form["prompt"] = prompt

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt_model in (model, FALLBACK_MODEL) if model != FALLBACK_MODEL else (model,):
            form["model"] = attempt_model
            try:
                response = await client.post(
                    TRANSCRIPTIONS_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=form,
                    files={"file": (upload_name, data, mime)},
                )
            except httpx.HTTPError as exc:
                logger.error(f"Audio transcription request failed: {exc}")
                return None

            if response.status_code == 200:
                text = (response.json().get("text") or "").strip()
                return text or None

            # A model the account cannot use -> try whisper-1 once.
            if response.status_code in (400, 403, 404) and attempt_model != FALLBACK_MODEL:
                logger.warning(
                    f"Transcription model {attempt_model} rejected "
                    f"({response.status_code}); retrying with {FALLBACK_MODEL}"
                )
                continue

            logger.error(
                f"Audio transcription failed ({response.status_code}): {response.text[:200]}"
            )
            return None
        return None
    finally:
        if owns_client:
            await client.aclose()
