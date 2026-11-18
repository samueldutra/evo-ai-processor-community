"""Voice notes are transcribed and handed to the model as text (fork addition)."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from google.genai.types import Blob, Part

from src.schemas.chat import FileData
from src.services.adk import audio_transcription
from src.services.adk.audio_transcription import audio_extension, transcribe_audio
from src.services.adk.runners.runner_utils import RunnerUtils


def _run(coro):
    return asyncio.run(coro)


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_extension_for_whatsapp_opus():
    assert audio_extension("audio/ogg; codecs=opus", "voice.opus") == "ogg"
    assert audio_extension("audio/opus", None) == "ogg"
    assert audio_extension("audio/mpeg", None) == "mp3"
    assert audio_extension(None, "note.m4a") == "m4a"
    assert audio_extension(None, None) == "ogg"


def test_transcribe_posts_audio_and_returns_text(monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "  quanto custa a wahl magic clip  "})

    monkeypatch.setenv("AUDIO_TRANSCRIPTION_PROMPT", "Wahl, Babyliss")
    text = _run(transcribe_audio("sk-test", b"OggS-bytes", "audio/ogg; codecs=opus", "v.ogg", _client(handler)))

    assert text == "quanto custa a wahl magic clip"
    assert seen["auth"] == "Bearer sk-test"
    assert b'filename="audio.ogg"' in seen["body"]
    assert b"Wahl, Babyliss" in seen["body"]
    assert b"gpt-4o-mini-transcribe" in seen["body"]


def test_falls_back_to_whisper_when_model_rejected():
    models = []

    def handler(request):
        body = request.content
        models.append(b"whisper-1" in body)
        if b"whisper-1" in body:
            return httpx.Response(200, json={"text": "oi"})
        return httpx.Response(404, json={"error": "model not found"})

    assert _run(transcribe_audio("k", b"x", "audio/ogg", None, _client(handler))) == "oi"
    assert models == [False, True]


def test_returns_none_on_http_error_and_without_key():
    def boom(request):
        return httpx.Response(500, text="nope")

    assert _run(transcribe_audio("k", b"x", "audio/ogg", None, _client(boom))) is None
    assert _run(transcribe_audio("", b"x")) is None
    assert _run(transcribe_audio("k", b"")) is None


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("AUDIO_TRANSCRIPTION", "false")
    assert audio_transcription.is_enabled() is False
    monkeypatch.delenv("AUDIO_TRANSCRIPTION")
    assert audio_transcription.is_enabled() is True


# --- process_files integration -------------------------------------------------


def _file(name, ctype):
    return FileData(filename=name, content_type=ctype, data=base64.b64encode(b"bytes").decode())


def _artifacts():
    a = MagicMock()
    a.save_artifact = AsyncMock()
    return a


def _utils(key):
    u = RunnerUtils.__new__(RunnerUtils)
    u._openai_key_for_agent = lambda agent_id: key
    return u


def test_audio_is_transcribed_and_not_sent_to_the_model(monkeypatch):
    monkeypatch.setattr(
        audio_transcription, "transcribe_audio", AsyncMock(return_value="tem a magic clip?")
    )
    parts, texts = _run(
        _utils("sk").process_files([_file("v.ogg", "audio/ogg")], _artifacts(), "a", "e", "s")
    )
    assert parts == []
    assert texts == ["[Áudio do cliente transcrito]: tem a magic clip?"]


def test_failed_transcription_leaves_a_marker_and_no_raw_audio(monkeypatch):
    monkeypatch.setattr(audio_transcription, "transcribe_audio", AsyncMock(return_value=None))
    parts, texts = _run(
        _utils("sk").process_files([_file("v.ogg", "audio/ogg")], _artifacts(), "a", "e", "s")
    )
    assert parts == []
    assert len(texts) == 1 and "não foi possível transcrever" in texts[0]


def test_image_next_to_audio_still_reaches_the_model(monkeypatch):
    monkeypatch.setattr(audio_transcription, "transcribe_audio", AsyncMock(return_value="oi"))
    parts, texts = _run(
        _utils("sk").process_files(
            [_file("p.png", "image/png"), _file("v.ogg", "audio/ogg")], _artifacts(), "a", "e", "s"
        )
    )
    assert [p.inline_data.mime_type for p in parts] == ["image/png"]
    assert len(texts) == 1


def test_without_an_openai_key_audio_keeps_the_old_inline_behavior():
    parts, texts = _run(
        _utils(None).process_files([_file("v.ogg", "audio/ogg")], _artifacts(), "a", "e", "s")
    )
    assert len(parts) == 1 and texts == []


def test_disabled_flag_keeps_the_old_inline_behavior(monkeypatch):
    monkeypatch.setenv("AUDIO_TRANSCRIPTION", "false")
    parts, texts = _run(
        _utils("sk").process_files([_file("v.ogg", "audio/ogg")], _artifacts(), "a", "e", "s")
    )
    assert len(parts) == 1 and texts == []
