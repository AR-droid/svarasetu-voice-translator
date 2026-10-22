import io

import soundfile as sf
from fastapi.testclient import TestClient

from conftest import tone
from voicenote import server


def test_translate_endpoint(fakes, monkeypatch):
    _, _, pipeline = fakes
    monkeypatch.setattr(server, "get_pipeline", lambda: pipeline)
    buffer = io.BytesIO()
    sf.write(buffer, tone(3.0), 16000, format="WAV")
    client = TestClient(server.app)
    response = client.post(
        "/translate",
        files={"audio": ("note.wav", buffer.getvalue(), "audio/wav")},
        data={"src_lang": "ta", "tgt_lang": "hi"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["translation"].startswith("[ta->hi]")
    assert body["audio_wav_base64"]


def test_bad_language_is_422(fakes, monkeypatch):
    _, _, pipeline = fakes
    monkeypatch.setattr(server, "get_pipeline", lambda: pipeline)
    buffer = io.BytesIO()
    sf.write(buffer, tone(1.0), 16000, format="WAV")
    response = TestClient(server.app).post(
        "/translate",
        files={"audio": ("note.wav", buffer.getvalue(), "audio/wav")},
        data={"src_lang": "ta", "tgt_lang": "fr"},
    )
    assert response.status_code == 422


def test_languages_endpoint():
    assert len(TestClient(server.app).get("/languages").json()) == 11
