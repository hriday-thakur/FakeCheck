"""Automated Phase 1 API tests (no dataset, checkpoint, or weight downloads)."""

import asyncio
import httpx
from app.main import app


async def _request(method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _run(coro):
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        return runner.run(coro)


def test_root_serves_frontend():
    resp = _run(_request("GET", "/"))
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "FAKE/CHECK" in resp.text


def test_health_endpoint():
    resp = _run(_request("GET", "/api/health"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert len(data["models"]) == 11
    for m in data["models"]:
        if m["model_id"] in {"fk_m3_clip", "snli_blip", "ff_xception_bilstm"}:
            assert m["inference_ready"] is True
        else:
            assert m["inference_ready"] is False
        assert isinstance(m["checkpoint_exists"], bool)


def test_benchmark_endpoint():
    resp = _run(_request("GET", "/api/benchmark"))
    assert resp.status_code == 200
    data = resp.json()
    assert "datasets" in data
    assert set(data["datasets"].keys()) == {"fakeddit", "snlive", "faceforensics"}
    assert len(data["datasets"]["fakeddit"]) >= 4
    assert len(data["datasets"]["snlive"]) >= 3


def test_fakeddit_validation_and_503():
    r_bad = _run(_request("POST", "/api/predict/fakeddit", data={"model_id": "fk_m1_text", "mode": "text-only"}))
    assert r_bad.status_code == 422

    r_unsupported = _run(_request("POST", "/api/predict/fakeddit", data={"model_id": "unknown_model", "mode": "text-only", "text": "x"}))
    assert r_unsupported.status_code == 400

    r_bad_img = _run(
        _request(
            "POST",
            "/api/predict/fakeddit",
            data={"model_id": "fk_m3_clip", "mode": "multimodal", "text": "sample headline"},
            files={"file": ("clip.mp4", b"not-an-image", "video/mp4")},
        )
    )
    assert r_bad_img.status_code == 415

    r_ok = _run(
        _request("POST", "/api/predict/fakeddit", data={"model_id": "fk_m1_text", "mode": "text-only", "text": "sample headline"})
    )
    assert r_ok.status_code == 503


def test_fakeddit_clip_real_prediction():
    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="JPEG")
    buf.seek(0)

    resp = _run(
        _request(
            "POST",
            "/api/predict/fakeddit",
            data={"model_id": "fk_m3_clip", "mode": "multimodal", "text": "Breaking news: local event confirmed"},
            files={"file": ("test.jpg", buf.read(), "image/jpeg")},
        )
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in {"REAL", "FAKE"}
    assert "REAL" in data["scores"] and "FAKE" in data["scores"]
    assert abs(data["scores"]["REAL"] + data["scores"]["FAKE"] - 1.0) < 1e-5
    assert 0.0 <= data["confidence"] <= 1.0
    assert "model_name" in data and "architecture" in data and "processing_time" in data


def test_snlive_validation_and_503():
    # Missing hypothesis -> 422
    r_bad = _run(
        _request(
            "POST",
            "/api/predict/snlive",
            data={"model_id": "snli_blip", "mode": "multimodal"},
            files={"file": ("img.jpg", b"fake-jpeg-bytes", "image/jpeg")},
        )
    )
    assert r_bad.status_code == 422

    # Missing image -> 422
    r_no_img = _run(
        _request(
            "POST",
            "/api/predict/snlive",
            data={"model_id": "snli_blip", "mode": "multimodal", "hypothesis": "a person walking"},
        )
    )
    assert r_no_img.status_code == 422

    # Invalid MIME type -> 415
    r_mime = _run(
        _request(
            "POST",
            "/api/predict/snlive",
            data={"model_id": "snli_blip", "mode": "multimodal", "hypothesis": "a person walking"},
            files={"file": ("clip.mp4", b"not-an-image", "video/mp4")},
        )
    )
    assert r_mime.status_code == 415

    # Corrupted image bytes -> 422
    r_corrupt = _run(
        _request(
            "POST",
            "/api/predict/snlive",
            data={"model_id": "snli_blip", "mode": "multimodal", "hypothesis": "a person walking"},
            files={"file": ("bad.jpg", b"not-valid-jpeg-bytes", "image/jpeg")},
        )
    )
    assert r_corrupt.status_code == 422

    # Unknown model_id -> 400
    r_unknown = _run(
        _request(
            "POST",
            "/api/predict/snlive",
            data={"model_id": "unknown_snli", "mode": "multimodal", "hypothesis": "a person walking"},
            files={"file": ("img.jpg", b"fake-jpeg-bytes", "image/jpeg")},
        )
    )
    assert r_unknown.status_code == 400

    # Secondary unimplemented model -> 503
    r_sec = _run(
        _request(
            "POST",
            "/api/predict/snlive",
            data={"model_id": "snli_vilt", "mode": "multimodal", "hypothesis": "a person walking"},
            files={"file": ("img.jpg", b"fake-jpeg-bytes", "image/jpeg")},
        )
    )
    assert r_sec.status_code == 503


def test_snlive_blip_real_prediction_and_singleton():
    from io import BytesIO
    from PIL import Image
    from app.main import get_snli_blip_predictor

    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    r1 = _run(
        _request(
            "POST",
            "/api/predict/snlive",
            data={"model_id": "snli_blip", "mode": "multimodal", "hypothesis": "A person standing outdoors"},
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert set(d1["scores"].keys()) == {"entailment", "neutral", "contradiction"}
    probs = list(d1["scores"].values())
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-5
    assert abs(d1["confidence"] - max(probs)) < 1e-6
    assert d1["prediction"] == max(d1["scores"], key=d1["scores"].get)

    p1 = get_snli_blip_predictor()
    r2 = _run(
        _request(
            "POST",
            "/api/predict/snlive",
            data={"model_id": "snli_blip", "mode": "multimodal", "hypothesis": "A person standing outdoors"},
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
    )
    assert r2.status_code == 200
    assert get_snli_blip_predictor() is p1


def test_faceforensics_validation_and_503():
    r_bad = _run(
        _request(
            "POST",
            "/api/predict/faceforensics",
            data={"model_id": "ff_xception_bilstm", "mode": "video-only"},
            files={"file": ("img.jpg", b"not-a-video", "image/jpeg")},
        )
    )
    assert r_bad.status_code == 415

    # Corrupt MP4 -> 422
    r_corrupt = _run(
        _request(
            "POST",
            "/api/predict/faceforensics",
            data={"model_id": "ff_xception_bilstm", "mode": "video-only"},
            files={"file": ("clip.mp4", b"fake-mp4-bytes", "video/mp4")},
        )
    )
    assert r_corrupt.status_code == 422

    # Secondary unimplemented model -> 503
    r_sec = _run(
        _request(
            "POST",
            "/api/predict/faceforensics",
            data={"model_id": "ff_efficientnet_b4", "mode": "video-only"},
            files={"file": ("clip.mp4", b"fake-mp4-bytes", "video/mp4")},
        )
    )
    assert r_sec.status_code == 503


def test_faceforensics_real_prediction_and_singleton():
    import os
    import tempfile
    import cv2
    import numpy as np
    from app.main import get_faceforensics_predictor

    # Create a tiny synthetic 4-frame (<8) MP4 video
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
        for i in range(4):
            frame = np.full((64, 64, 3), 40 * (i + 1), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        with open(tmp_path, "rb") as f:
            mp4_bytes = f.read()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    r1 = _run(
        _request(
            "POST",
            "/api/predict/faceforensics",
            data={"model_id": "ff_xception_bilstm", "mode": "video-only"},
            files={"file": ("synthetic.mp4", mp4_bytes, "video/mp4")},
        )
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["prediction"] in {"REAL", "FAKE"}
    assert set(d1["scores"].keys()) == {"REAL", "FAKE"}
    probs = [d1["scores"]["REAL"], d1["scores"]["FAKE"]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-5
    assert abs(d1["confidence"] - max(probs)) < 1e-6
    assert d1["frames_sampled"] == 4
    assert len(d1["warnings"]) >= 1

    p1 = get_faceforensics_predictor()
    r2 = _run(
        _request(
            "POST",
            "/api/predict/faceforensics",
            data={"model_id": "ff_xception_bilstm", "mode": "video-only"},
            files={"file": ("synthetic.mp4", mp4_bytes, "video/mp4")},
        )
    )
    assert r2.status_code == 200
    assert get_faceforensics_predictor() is p1


def test_faceforensics_upload_size_limit(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_video_mb", 1)
    one_mb = 1024 * 1024

    # Exactly at limit (1 MB) -> passes size check, fails MP4 decode with 422
    r_exact = _run(
        _request(
            "POST",
            "/api/predict/faceforensics",
            data={"model_id": "ff_xception_bilstm", "mode": "video-only"},
            files={"file": ("exact.mp4", b"0" * one_mb, "video/mp4")},
        )
    )
    assert r_exact.status_code == 422

    # Exceeding limit (1 MB + 1 byte) -> 413 Payload Too Large
    r_over = _run(
        _request(
            "POST",
            "/api/predict/faceforensics",
            data={"model_id": "ff_xception_bilstm", "mode": "video-only"},
            files={"file": ("over.mp4", b"0" * (one_mb + 1), "video/mp4")},
        )
    )
    assert r_over.status_code == 413


def test_concurrent_singleton_initialization(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time
    import app.main as m

    monkeypatch.setattr(m, "_fakeddit_clip_predictor", None)
    monkeypatch.setattr(m, "_snli_blip_predictor", None)
    monkeypatch.setattr(m, "_faceforensics_predictor", None)

    counts = {"fk": 0, "snli": 0, "ff": 0}

    class Dummy:
        def __init__(self, key):
            time.sleep(0.02)
            counts[key] += 1

    monkeypatch.setattr(m, "FakedditClipPredictor", lambda **kw: Dummy("fk"))
    monkeypatch.setattr(m, "SnliBlipPredictor", lambda **kw: Dummy("snli"))
    monkeypatch.setattr(m, "FaceForensicsPredictor", lambda **kw: Dummy("ff"))

    with ThreadPoolExecutor(max_workers=6) as ex:
        fk_res = list(ex.map(lambda _: m.get_fakeddit_clip_predictor(), range(4)))
        snli_res = list(ex.map(lambda _: m.get_snli_blip_predictor(), range(4)))
        ff_res = list(ex.map(lambda _: m.get_faceforensics_predictor(), range(4)))

    assert counts == {"fk": 1, "snli": 1, "ff": 1}
    assert len(set(id(x) for x in fk_res)) == 1
    assert len(set(id(x) for x in snli_res)) == 1
    assert len(set(id(x) for x in ff_res)) == 1


def test_cors_allowed_and_rejected_origins():
    r_ok = _run(
        _request(
            "OPTIONS",
            "/api/health",
            headers={"Origin": "http://127.0.0.1:5500", "Access-Control-Request-Method": "GET"},
        )
    )
    assert r_ok.status_code == 200
    assert r_ok.headers.get("access-control-allow-origin") == "http://127.0.0.1:5500"

    r_bad = _run(
        _request(
            "OPTIONS",
            "/api/health",
            headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "GET"},
        )
    )
    assert "access-control-allow-origin" not in r_bad.headers


def test_dynamic_health_readiness_states(monkeypatch):
    import app.main as m
    from pathlib import Path

    monkeypatch.setitem(m._model_runtime_states, "fk_m3_clip", "failed")
    r_fail = _run(_request("GET", "/api/health")).json()
    fk_entry = next(x for x in r_fail["models"] if x["model_id"] == "fk_m3_clip")
    snli_entry = next(x for x in r_fail["models"] if x["model_id"] == "snli_blip")
    assert fk_entry["runtime_state"] == "failed"
    assert fk_entry["inference_ready"] is False
    assert snli_entry["inference_ready"] is True

    monkeypatch.setattr(m.settings, "models_dir", Path("/nonexistent_models_dir"))
    r_miss = _run(_request("GET", "/api/health")).json()
    snli_miss = next(x for x in r_miss["models"] if x["model_id"] == "snli_blip")
    assert snli_miss["runtime_state"] == "missing_checkpoint"
    assert snli_miss["inference_ready"] is False
