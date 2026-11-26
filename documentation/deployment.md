# FakeCheck — Deployment & Operations Guide

**Date:** 2026-11-26  
**Note:** Local CPU/HTTP/browser verification has been completed; cloud/container deployment has **not** been tested.

---

## 1. Local Development Configuration
- Copy `backend/.env.example` to `backend/.env` if overriding defaults.
- Default `FAKECHECK_CORS_ORIGINS` is `http://127.0.0.1:5500,http://localhost:5500`.

## 2. Production Configuration Requirements
- **CORS:** Set `FAKECHECK_CORS_ORIGINS` to your exact HTTPS frontend origin(s) (never `*` with credentials).
- **Upload Limits:** `FAKECHECK_MAX_IMAGE_MB=15` and `FAKECHECK_MAX_VIDEO_MB=100` (enforced with `HTTP 413`).
- **Reverse Proxy / HTTPS:** Terminate TLS at Nginx/Caddy/ALB and forward `X-Forwarded-For` / `X-Forwarded-Proto` to Uvicorn.

## 3. CPU, Memory & Model Loading Behavior
- Models are **lazy-loaded** on first prediction request using per-model `threading.Lock()` singletons (~573 MB for `M3_clip`, ~1.7 GB for `BLIP`, ~98 MB for `Xception+BiLSTM`; ~2.4 GB+ total if all three are loaded).
- `GET /api/health` checks `checkpoint_exists` and `runtime_state` (`not_initialized`, `loading`, `ready`, `failed`, `missing_checkpoint`, `unimplemented`) without loading weights.
- `Salesforce/blip-itm-base-coco` (revision `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`) must be present in the HuggingFace cache for `local_files_only=True` offline loading, and `backend/models/clip_processor/` must be present for `M3_clip`.
