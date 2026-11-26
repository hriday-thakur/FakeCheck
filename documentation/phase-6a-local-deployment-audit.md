# Phase 6A — Local Public Deployment Audit (Cloudflare Tunnel)

**Date:** 2026-11-26  
**Scope:** Read-only audit for single-origin public deployment via `Browser -> Cloudflare Tunnel -> Local FastAPI (127.0.0.1:8000)`.  
**Status:** Read-only audit complete; no files modified, no tunnel started.

---

## 1. Current Architecture & Startup Commands
- **Backend:** `backend/app/main.py` (`uvicorn app.main:app --host 127.0.0.1 --port 8000`), exposing `/api/health`, `/api/benchmark`, `/api/predict/fakeddit`, `/api/predict/snlive`, `/api/predict/faceforensics`.
- **Frontend:** `frontend/index.html` (self-contained single-file HTML/CSS/JS, currently served separately via `python -m http.server 5500 --directory frontend`).

---

## 2. API URL Configuration Findings (`frontend/index.html`)
- `fetchBenchmarkData()` and `runAnalysisBtn` read `base = $('apiBaseInput').value.trim().replace(/\/+$/, '')` (persisted in `localStorage` under `fakecheck_api_base_v1`) and call `${base}/api/benchmark` and `${base}/api/predict/${pipe.id}`.
- If `base` is `""` (empty) or `window.location.origin`, all requests automatically resolve to the **same origin** as the page.

---

## 3. Single-Origin Deployment Feasibility
- **Feasible with zero route conflicts:** All backend endpoints are under `/api/*`. Adding `GET /` returning `FileResponse(PROJECT_ROOT / "frontend" / "index.html")` in `backend/app/main.py` will serve the complete UI at `/` without shadowing `/api/*`.
- **Do NOT** mount `StaticFiles` over `PROJECT_ROOT` or `models/`—serving only `frontend/index.html` at `GET /` prevents any accidental filesystem exposure.

---

## 4. Required Minimal Code Changes (for Phase 6B)
1. **`backend/app/main.py`:** Add `GET /` returning `FileResponse(PROJECT_ROOT / "frontend" / "index.html")`.
2. **`frontend/index.html`:** When `localStorage.getItem('fakecheck_api_base_v1')` is unset, default `#apiBaseInput` to `window.location.origin` (instead of `http://127.0.0.1:8000`) so public Cloudflare Tunnel visitors work immediately with zero manual setup.

---

## 5. Security & Local Readiness
- **Interface Binding:** Keep Uvicorn bound to `127.0.0.1:8000` so only `cloudflared` on the local laptop can reach it.
- **Upload Limits:** `max_image_mb=15` and `max_video_mb=100` (`HTTP 413`) are already enforced.
- **Memory:** All 3 primary models lazy-load via `threading.Lock()` singletons (~2.5–3.1 GB total RAM if all 3 are loaded).
- **Cloudflare Tunnel Command (Phase 6B):** `cloudflared tunnel --url http://127.0.0.1:8000`.
