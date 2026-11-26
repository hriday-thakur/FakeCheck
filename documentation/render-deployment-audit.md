# Phase 6A — GitHub + Render Deployment Compatibility Audit

**Date:** 2026-11-26  
**Scope:** Read-only deployment compatibility audit for GitHub + Render.  
**Verdict:** **NOT COMPATIBLE WITH RENDER FREE (512 MB RAM)** — All three models cannot run on Render Free; use **Hugging Face Spaces (Free Docker Tier: 16 GB RAM, 2 vCPU)** for the backend + **Render Static Site / GitHub Pages** for the frontend.

---

## 1. Render Configuration
- **Backend Build Command:** `pip install -r backend/requirements.txt`
- **Backend Start Command:** `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
  - *Note:* `backend/app/config.py` uses `FAKECHECK_PORT`, whereas Render injects `$PORT`. Passing `--port $PORT` on the Uvicorn CLI works directly.
- **Static Frontend:** Publish directory `frontend` (serves `frontend/index.html` with zero build step).
- **Frontend API URL:** Configurable via the UI's API Settings modal (`#apiBaseInput`, persisted in `localStorage` as `fakecheck_api_base_v1`) without any UI redesign.

---

## 2. Storage & Memory Requirements
| Component | Disk Size | Estimated Runtime RAM |
| :--- | :--- | :--- |
| **Fakeddit (`M3_clip_best.pt` + `clip_processor/`)** | `572.92 MB` + `3.47 MB` | ~600–800 MB |
| **SNLI-VE (`blip-best.pt` + `Salesforce/blip-itm-base-coco`)** | `145.02 MB` + `1,708.15 MB` | ~1.7–2.0 GB |
| **FaceForensics++ (`xception_bilstm_video_best.pt` + `Xception_best.pt`)** | `18.02 MB` + `79.68 MB` | ~200–350 MB |
| **Total (All 3 Primary Models)** | **~2.53 GB** | **~2.5–3.1 GB** |

---

## 3. Render Free Verdict & Zero-Cost Fallback
- **Render Free (512 MB RAM, 0.1 CPU):** **Cannot run** `M3_clip` or `BLIP` (both exceed 512 MB RAM on load and will trigger OOM kills).
- **GitHub Storage:** `.pt` checkpoints (`572.92 MB`, `145.02 MB`, `79.68 MB`) exceed GitHub's `100 MB` file limit and must be excluded via `.gitignore` and downloaded at build/startup from Hugging Face Hub or object storage.
- **Zero-Cost Fallback:**
  1. **Backend:** **Hugging Face Spaces (Docker SDK, Free Tier — 16 GB RAM, 2 vCPU)**, which easily fits all three models (~3 GB RAM) and supports `$PORT` (`7860`).
  2. **Frontend:** **Render Static Site** or **GitHub Pages** (free), pointing `#apiBaseInput` + `FAKECHECK_CORS_ORIGINS` to the Space URL.
