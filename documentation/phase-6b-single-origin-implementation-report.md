# Phase 6B — Single-Origin Frontend & Backend Integration Report

**Date:** 2026-11-26  
**Status:** **PASSED** (Single-origin `GET /` + `/api/*` verified via `pytest` and `node --check`).

---

## 1. Files Modified
1. **`backend/app/main.py`:** Added `GET /` (`serve_frontend()`) returning `FileResponse(PROJECT_ROOT / "frontend" / "index.html", media_type="text/html")`, returning `404` if missing, without exposing `backend/` or `models/`.
2. **`frontend/index.html`:** Updated `#apiBaseInput` initialization (line 2586) to default to `window.location.origin` when `localStorage.getItem('fakecheck_api_base_v1')` is unset, preserving explicitly saved URLs and leaving all CSS/layout untouched.
3. **`backend/tests/test_api.py`:** Added `test_root_serves_frontend`.

---

## 2. Local Startup & Access
```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000/`** in a browser (single-origin; no separate static server required).
