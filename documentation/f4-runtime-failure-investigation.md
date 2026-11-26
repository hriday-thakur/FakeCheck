# F4 Runtime Failure Investigation (`HTTP 400`)

**Date:** 2026-11-27  
**Endpoint:** `POST http://127.0.0.1:8000/api/predict/fakeddit` (`model_id=fk_f4_gated`)

---

## 1. Reproduced HTTP Response
- **Status:** `400 Bad Request`
- **Response Body:** `{"detail":"Unknown model_id 'fk_f4_gated' for pipeline 'fakeddit'."}`

---

## 2. Root Cause
- The failure occurs **before model initialization** in `_validate_model("fakeddit", "fk_f4_gated")`.
- The Uvicorn server process currently bound to `127.0.0.1:8000` was started **before** F4 Phase 2 integration added `fk_f4_gated` to `backend/app/registry.py` and `backend/app/main.py` (whereas `pytest` running against the current code passes `14/14`, including `test_fakeddit_f4_default_and_real_prediction`).

---

## 3. Resolution
- No code changes are required. Stop the stale Uvicorn process on port `8000` and restart it from `backend/`:
  ```powershell
  cd backend
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
  ```
