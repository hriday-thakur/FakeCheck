# Phase 5A — Integration Hardening & Release Readiness Audit

**Date:** 2026-11-26  
**Scope:** Read-only readiness audit of `backend/`, `frontend/index.html`, `models/`, and `documentation/`.  
**Verdict:** **READY WITH CONDITIONS** (All three primary models pass local E2E and `pytest`, but production hardening items—such as MP4 upload size limits, CORS configuration, and missing startup/deployment docs—remain).

---

## 1. Executive Summary
- **Core Functionality:** All three primary pipelines (`fk_m3_clip`, `snli_blip`, `ff_xception_bilstm`) are implemented, lazy-loaded via singletons, and verified end-to-end in both `pytest` (`8 passed`) and live browser testing (`0` console errors).
- **Benchmark Integrity:** Every metric served by `GET /api/benchmark` is read directly from original CSV evaluation artifacts (`E1_baselines_vs_multimodal.csv`, `snlive-four-model-test-comparison.csv`, `video_level_comparison.csv`). Specifically, FaceForensics++ `Xception + BiLSTM video-level` precision (`91.00%` / `0.9100129`) and recall (`88.15%` / `0.8815436`) match `video_level_comparison.csv` line 4 exactly, and video-level vs. frame-level rows are kept distinct.

---

## 2. Findings by Severity

### CRITICAL
- *None observed.*

### HIGH
1. **No Upload Size Limit on MP4 Videos (`backend/app/main.py`):** **RESOLVED (Phase 5B)** — Added `max_video_mb: int = 100` (`FAKECHECK_MAX_VIDEO_MB=100` in `.env.example` and `config.py`) and enforced `max_mb=settings.max_video_mb` in `POST /api/predict/faceforensics` (`413` on oversized uploads).
2. **Thread-Safety of Lazy Singletons (`backend/app/main.py`):** **RESOLVED (Phase 5B)** — Added per-model `threading.Lock()` (`_fakeddit_clip_lock`, `_snli_blip_lock`, `_faceforensics_lock`) with double-checked locking; locks are not held during inference.

*Verification:* `pytest tests/test_api.py -vv` (**10 passed**), `node --check` (**0 errors**).

### MEDIUM
1. **Permissive Default CORS (`backend/app/config.py`):** **RESOLVED (Phase 5C)** — Default `FAKECHECK_CORS_ORIGINS` is now `http://127.0.0.1:5500,http://localhost:5500`; `allow_credentials` is disabled if `*` is ever configured.
2. **`health` Endpoint Static `inference_ready` (`backend/app/main.py`):** **RESOLVED (Phase 5C)** — `GET /api/health` now computes `runtime_state` (`not_initialized`, `loading`, `ready`, `failed`, `missing_checkpoint`, `unimplemented`) and `inference_ready` dynamically without eager model loading.
3. **Fakeddit `text-only` Mode Removed from Frontend (`frontend/index.html`):**
   - `fk_m3_clip` requires `multimodal` (image + text), so `text-only` is not exposed in `PIPELINES`; this is accurate for `M3_clip_best.pt` but limits text-only use until `fk_m1_text` is implemented.

### LOW
1. **Pydantic `model_id` Protected Namespace Warning:** Mitigated in `Settings` (`protected_namespaces=()`).
2. **Missing `README.md` / Production Startup Guide:** **RESOLVED (Phase 5C)** — Created root `README.md` and `documentation/deployment.md`.

---

## 3. Tests Already Passing
- `backend/tests/test_api.py`: **8 passed** (`test_health_endpoint`, `test_benchmark_endpoint`, `test_fakeddit_validation_and_503`, `test_fakeddit_clip_real_prediction`, `test_snlive_validation_and_503`, `test_snlive_blip_real_prediction_and_singleton`, `test_faceforensics_validation_and_503`, `test_faceforensics_real_prediction_and_singleton`).
- `frontend/index.html`: `node --check` passed (`0` errors); live browser E2E passed for all 3 primary models, benchmark sync, and Light/Dark theme (`0` console errors).

---

## 4. Missing Tests
- Oversized file (`413 Payload Too Large`) test for images (>15 MB) and videos.
- Concurrent request / thread-safety test on cold-start singleton initialization.
- Missing-checkpoint / missing-processor `503` runtime failure test.

---

## 5. Recommended Fixes (Priority Order)
1. Add `max_mb` limit (e.g., `120` MB matching `frontend/index.html` line 2970) to `_validate_upload` in `POST /api/predict/faceforensics`.
2. Add `threading.Lock()` around lazy singleton initialization in `backend/app/main.py`.
3. Restrict `FAKECHECK_CORS_ORIGINS` in `.env` for production deployments and add a root `README.md` with startup instructions.

---

## 6. Release Readiness Verdict
**READY WITH CONDITIONS** — Suitable for local/demo deployment now; apply the `max_mb` video limit, singleton locking, and CORS restriction before public production exposure.
