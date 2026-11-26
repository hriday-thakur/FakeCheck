# Phase 4 — End-to-End Frontend Verification Test Plan

**Date:** 2026-11-26  
**Frontend:** `frontend/index.html`  
**Backend:** `backend/app/main.py` (`http://127.0.0.1:8000`)  
**Browser Automation:** Available (`browser` subagent).

---

## Test Cases

### 1. Fakeddit (`fk_m3_clip`)
- **Text-only submission:** Verify validation (`422` since `fk_m3_clip` requires `multimodal` image + text).
- **Image + text submission:** Upload JPG/PNG + headline, verify `model_id=fk_m3_clip`, `mode=multimodal`, `200 OK`, `REAL`/`FAKE` verdict, and probabilities summing to `1.0`.

### 2. SNLI-VE (`snli_blip`)
- **Image + hypothesis submission:** Upload JPG/PNG + hypothesis, verify `model_id=snli_blip`, `mode=multimodal`, `200 OK`, `entailment`/`neutral`/`contradiction` scores, and warning that SNLI-VE is visual entailment, not deepfake detection.

### 3. FaceForensics++ (`ff_xception_bilstm`)
- **MP4 upload:** Upload `video/mp4`, verify `model_id=ff_xception_bilstm`, `mode=video-only`, `200 OK`, `REAL`/`FAKE` verdict, processing time, and preprocessing warning.

### 4. Dashboard & Leaderboard
- Verify `GET /api/benchmark` loads and populates `LEADERBOARD_DATA` (`fk_m3_clip`, `snli_blip`, `ff_xception_bilstm`) with verified evaluation metrics only (no live prediction confidence).

### 5. Error Handling
- Missing required input (`422`), unsupported file type (`415`), backend unavailable (network error banner, no mock fallback), and `503` handling.

### 6. UI & Responsiveness
- Navigation across views (`Analyze`, `Result`, `Dashboard`, `Models`, `Datasets`, `History`), upload controls, Light/Dark theme toggle, responsive layout, and zero browser console errors.
