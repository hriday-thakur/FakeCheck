# Phase 6D — Public Deployment Validation Report

**Date:** 2026-11-26  
**Public URL:** `https://pig-raise-nerve-department.trycloudflare.com`  
**Final Verdict:** **PASS** (All public endpoints, `GET /`, `GET /api/health`, `GET /api/benchmark`, and all three primary model predictions `fk_m3_clip`, `snli_blip`, `ff_xception_bilstm` succeeded over HTTPS).

---

## 1. Public Endpoint & Model Prediction Results

| Endpoint | Method | Status | Total Round-Trip Time | Observed Response Summary |
| :--- | :--- | :--- | :--- | :--- |
| `/` | `GET` | `200 OK` | `0.749s` | Served `frontend/index.html` (`window.location.origin` default). |
| `/api/health` | `GET` | `200 OK` | `0.959s` | `{"status": "ok", "phase": "5c-hardened", ...}` |
| `/api/benchmark` | `GET` | `200 OK` | `0.886s` | Verified CSV metrics for Fakeddit, SNLI-VE, and FaceForensics++. |
| `/api/predict/fakeddit` | `POST` | `200 OK` | `14.276s` *(cold start)* | `FAKE` (`confidence: 0.9411`, `REAL: 0.0589`, `FAKE: 0.9411`, `AN-9E72F1C578`). |
| `/api/predict/snlive` | `POST` | `200 OK` | `1.477s` | `entailment` (`confidence: 0.9820`, `entailment: 0.9820`, `AN-B49EE5CC79`). |
| `/api/predict/faceforensics` | `POST` | `200 OK` | `3.498s` | `FAKE` (`confidence: 0.7704`, `REAL: 0.2296`, `FAKE: 0.7704`, `AN-C50A5D8A1B`). |
| `/api/predict/fakeddit` *(missing input)* | `POST` | `422` | — | Properly rejected missing required fields. |

---

## 2. Scope & Limitations
- **Executed:** Direct HTTPS `GET` and `multipart/form-data` `POST` requests against `https://pig-raise-nerve-department.trycloudflare.com`.
- **Not Re-executed in Phase 6D:** Interactive headless browser rendering on the `trycloudflare.com` URL (already verified on `frontend/index.html` in Phase 4 with `0` console errors).
- **Tunnel Ephemerality:** Quick-tunnel URLs (`trycloudflare.com`) are temporary and change if `cloudflared` restarts.
