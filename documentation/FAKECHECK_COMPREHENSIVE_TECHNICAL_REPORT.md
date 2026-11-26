# FakeCheck — Comprehensive Technical Project Report

**Project:** FakeCheck  
**Date:** 2026-11-26  
**Scope:** End-to-end engineering documentation covering Phases 1–6D (Model Audit, Inference Recovery, FastAPI Backend, Single-File Frontend Integration, Hardening, and Cloudflare Tunnel Public Deployment).

---

## 1. Project Overview
FakeCheck is a multi-pipeline content and manipulation detection platform integrating three independently trained PyTorch models behind a unified FastAPI backend (`backend/app/main.py`) and a Neo-Brutalist single-file frontend (`frontend/index.html`).

## 2. Abstract
This report documents the technical audit, source recovery, backend inference implementation, frontend integration, security hardening, and public deployment validation of FakeCheck. Three primary checkpoints—`M3_clip_best.pt` (Fakeddit), `blip-best.pt` (SNLI-VE), and `xception_bilstm_video_best.pt` + `Xception_best.pt` (FaceForensics++)—were recovered from training artifacts, wrapped in thread-safe lazy singletons, exposed via standardized `multipart/form-data` FastAPI endpoints, and validated end-to-end both locally (`pytest`: 13 passed) and over HTTPS (`https://pig-raise-nerve-department.trycloudflare.com`).

## 3. Problem Statement & Motivation
AI-generated and manipulated media span text-image misinformation, semantic image-text contradictions, and facial video manipulation. Integrating heterogeneous research checkpoints—often saved without standalone inference scripts, with partial `state_dict` exports, or with differing preprocessing requirements—into a reliable web application requires strict architectural verification, calibrated output mapping, and clear task-scope disclosures.

## 4. Project Objectives & Scope
- Audit all three model folders (`models/text`, `models/image`, `models/video`) without modifying or retraining checkpoints.
- Implement CPU-compatible, lazy-loaded FastAPI inference for the three team-selected primary models.
- Preserve the existing Neo-Brutalist UI in `frontend/index.html` while aligning `PIPELINES` and `LEADERBOARD_DATA` with the backend registry.
- Harden upload limits, CORS, and health/readiness reporting, and validate single-origin deployment via Cloudflare Tunnel.

## 5. System Architecture
```text
Browser (frontend/index.html)
   │
   ▼  (HTTPS via Cloudflare Tunnel or HTTP 127.0.0.1:8000)
FastAPI (backend/app/main.py)
   ├── GET  /                      -> FileResponse(frontend/index.html)
   ├── GET  /api/health            -> Dynamic runtime_state & checkpoint_exists
   ├── GET  /api/benchmark         -> Verified CSV metrics (backend/app/benchmarks.py)
   ├── POST /api/predict/fakeddit  -> FakedditClipPredictor (M3_clip_best.pt)
   ├── POST /api/predict/snlive    -> SnliBlipPredictor (blip-best.pt)
   └── POST /api/predict/faceforensics -> FaceForensicsPredictor (Xception + BiLSTM)
```

## 6. Technology Stack & Development Environment
- **OS / Runtime:** Windows, Python `3.13.2`
- **ML / Vision:** PyTorch `2.11.0+cpu`, Torchvision `0.26.0+cpu`, Transformers `5.24.1`, `timm 1.0.31`, OpenCV `4.11.0`, Pillow `11.1.0`, NumPy `2.2.3`
- **Backend / Testing:** FastAPI, Uvicorn, Pydantic Settings, `pytest 9.1.1`, `httpx`
- **Frontend:** Vanilla HTML5/CSS3/ES6+ (`frontend/index.html`, verified with `node --check`)

## 7. Dataset & Model Audit
- **Fakeddit (`AdoCleanCode/Fakeddit`):** Binary misinformation classification (`0 = FAKE / Non-true`, `1 = REAL / True`; `4,453` test samples). Checkpoints: `M3_clip_best.pt` (`572.92 MB`), `M1_text_best.pt` (`476.32 MB`), `M2_image_best.pt` (`106.95 MB`), `M4_vilt_best.pt` (`426.54 MB`).
- **SNLI-VE (`sedrickkeh/snli-ve`):** 3-class visual entailment (`0 = entailment`, `1 = neutral`, `2 = contradiction`; `6,000` test samples). Checkpoints: `blip-best.pt` (`145.02 MB`), `vilt-best.pt` (`111.20 MB`), `bridgetower-best.pt` (`150.28 MB`), `fusion-best.pt` (`1.77 MB`).
- **FaceForensics++ (`KoreaPeter/faceforensics-c23`):** Binary video manipulation (`0 = REAL`, `1 = FAKE`; custom `GroupShuffleSplit`). Checkpoints: `xception_bilstm_video_best.pt` (`18.02 MB`), `Xception_best.pt` (`79.68 MB`), `EfficientNet_B4_best.pt` (`67.68 MB`), `ResNet50_best.pt` (`90.00 MB`).

## 8. Primary Model Selection Rationale
Confirmed by the project team:
1. **`fk_m3_clip` (`M3_clip_best.pt`):** Highest test accuracy (`0.9007`), Macro F1 (`0.8991`), and ROC-AUC (`0.9621`) among saved Fakeddit checkpoints.
2. **`snli_blip` (`blip-best.pt`):** Highest SNLI-VE test accuracy (`0.7575`) and validation accuracy (`0.7650`).
3. **`ff_xception_bilstm` (`xception_bilstm_video_best.pt` + `Xception_best.pt`):** Video-level temporal model (`0.9163` accuracy, `0.9619` ROC-AUC on custom grouped split).

## 9. Individual Model Architectures & Inference Pipelines
- **Fakeddit `M3_clip` (`backend/app/inference/fakeddit_clip.py`):** `CLIPModel(CLIPConfig(vision_config={"patch_size": 16}))` (`ViT-B/16`), L2-normalized image/text embeddings combined into `[img, txt, img*txt, |img-txt|, cos_sim]` (`2049`-d) -> `Linear(2049, 256) -> LayerNorm(256) -> GELU -> Linear(256, 2)`. Loaded with `strict=True` (`404` keys), local `backend/models/clip_processor/`, temperature `1.2760057303751002`.
- **SNLI-VE `BLIP` (`backend/app/inference/snlive_blip.py`):** `BlipForImageTextRetrieval` (`Salesforce/blip-itm-base-coco`, revision `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`, cached locally at `1708.15 MB`) + `project` (`LayerNorm(768)->Linear(768,256)->GELU->LayerNorm(256)`) + `head` (`Dropout(0.15)->Linear(256,3)`). Loaded `112` trainable keys (`0` unexpected, `368` frozen base keys).
- **FaceForensics++ `Xception + BiLSTM` (`backend/app/inference/faceforensics_bilstm.py`):** `timm` `legacy_xception.tf_in1k` (`2048`-d) + `nn.LSTM(2048, 256, bidirectional=True, batch_first=True)` + `nn.Linear(512, 2)`. Samples up to `8` RGB frames via OpenCV (`cv2.VideoCapture`), applies `timm` transform (`299x299`), packs true sequence lengths, and includes a disclosure warning that upstream face-cropping could not be recovered.

## 10. Backend Architecture & API Implementation
Implemented in `backend/app/main.py`, `backend/app/config.py`, `backend/app/registry.py`, and `backend/app/benchmarks.py`:
- Per-model `threading.Lock()` double-checked singletons (`get_fakeddit_clip_predictor`, `get_snli_blip_predictor`, `get_faceforensics_predictor`).
- `GET /` serves `frontend/index.html` via `FileResponse` without exposing `backend/` or `models/`.

## 11. Frontend Integration & Model Registry
`frontend/index.html` was updated minimally in `PIPELINES` (`fk_m3_clip`, `snli_blip`, `ff_xception_bilstm`), `LEADERBOARD_DATA`, `handleUploadedFile` (immediate MP4 state assignment), `adaptApiResult` (`mismatchCategory`), and `#apiBaseInput` (`window.location.origin` default). All CSS, layout, and Neo-Brutalist UI components were preserved.

## 12. Prediction Workflows & Response Handling
Each `POST /api/predict/{fakeddit,snlive,faceforensics}` returns `analysis_id`, `prediction`, `scores`, `confidence`, `model_name`, `architecture`, `processing_time`, `abstained`, `abstention_reason`, and `warnings`.

## 13. Model Benchmarks & Evaluation Metrics
Served by `GET /api/benchmark` directly from original CSVs. **Note:** Each model was evaluated on a different dataset, task, and split protocol; these scores are **not** directly comparable across pipelines, and live prediction confidence is never used as benchmark accuracy.
- **Fakeddit `M3_clip` (`n = 4,453` test split):** `90.07%` Acc, `89.67%` Prec, `90.31%` Rec, `89.91%` F1, `96.21%` ROC-AUC.
- **SNLI-VE `BLIP` (`n = 6,000` test split, 3-class NLI):** `75.75%` Acc, `75.62%` Macro-F1, `90.28%` ROC-AUC.
- **FaceForensics++ `Xception + BiLSTM` (custom `GroupShuffleSplit` video-level):** `91.63%` Acc, `91.00%` Prec, `88.15%` Rec, `89.42%` Macro-F1, `96.19%` ROC-AUC.

## 14. Security, Validation & Error Handling
- **CORS:** Restricted to `http://127.0.0.1:5500,http://localhost:5500` by default (`allow_credentials=False` if `*` is configured).
- **Upload Limits:** `max_image_mb=15`, `max_video_mb=100` (`HTTP 413` on oversized uploads; `HTTP 415` on invalid MIME; `HTTP 422` on missing/corrupt inputs; `HTTP 503` on unimplemented secondary models).

## 15. Testing Methodology & Results
- **Backend `pytest` (`backend/tests/test_api.py`):** **13 passed, 1 warning** (harmless FastAPI `Form(model_id=...)` Pydantic warning).
- **Frontend JS (`node --check`):** **0 errors**.
- **Live Browser E2E (Phase 4):** All 3 primary models and `GET /api/benchmark` (`9` records) passed with **0 console errors**.

## 16. Single-Request Latency Measurements (Public Cloudflare Tunnel)
Measured via direct HTTPS `httpx` requests (not throughput benchmarks):
- **Successful `200 OK` Responses:**
  - `GET /`: `0.749s`
  - `GET /api/health`: `0.959s`
  - `GET /api/benchmark`: `0.886s`
  - `POST /api/predict/fakeddit`: `14.276s` *(cold start)*
  - `POST /api/predict/snlive`: `1.477s`
  - `POST /api/predict/faceforensics`: `3.498s`
- **Validation Check:** `POST /api/predict/fakeddit` (missing input) returned `422 Unprocessable Entity`.

## 17. Deployment Architecture
Single-origin local laptop deployment: `Browser -> Cloudflare Quick Tunnel (ephemeral HTTPS, not permanent cloud hosting) -> Local FastAPI (127.0.0.1:8000)`.

## 18. Limitations & Future Work
- Secondary models (`fk_m1_text`, `fk_m2_image`, `fk_m4_vilt`, `snli_vilt`, `snli_bridgetower`, `ff_efficientnet_b4`, `ff_xception`, `ff_resnet50`) remain unimplemented (`503`).
- FaceForensics++ uses full-frame uniform sampling because upstream face-cropping was not included in the original repo.
- Cloud/container deployment has not been tested; `trycloudflare.com` quick-tunnel URLs are ephemeral.
