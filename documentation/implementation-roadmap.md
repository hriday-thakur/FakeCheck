# FakeCheck — Implementation Roadmap

**Date:** 2026-11-26  
**Primary Production Models (Team-Confirmed):**
1. **Fakeddit:** `M3_clip_best.pt` (`CLIP Dual-Encoder`)
2. **SNLI-VE:** `blip-best.pt` (`BLIP`)
3. **FaceForensics++:** `xception_bilstm_video_best.pt` + `Xception_best.pt` (`Xception + BiLSTM`)

---

## Phase 1: FastAPI Backend Foundation
- Scaffold `backend/` with FastAPI, Uvicorn, `CORSMiddleware`, and `GET /api/benchmark` (serving verified metrics from `models/**/metrics/`).
- Implement `POST /api/predict/{fakeddit,snlive,faceforensics}` request validation, `FormData` parsing, and `adaptApiResult()`-compatible JSON responses.

## Phase 2: Fakeddit `M3_clip` Inference (**COMPLETED — Verified CPU & HTTP Inference**)
- **Exact Checkpoint:** `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M3_clip_best.pt` (`572.92 MB`, `404` keys).
- **Architecture:** Reconstructed CLIP ViT-B/16 Dual-Encoder (`CLIPConfig(vision_config={"patch_size": 16})` + `head_proj` `Linear(2049, 256) -> LayerNorm(256)` + `head` `Linear(256, 2)`) in `backend/app/inference/fakeddit_clip.py`, loaded with `strict=True`.
- **Required Preprocessing:** Paired RGB image + text (max 77 tokens) via local `backend/models/clip_processor/` (`openai/clip-vit-base-patch16`, `local_files_only=True`); calibration temperature `1.2760057303751002`.
- **Validation:** `strict=True` load (404 keys), CPU forward pass, and real HTTP `POST /api/predict/fakeddit` (`200 OK`) all passed. `fk_m3_clip` is marked `inference_ready=True`.

## Phase 3: SNLI-VE `BLIP` Inference (**COMPLETED — Verified CPU & API Inference**)
- **Exact Checkpoint:** `models/image/original/SNLI_VE_COMPLETE_PROJECT/blip-best.pt` (`145.02 MB`, `112` trainable keys).
- **Architecture:** `BlipBranch` in `backend/app/inference/snlive_blip.py` wrapping `Salesforce/blip-itm-base-coco` (revision `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`, `local_files_only=True`).
- **Validation:** `112` trainable keys loaded (`0` unexpected, `368` frozen base keys), CPU forward pass (`0.8822s`), and `pytest` (`7 passed`) all succeeded. `snli_blip` is marked `inference_ready=True`.

## Phase 4: FaceForensics++ `Xception + BiLSTM` Inference (**COMPLETED — Verified CPU & API Inference**)
- **Exact Checkpoints:** `models/video/original/FFPP_PJ07_FINAL_corrected/checkpoints/xception_bilstm_video_best.pt` (`18.02 MB`, `10` keys) + `Xception_best.pt` (`79.68 MB`, `276` keys).
- **Architecture:** `timm` `legacy_xception.tf_in1k` (2048-d) + 1-layer bidirectional `nn.LSTM(2048, 256, bidirectional=True, batch_first=True)` + `nn.Linear(512, 2)` in `backend/app/inference/faceforensics_bilstm.py`.
- **Validation:** `strict=True` loading, `timm==1.0.31`, OpenCV MP4 sampling (`<=8` frames), packed sequence lengths, and `pytest` (`8 passed`) all succeeded. `ff_xception_bilstm` is marked `inference_ready=True`.

## Phase 5: Frontend Integration (**COMPLETED — Minimal Registry Alignment**)
- Updated `PIPELINES` and `LEADERBOARD_DATA` in `frontend/index.html` to use `fk_m3_clip`, `snli_blip`, and `ff_xception_bilstm` as primary models.
- Preserved all CSS, layout, typography, and Neo-Brutalist UI components untouched.

## Phase 6: End-to-End Testing & Deployment (**COMPLETED — Hardened & Verified**)
- Verified `GET /api/benchmark` (`9` records synced across Fakeddit, SNLI-VE, and FaceForensics++).
- Verified `POST /api/predict/{fakeddit,snlive,faceforensics}` end-to-end from `frontend/index.html` (`0` browser console errors).
- Phase 5B hardening: Added `max_video_mb: 100` (`413` enforcement) and per-model `threading.Lock()` singleton initialization.
- Phase 5C hardening: Restricted default CORS (`http://127.0.0.1:5500,http://localhost:5500`), added dynamic `runtime_state` in `GET /api/health`, created `README.md` and `documentation/deployment.md`; `pytest` (**12 passed**), `node --check` (**0 errors**).
