# FakeCheck — Backend API Contract

**Date:** 2026-11-26  
**Scope:** Exact frontend-to-backend API contract extracted from `frontend/index.html`, mapped to the verified checkpoints in `documentation/model-audit.md` and `documentation/inference-readiness.md`.  
**Design Preservation:** The existing Neo-Brutalist UI in `frontend/index.html` is preserved; only `PIPELINES` and `LEADERBOARD_DATA` registries will need ID alignment during integration.

---

## 1. Exact Existing Frontend API Contract

### 1.1 `GET /api/benchmark` (`fetchBenchmarkData()`, lines 2663–2740)
- **Method & URL:** `GET ${baseUrl}/api/benchmark`
- **Headers:** `Accept: application/json` (8-second `AbortController` timeout).
- **Expected Response JSON:**
  Supports `{ datasets: { fakeddit: [...], faceforensics: [...], snlive: [...] } }`, top-level `{ fakeddit: [...], ... }`, or a flat array `[...]`.
  Each item is matched against `LEADERBOARD_DATA[dsKey]` by `id` / `model_id` / `config_id` / `name` and consumes:
  - `status`: `'Trained' | 'Evaluated' | 'Planned' | 'Unavailable'`
  - `metrics` (or top-level): `accuracy` (`acc`), `precision` (`prec`), `recall` (`rec`), `macro_f1` (`macroF1`, `f1`), `roc_auc` (`rocAuc`, `auc`).

### 1.2 `POST /api/predict/{pipeline_id}` (lines 3267–3304)
- **Method & URL:** `POST ${base}/api/predict/${pipe.id}` where `pipe.id` ∈ `['fakeddit', 'faceforensics', 'snlive']`.
- **Content-Type:** `multipart/form-data` (`FormData`).
- **FormData Fields Sent by Frontend:**
  - `file` *(optional/required by mode)*: Uploaded `File` object (`JPG`/`JPEG`/`PNG` for `fakeddit`/`snlive`; `MP4` for `faceforensics`). Omitted when `mode === 'text-only'`.
  - `text` *(for `fakeddit`)* or `hypothesis` *(for `snlive`)*: Trimmed text string. Omitted for `faceforensics` or `image-only`.
  - `model_id` *(required)*: Selected model ID from `PIPELINES[pipe.id].models`.
  - `mode` *(required)*: `'multimodal' | 'text-only' | 'image-only' | 'video-only'`.
  - `xai` *(required)*: `'true' | 'false'`.
  - `temperature` *(required)*: Stringified float from `#cfgTemp`.
  - `confidence_threshold` *(required)*: Stringified float from `#cfgConf`.
  - `entropy_threshold` *(required)*: Stringified float from `#cfgEnt`.
  - `mismatch_heuristic` *(only for `snlive`)*: `'true' | 'false'`.
- **Expected Response JSON (`adaptApiResult()`, lines 3197–3241):**
  - `prediction` *(required)*: String label (e.g., `'REAL'`, `'FAKE'`, `'ENTAILMENT'`, `'NEUTRAL'`, `'CONTRADICTION'`). If `api.prediction == null`, the frontend throws `response has no "prediction" field`.
  - `scores` *(object)*: Class probabilities. For binary pipelines (`fakeddit`, `faceforensics`), `adaptApiResult` checks `'REAL' in scores && 'FAKE' in scores`. For `snlive`, `scores` maps NLI class names to probabilities.
  - `confidence` *(number or `{ value: number }`)*.
  - `analysis_id` *(string, optional)*, `model_name` *(string, optional)*, `architecture` *(string, optional)*, `manipulation_family` *(string, optional)*.
  - `abstained` *(boolean, optional)*, `abstention_reason` *(string, optional)*.
  - `modalities` *(object, optional)*, `warnings` *(string[], optional)*, `summary` *(string, optional)*, `processing_time` *(number, optional)*.
  - `evidence.text_attributions` *(array, optional, used when `xai === 'true'`)*.
- **Error Handling:** Any non-2xx HTTP status (`!resp.ok`) or missing `prediction` field triggers `showAnalyzeError(...)` with no simulated fallback.

---

## 2. Proposed FastAPI Backend Architecture

- **Framework:** FastAPI + Uvicorn, with `CORSMiddleware` enabled for local static serving of `frontend/index.html`.
- **Routers:**
  - `GET /api/benchmark` — returns verified evaluation metrics from `models/**/metrics/` and `models/**/audit/` (zero fabricated numbers).
  - `POST /api/predict/fakeddit` — routes by `mode` / `model_id` to `M1_text`, `M2_image`, `M3_clip`, or `M4_vilt`.
  - `POST /api/predict/snlive` — routes to `BLIP` (`blip-best.pt`), `ViLT`, or `BridgeTower`.
  - `POST /api/predict/faceforensics` — routes to `Xception + BiLSTM` (`xception_bilstm_video_best.pt` + `Xception_best.pt`) or frame-level models.

---

## 3. Pipeline-to-Model Mapping

Default/primary production models are marked **(Primary)**; secondary models are retained for future benchmarking.

| Frontend `pipe.id` | `model_id` | Actual Checkpoint | Task & Label Mapping |
| :--- | :--- | :--- | :--- |
| `fakeddit` | `fk_m3_clip` **(Primary / Default)** | `M3_clip_best.pt` (`CLIP Dual-Encoder`) | Misinformation (`0 = FAKE / Non-true`, `1 = REAL / True`) |
| `fakeddit` | `fk_m1_text` *(Secondary)* | `M1_text_best.pt` (`roberta-base`) | Misinformation (`0 = FAKE / Non-true`, `1 = REAL / True`) |
| `fakeddit` | `fk_m2_image` *(Secondary)* | `M2_image_best.pt` (`ConvNeXt-Tiny`) | Misinformation (`0 = FAKE / Non-true`, `1 = REAL / True`) |
| `fakeddit` | `fk_m4_vilt` *(Secondary)* | `M4_vilt_best.pt` (`ViLT`) | Misinformation (`0 = FAKE / Non-true`, `1 = REAL / True`) |
| `snlive` | `snli_blip` **(Primary / Default)** | `blip-best.pt` (`Salesforce/blip-itm-base-coco`) | Visual Entailment (`entailment`, `neutral`, `contradiction`) |
| `snlive` | `snli_vilt` *(Secondary)* | `vilt-best.pt` (`dandelin/vilt-b32-mlm`) | Visual Entailment (`entailment`, `neutral`, `contradiction`) |
| `snlive` | `snli_bridgetower` *(Secondary)* | `bridgetower-best.pt` (`BridgeTower/bridgetower-base`) | Visual Entailment (`entailment`, `neutral`, `contradiction`) |
| `faceforensics` | `ff_xception_bilstm` **(Primary / Default)** | `xception_bilstm_video_best.pt` + `Xception_best.pt` | Video Manipulation (`0 = REAL`, `1 = FAKE`) |
| `faceforensics` | `ff_efficientnet_b4` *(Secondary)* | `EfficientNet_B4_best.pt` | Frame Manipulation (`0 = REAL`, `1 = FAKE`) |
| `faceforensics` | `ff_xception` *(Secondary)* | `Xception_best.pt` | Frame Manipulation (`0 = REAL`, `1 = FAKE`) |
| `faceforensics` | `ff_resnet50` *(Secondary)* | `ResNet50_best.pt` | Frame Manipulation (`0 = REAL`, `1 = FAKE`) |

---

## 4. Request & Response Examples

### `POST /api/predict/fakeddit` (Response Shape)
```json
{
  "analysis_id": "AN-FAKEDDIT-001",
  "prediction": "FAKE",
  "scores": { "REAL": 0.14, "FAKE": 0.86 },
  "confidence": 0.86,
  "model_name": "M1_text (roberta-base)",
  "architecture": "RoBERTa-base + 256-d Projection",
  "abstained": false,
  "abstention_reason": "",
  "warnings": [],
  "summary": "Classified as Non-true / misinformation-related.",
  "processing_time": 0.18
}
```

### `POST /api/predict/snlive` (Response Shape)
```json
{
  "analysis_id": "AN-SNLIVE-001",
  "prediction": "CONTRADICTION",
  "scores": { "entailment": 0.08, "neutral": 0.15, "contradiction": 0.77 },
  "confidence": 0.77,
  "model_name": "BLIP (blip-best.pt)",
  "architecture": "Salesforce/blip-itm-base-coco + 256-d Projection",
  "abstained": false,
  "warnings": ["SNLI-VE predicts visual entailment, not deepfake manipulation."],
  "processing_time": 0.42
}
```

---

## 5. Validation & Error Handling
- Reject missing required `file` or `text`/`hypothesis` for the selected `mode` with `HTTP 422` / `HTTP 400` and `{ "detail": "..." }`.
- Enforce file type (`JPG`/`PNG` for images, `MP4` for video) and size limits (15 MB for images).
- Return `HTTP 503` if a requested model's HuggingFace base weights are not cached and cannot be downloaded.

---

## 6. Model Loading Strategy
- Lazy-load or startup-load PyTorch models on `cpu` (or `cuda` if available) using `torch.load(..., map_location=device, weights_only=False)`.
- Never import `executed-notebook-code.py` directly (due to Kaggle paths and `assert torch.cuda.device_count() >= 1`); extract clean, device-agnostic PyTorch module definitions into `backend/`.

---

## 7. Remaining Inference Blockers
1. **Fakeddit (`M1_text`–`M4_vilt`):** Wrapper classes (`encoder`/`clip` + `proj`/`head_proj` + `head`) and HuggingFace tokenizers/configs must be implemented/cached.
2. **SNLI-VE (`BLIP`):** `Salesforce/blip-itm-base-coco` base weights must be cached locally or downloaded at startup.
3. **FaceForensics++ (`Xception + BiLSTM`):** BiLSTM wrapper class, MP4 frame sampler (`temporal_frames: 8`), and face-cropper must be implemented in `backend/`.

---

## 8. Frontend Files Requiring Modification
- `frontend/index.html`: Update only the `PIPELINES` model registry (lines 2230–2296) and `LEADERBOARD_DATA` IDs (lines 2305+) to match the actual trained checkpoints listed in Section 3. All CSS, layout, and Neo-Brutalist UI components remain unchanged.
