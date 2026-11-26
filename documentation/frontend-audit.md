# FakeCheck — Frontend Technical Audit Report

**Audit Date:** 2026-11-26  
**Scope:** Read-only technical audit of `frontend/index.html` (`4,011` lines, `149,914` bytes).  
**Design Preservation:** The existing Neo-Brutalist Light/Dark interface in `frontend/index.html` is the definitive design reference and was not modified.

---

## 1. Framework, Language, Build Tools & Dependencies
- **Architecture:** Single-file static web application (`frontend/index.html`) — **no** React, Next.js, Vite, `package.json`, or `node_modules`.
- **Languages:** Vanilla HTML5, CSS3 (custom properties, Neo-Brutalist Light/Dark theme), and Vanilla ES6+ JavaScript (wrapped in an IIFE).
- **External Dependencies:** Google Fonts only (`Archivo Black`, `Playfair Display`, `Space Mono` via `fonts.googleapis.com`).
- **Build Tools:** None required; served directly as a static HTML file.

---

## 2. Complete Folder & File Structure
```text
frontend/
└── index.html   (149,914 bytes, 4,011 lines)
```

---

## 3. Pages, Routes & Navigation
Client-side view switching via `navigateTo(viewName)` (toggling `.view` sections):
- **Home / Hero & Overview** (`<section class="hero">`, `#home-pipelines`)
- **Analyze** (`#view-analyze`): Pipeline selector (`fakeddit`, `faceforensics`, `snlive`), model selector, mode toggles, file dropzone, text/hypothesis input, calibration/threshold controls, and terminal step log.
- **Result** (`#view-result`): Unified prediction result & XAI view (`renderResultView(res)`).
- **Unified** (`#view-unified`): Unified cross-pipeline view (`renderUnified()`).
- **Dashboard** (`#view-dashboard`): Summary analytics (`renderDashboard()`).
- **Models / Leaderboard** (`#view-models`): Model registry & benchmark table (`renderLeaderboard()`).
- **Datasets** (`#view-datasets`): Dataset overview cards.
- **History** (`#view-history`): Saved analysis history (`renderHistory()`).
- **API Settings Modal** (`#apiModal`): Configure backend base URL (`#apiBaseInput`).

---

## 4. Reusable Components & UI Patterns
Implemented as DOM templates and render functions inside `frontend/index.html`:
- Neo-Brutalist cards, badges, toast notifications (`showToast`), terminal progress log (`#terminalSteps`), SVG sample generators (`buildSampleSvg`), and HTML5 `<canvas>` sample video animation (`startSampleVideoCanvas`).

---

## 5. Existing Upload Workflows (Text, Images & Videos)
- **Fakeddit (`fakeddit`):** Accepts `image/jpeg,image/png,image/jpg` (max 15 MB) and/or text (`text` field in `FormData`), supporting `multimodal`, `text-only`, and `image-only` modes.
- **FaceForensics++ (`faceforensics`):** Accepts `video/mp4` only (`video-only` mode; text input is explicitly disabled).
- **SNLI-VE (`snlive`):** Accepts `image/jpeg,image/png,image/jpg` + required `hypothesis` text (`multimodal` mode, plus `mismatch_heuristic` checkbox).

---

## 6. Existing Analysis & Prediction Result Components
- `adaptApiResult(api, pipe, model, text, elapsed, xaiOn)` normalizes backend JSON into the internal result object.
- `renderResultView(res)` renders the verdict badge, class probabilities, Shannon entropy, calibration temperature, and XAI visualizations.

---

## 7. Dashboard, Model Information & History
- **Leaderboard (`LEADERBOARD_DATA`):** Initialized with `createEmptyMetrics()` (`null` values, status `'Unavailable'`) — **zero fabricated scores**. Populated dynamically if `GET /api/benchmark` succeeds.
- **History (`historyList`):** Persisted in `localStorage` under `fakecheck_history_v1`.

---

## 8. Existing API Calls, Mock Data & Placeholder Functions
- **No mock predictions:** When the backend is unreachable, `runAnalysisBtn` catches the error and displays an explicit error message (*"No prediction was produced"*).
- **Existing `fetch()` calls:**
  1. `GET ${baseUrl}/api/benchmark` (`fetchBenchmarkData()`, line 2681)
  2. `POST ${base}/api/predict/${pipe.id}` (`runAnalysisBtn` handler, line 3288)
- **Sample Loaders:** `buildSampleSvg()` and `startSampleVideoCanvas()` provide local sample previews for testing the UI.

---

## 9. State Management & Data Flow
- In-memory state (`currentResult`, `historyList`, `uploadedMedia`, `activeMode`, `isRunning`) + `localStorage` persistence:
  - `fakecheck_history_v1`
  - `fakecheck_theme_v1`
  - `fakecheck_api_base_v1`

---

## 10. Environment Variables & Backend URL Configuration
- No `.env` file. Backend URL is configured in the UI via `#apiBaseInput` inside `#apiModal` and persisted in `localStorage` (`fakecheck_api_base_v1`).

---

## 11. Loading, Error, Empty & Success States
- **Loading:** Disables `#runAnalysisBtn`, shows `#terminalSteps` with upload/waiting messages.
- **Error:** `showAnalyzeError(msg)` displays inline error banners on validation failure or HTTP/network errors.
- **Empty:** Leaderboard shows `Unavailable` / `—` until `/api/benchmark` responds; History shows empty state when `historyList` is empty.
- **Success:** Saves result to `historyList`, calls `renderResultView(result)`, and navigates to `result`.

---

## 12. Responsive Behavior & Design Conventions
- Neo-Brutalist Light (`#f1ede4` bg, `#0d0d0d` ink) and Dark (`#0d0d0d` bg, `#f1ede4` ink) themes with `--acid` (`#d8ff00`), `--red` (`#ff2b2b`), `--cyan` (`#00e5ff`), `--green` (`#7dff7d`), `4px` borders, and hard offset shadows.
- Responsive viewport meta tag and CSS media queries.

---

## Model Integration Analysis & Proposed Backend API Contracts

### Critical Mismatch Between Frontend `PIPELINES` and Actual Trained Checkpoints
| Pipeline | Frontend `PIPELINES` Model IDs | Actual Trained Checkpoints in `models/` |
| :--- | :--- | :--- |
| **Fakeddit** | `fk_concat_fusion`, `fk_weighted_late_fusion`, `fk_bert_base`, `fk_roberta_base`, `fk_distilbert`, `fk_deberta_v3`, `fk_resnet50`, `fk_efficientnet_b0`, `fk_vit_b16`, `fk_convnext_tiny` | `M1_text_best.pt` (`roberta-base`), `M2_image_best.pt` (`ConvNeXt-Tiny`), `M3_clip_best.pt` (`CLIP Dual-Encoder`), `M4_vilt_best.pt` (`ViLT`). *(No `F4` fusion checkpoint.)* |
| **SNLI-VE** | `snli_clip_interactions`, `snli_clip_concat` | `blip-best.pt` (winner), `vilt-best.pt`, `bridgetower-best.pt`, `fusion-best.pt`. |
| **FaceForensics++** | `ff_two_model_ensemble`, `ff_ordered_lstm`, `ff_xception`, `ff_efficientnet_b4`, `ff_resnet50` | `xception_bilstm_video_best.pt`, `Xception_best.pt`, `EfficientNet_B4_best.pt`, `ResNet50_best.pt`. |

### Proposed Backend API Contracts (Matching `frontend/index.html`)
1. **`GET /api/benchmark`**: Returns benchmark metrics per pipeline/model to populate `LEADERBOARD_DATA`.
2. **`POST /api/predict/fakeddit`** (`multipart/form-data`: `text`, `file`, `model_id`, `mode`, `temperature`, `confidence_threshold`, `entropy_threshold`, `xai`):
   - `text-only` -> `M1_text_best.pt` (RoBERTa-base)
   - `image-only` -> `M2_image_best.pt` (ConvNeXt-Tiny)
   - `multimodal` -> `M3_clip_best.pt` (CLIP) or `M4_vilt_best.pt` (ViLT)
   - Returns `{ prediction, probabilities, confidence, entropy, ... }` (labels: `0 = Non-true / misinformation-related`, `1 = True / REAL`).
3. **`POST /api/predict/snlive`** (`multipart/form-data`: `file`, `hypothesis`, `model_id`, `temperature`, `confidence_threshold`, `entropy_threshold`, `mismatch_heuristic`, `xai`):
   - Uses `blip-best.pt` (or `vilt-best.pt` / `bridgetower-best.pt`).
   - Returns `{ prediction, probabilities, ... }` (labels: `0 = entailment`, `1 = neutral`, `2 = contradiction`; **not** a deepfake label).
4. **`POST /api/predict/faceforensics`** (`multipart/form-data`: `file` MP4, `model_id`, `temperature`, `confidence_threshold`, `entropy_threshold`, `xai`):
   - Uses `xception_bilstm_video_best.pt` + `Xception_best.pt` (or frame-level models).
   - Returns `{ prediction, probabilities, ... }` (labels: `0 = REAL`, `1 = FAKE`).

### Files Needing Modification for Integration
- `frontend/index.html`: Only the `PIPELINES` model IDs/names (lines 2230–2296) and `LEADERBOARD_DATA` entries (lines 2305+) should be updated to match the actual trained checkpoints (`M1_text`, `M2_image`, `M3_clip`, `M4_vilt`, `blip-best.pt`, `xception_bilstm_video_best.pt`). **Do not alter CSS, layout, or Neo-Brutalist design.**
