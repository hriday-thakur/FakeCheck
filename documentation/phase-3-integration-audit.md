# Phase 3 — Frontend and Backend Integration Audit

**Date:** 2026-11-26  
**Scope:** Read-only audit of `frontend/index.html` against `backend/app/{main.py,registry.py,benchmarks.py}`.  
**Design Preservation:** All CSS, layout, Neo-Brutalist Light/Dark themes, typography, animations, and navigation remain untouched.

---

## 1. Frontend-to-Backend Pipeline & Model Mapping

| Frontend `pipe.id` | Current Frontend `PIPELINES` Model IDs | Backend Primary `model_id` | Accepted `mode` | Required Multipart Fields |
| :--- | :--- | :--- | :--- | :--- |
| `fakeddit` | `fk_concat_fusion`, `fk_weighted_late_fusion`, `fk_bert_base`, `fk_roberta_base`, `fk_distilbert`, `fk_deberta_v3`, `fk_resnet50`, `fk_efficientnet_b0`, `fk_vit_b16`, `fk_convnext_tiny` | **`fk_m3_clip`** | `multimodal` | `file` (`image/jpeg`, `image/png`), `text` |
| `snlive` | `snli_clip_interactions`, `snli_clip_concat` | **`snli_blip`** | `multimodal` | `file` (`image/jpeg`, `image/png`), `hypothesis` |
| `faceforensics` | `ff_two_model_ensemble`, `ff_ordered_lstm`, `ff_xception`, `ff_efficientnet_b4`, `ff_resnet50` | **`ff_xception_bilstm`** | `video-only` | `file` (`video/mp4`) |

---

## 2. `adaptApiResult()` & `GET /api/benchmark` Compatibility
- **`adaptApiResult()` (`frontend/index.html` lines 3197–3241):**
  - Consumes `prediction`, `scores`, `confidence`, `model_name`, `architecture`, `processing_time`, `warnings`, `abstained`, `abstention_reason`, and `analysis_id`.
  - **Exact Match:** All three backend `/api/predict/{fakeddit,snlive,faceforensics}` routes return these exact keys.
  - **Binary vs. NLI:** For `fakeddit` and `faceforensics`, `'REAL' in scores && 'FAKE' in scores` triggers `isBinary = true`. For `snlive`, `nli = true` maps `scores` (`entailment`, `neutral`, `contradiction`) to `nliProbs`.
- **`GET /api/benchmark` (`fetchBenchmarkData()`, lines 2663–2740):**
  - Backend returns `{ datasets: { fakeddit: [...], snlive: [...], faceforensics: [...] } }` with verified CSV metrics only (no live prediction confidence).
  - **Mismatch:** `LEADERBOARD_DATA` in `frontend/index.html` uses old IDs (`bert-base`, `concatenation-fusion`, etc.) instead of `fk_m3_clip`, `fk_m1_text`, `fk_m2_image`, `fk_m4_vilt`, `snli_blip`, `snli_vilt`, `snli_bridgetower`, `ff_xception_bilstm`, `ff_efficientnet_b4`, `ff_xception`, `ff_resnet50`.

---

## 3. Identified Mismatches & Risks
1. **Stale `PIPELINES` Model IDs:** Selecting any current frontend model ID sends an unknown `model_id` to the backend, resulting in `HTTP 400`.
2. **Unsupported Secondary Models:** Secondary models (`fk_m1_text`, `snli_vilt`, `ff_efficientnet_b4`, etc.) return `HTTP 503` if selected; the primary models (`fk_m3_clip`, `snli_blip`, `ff_xception_bilstm`) should be placed first as defaults.
3. **Error Handling:** `frontend/index.html` checks `if (!resp.ok) throw new Error('HTTP ' + resp.status);` and displays `showAnalyzeError(...)` for `400`, `415`, `422`, `500`, and `503`.
4. **Task Distinctions:** `legendHtml` in `PIPELINES` and backend `warnings` distinguish Fakeddit misinformation, SNLI-VE visual entailment, and FaceForensics++ video manipulation.

---

## 4. Minimal Implementation Plan (`frontend/index.html` Only)
1. Update `PIPELINES.fakeddit.models`, `PIPELINES.snlive.models`, and `PIPELINES.faceforensics.models` (lines 2230–2296) so the primary models (`fk_m3_clip`, `snli_blip`, `ff_xception_bilstm`) are listed first and secondary IDs match `backend/app/registry.py`.
2. Update `LEADERBOARD_DATA` (lines 2305–2340) IDs/names to match `GET /api/benchmark` (`fk_m3_clip`, `snli_blip`, `ff_xception_bilstm`, etc.).
3. Do **not** modify any CSS, HTML layout, or visual styling.
