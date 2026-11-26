# Phase 2B — SNLI-VE BLIP (`snli_blip`) Inference Readiness Audit

**Date:** 2026-11-26  
**Scope:** Read-only audit of `models/image/original/SNLI_VE_COMPLETE_PROJECT/blip-best.pt` and `executed-notebook-code.py`.  
**Verdict:** **PARTIALLY READY** (architecture and preprocessing are fully recovered, but `Salesforce/blip-itm-base-coco` base weights & processor are not cached locally).

---

## 1. Checkpoint Inspection Findings
- **Exact Path & Size:** `models/image/original/SNLI_VE_COMPLETE_PROJECT/blip-best.pt` (`152,065,355` bytes / `145.02 MB`).
- **Structure & Parameter Count:** Raw `state_dict` (`OrderedDict`) with **`112` keys** (no `module.` or `model.` wrapper prefixes).
- **Representative Keys & Shapes:**
  - `base.text_encoder.encoder.layer.8.attention.self.query.weight`: `(768, 768)`
  - `project.1.bias`: `(256,)`, `project.3.weight`: `(256,)`
  - `head.1.weight`: `(3, 256)`, `head.1.bias`: `(3,)`
- **Completeness & Strict Loading:** `blip-best.pt` contains **only** the trainable parameters (`base.text_encoder.encoder.layer[-4:]`, `project.*`, `head.*`). **`strict=True` is NOT feasible**; `restore()` in `executed-notebook-code.py` (lines 225–228) loads with `strict=False` and asserts `not extra` and all `requires_grad` parameters are present.

---

## 2. Recovered Architecture (`Branch("blip")`, `executed-notebook-code.py` lines 179–208)
- **Base Model:** `BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco", revision="bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c")`.
- **Trainable/Frozen Split:** `base.requires_grad_(False)`, then `base.text_encoder.encoder.layer[-4:]` unfrozen.
- **Forward Pass:**
  1. `vision = self.base.vision_model(pixel_values=x["pixel_values"]).last_hidden_state`
  2. `mask = torch.ones(vision.shape[:2], device=vision.device, dtype=torch.long)`
  3. `raw = self.base.text_encoder(input_ids=x["input_ids"], attention_mask=x["attention_mask"], encoder_hidden_states=vision, encoder_attention_mask=mask, return_dict=True).last_hidden_state[:, 0]`
  4. `z = self.project(raw)` (`LayerNorm(768) -> Linear(768, 256) -> GELU -> LayerNorm(256)`)
  5. `logits = self.head(z)` (`Dropout(0.15) -> Linear(256, 3)`), returning `(logits, z)`.

---

## 3. Original Preprocessing & Inference Pipeline (`executed-notebook-code.py` lines 167–177)
- **Processor:** `BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco", revision="bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c")`.
- **Image:** `PIL.Image.open(...).convert("RGB")`.
- **Tokenization & Collation:** `proc(text=texts, images=pics, padding=True, truncation=True, max_length=64, return_tensors="pt")`.
- **Calibration / Postprocessing:** No temperature scaling; probabilities via `logits.float().softmax(-1)`.

---

## 4. Label Mapping Evidence (`complete-experiment-info.json` & `executed-notebook-code.py` line 31)
- `0`: `"entailment"`
- `1`: `"neutral"`
- `2`: `"contradiction"`
- *Note:* Visual Entailment describes image-text consistency, **not** deepfake manipulation.

---

## 5. Base Model Asset Availability
- **Local HF Cache Check:** `HF cache matches: []` (`~/.cache/huggingface/hub/` has no `blip` repo).
- **Local Project Directory:** No `config.json`, `preprocessor_config.json`, `tokenizer.json`, or base weights exist locally.
- **Impact:** `local_files_only=True` **cannot** run yet; network access is required once to download `Salesforce/blip-itm-base-coco` (~900 MB–1 GB).

---

## 6. Dependency Compatibility
- Current backend environment: Python `3.13.2`, PyTorch `2.11.0+cpu`, torchvision `0.26.0+cpu`, Transformers `5.24.1`, Pillow `11.3.0`.
- Original training used `transformers==4.49.0`. Compatibility of `BlipForImageTextRetrieval` and `BlipProcessor` with `transformers 5.24.1` is **untested** until base model download.

---

## 7. Inference Feasibility & 8. Backend Integration Requirements
- **Feasibility:** Architecture is 100% recoverable without modifying trained weights. CPU inference is feasible (~1 GB RAM).
- **Integration:** Wire `POST /api/predict/snlive` (`model_id="snli_blip"`, `file`, `hypothesis`) to a lazy-loaded `SnliBlipPredictor` singleton returning `{ prediction, scores: { entailment, neutral, contradiction }, confidence, ... }`. Keep `executed-notebook-code.py` unimported (due to Kaggle paths and `.cuda()` assertions).

---

## 9. Missing Information & Unresolved Risks
1. **Missing Base Weights:** `Salesforce/blip-itm-base-coco` (~900 MB–1 GB) is not cached locally.
2. **Transformers 5.24.1 Compatibility:** Needs verification once base weights are cached.

## 10. Readiness Verdict
**READY** — `SnliBlipPredictor` is implemented in `backend/app/inference/snlive_blip.py`, wired to `POST /api/predict/snlive`, and verified with CPU forward pass (`0.8822s`) and `pytest` (`7 passed`). `snli_blip` is marked `inference_ready=True`.

---

## 11. Base-Weight Caching & Compatibility Findings (Stage 2)
- **Download Status:** Completed (`BlipProcessor` + `BlipForImageTextRetrieval`, `472` base tensors).
- **Exact Revision Verified:** `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`.
- **Cache Location:** `C:\Users\Neji Hyuga\.cache\huggingface\hub\models--Salesforce--blip-itm-base-coco`.
- **Total Disk Usage:** `1708.15 MB` (~1.67 GB).
- **Offline Loading (`local_files_only=True`):** Both `BlipProcessor` and `BlipForImageTextRetrieval` load cleanly without network access.
- **Transformers `5.24.1` Compatibility:** `BlipProcessor` and `BlipForImageTextRetrieval` instantiate and load all `472` base weights cleanly; `blip-best.pt` remains unchanged (`112` keys).
- **Remaining Blocker:** `SnliBlipPredictor` has not yet been implemented in `backend/app/inference/` or wired to `POST /api/predict/snlive`.
