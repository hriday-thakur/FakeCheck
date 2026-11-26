# FakeCheck — Phase 2: Fakeddit `M3_clip` Inference Plan

**Date:** 2026-11-26  
**Scope:** Read-only technical investigation for `fk_m3_clip` (`M3_clip_best.pt`).  
**Status:** Investigation complete; inference not yet implemented.

---

## 1. Recovered Architecture & Checkpoint Compatibility
- **Checkpoint:** `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M3_clip_best.pt` (`600,748,249` bytes / `572.92 MB`, `404` `state_dict` keys, raw `OrderedDict`).
- **Pretrained Backbone:** `CLIPModel` (`openai/clip-vit-base-patch32` dimensions: `clip.logit_scale`, `clip.text_model.embeddings.token_embedding.weight` `[49408, 512]`, `position_embedding.weight` `[77, 512]`, plus `clip.vision_model`).
- **Custom Head:** `head_proj.0.{weight,bias}`, `head_proj.1.{weight,bias}` (projecting concatenated/combined CLIP features to `256`-d) + `head.{weight,bias}` (`256 -> 2`).
- **Gap:** No `.py` or `.ipynb` source file exists in `fakeddit_leakage_controlled_run`, so the exact `forward()` combination of `image_embeds` and `text_embeds` (e.g., `[img, txt]` concatenation vs. elementwise interaction) and `head_proj` layer types must be verified or reconstructed.

---

## 2. Exact Preprocessing Pipeline
- **Text:** `CLIPProcessor` / `CLIPTokenizer` (`openai/clip-vit-base-patch32`, max `77` tokens). Note: `splits/test.csv` includes `text_norm` (lowercased, punctuation-stripped).
- **Image:** RGB `PIL.Image` processed by `CLIPProcessor` (224x224 resize/crop, CLIP mean/std normalization).
- **Calibration:** Temperature scaling `T = 1.2760057303751002` (`run_summary.json`).

---

## 3. Verified Label Mapping
- `0`: `"FAKE"` (Non-true / misinformation-related, `6_way_label` 1–5)
- `1`: `"REAL"` (True / REAL, `6_way_label` 0)

---

## 4. Dependency Requirements
- `torch 2.14.1+cu130` (or CPU build), `transformers 5.22.1`/`5.24.1`, `Pillow 11.3.0`.
- **Pretrained Weights / Processor:** Not stored locally; `openai/clip-vit-base-patch32` processor/tokenizer files must be downloaded or cached from HuggingFace Hub (the model weights themselves are inside `M3_clip_best.pt`).

---

## 5. Missing Artifacts & Blockers
1. No `.py`/`.ipynb` model definition (`forward()` and `head_proj` structure).
2. No local `config.json` or CLIP tokenizer/processor files.

---

## 6. Proposed Inference Implementation Steps
1. Create `backend/app/inference/fakeddit_clip.py` with the reconstructed `M3ClipClassifier(nn.Module)` matching all `404` keys in `M3_clip_best.pt`.
2. Load `M3_clip_best.pt` on CPU (`map_location="cpu", weights_only=False`) with `strict=True` to verify 100% key/shape compatibility.
3. Wire `POST /api/predict/fakeddit` in `backend/app/main.py` for `model_id == "fk_m3_clip"` and `mode == "multimodal"`, returning `adaptApiResult()`-compatible JSON (`prediction`, `scores: { REAL, FAKE }`, `confidence`).
4. Update `MODEL_REGISTRY["fk_m3_clip"].inference_ready = True` in `backend/app/registry.py` only after real inference passes.

---

## 7. Minimal CPU Inference Validation Plan
1. Instantiate `M3ClipClassifier` and load `M3_clip_best.pt` with `strict=True`.
2. Run a single forward pass on CPU with a sample RGB image + headline text.
3. Verify output logits shape `(1, 2)`, temperature-scaled softmax probabilities summing to `1.0`, and valid `REAL`/`FAKE` label mapping.
