# F4 Gated Feature Fusion — Integration Feasibility Report

**Date:** 2026-11-26  
**Scope:** Read-only recovery audit for `F4_Gated_Feature_Fusion_best.pt` (`M1_text`, `M2_image`, `M3_clip`, `M4_vilt`, `F4`).  
**Verdict:** **PARTIALLY READY** (All 5 checkpoints exist locally, but no `.py`/`.ipynb` training source exists in `fakeddit_leakage_controlled_run`, and `roberta-base` / `dandelin/vilt-b32-mlm` tokenizers/processors must be cached locally).

---

## 1. Original Architecture & Checkpoint Availability
- **Training Source Code:** **No `.py` or `.ipynb` files** exist in `models/text/original/fakeddit_leakage_controlled_run/`. Architectures are recovered from `state_dict` keys/shapes and `E1_baselines_vs_multimodal.csv`:
  1. `M1_text_best.pt` (`476.32 MB`, `205` keys): `roberta-base` + `proj` (`768 -> 256`) + `head` (`256 -> 2`).
  2. `M2_image_best.pt` (`106.95 MB`, `186` keys): `ConvNeXt-Tiny` + `head_proj` (`-> 256`) + `head` (`256 -> 2`).
  3. `M3_clip_best.pt` (`572.92 MB`, `404` keys): `CLIP ViT-B/16` + `head_proj` (`2049 -> 256`) + `head` (`256 -> 2`) — **already implemented**.
  4. `M4_vilt_best.pt` (`426.54 MB`, `212` keys): `ViLT` (`dandelin/vilt-b32-mlm`) + `head_proj` (`-> 256`) + `head` (`256 -> 2`).
  5. `F4_Gated_Feature_Fusion_best.pt` (`0.55 MB`, `24` keys): `projs.0..3` (`256 -> 128`) + `gate_net` (`512 -> 4`) + `classifier` (`128 -> 64 -> 2`).
- **256-d Embedding Extraction:** Each of `M1`–`M4` produces a `256`-d vector right before `head` (`proj` / `head_proj` output), matching `embed_dims: [256, 256, 256, 256]` in `F4_Gated_Feature_Fusion_best.pt` in exact order `['M1_text', 'M2_image', 'M3_clip', 'M4_vilt']`.

---

## 2. Missing Dependencies & Assets
- `roberta-base` tokenizer/config (`M1_text`) and `dandelin/vilt-b32-mlm` `ViltProcessor` (`M4_vilt`) are not in `backend/models/` (requires local caching).
- `M1_text`, `M2_image`, and `M4_vilt` PyTorch wrapper modules and `F4FusionPredictor` are not yet implemented in `backend/app/inference/`.

---

## 3. Benchmark Reproducibility (`E1_baselines_vs_multimodal.csv`, `n = 4,453`)
- `F4` (`0.9129` Acc, `0.9109` Macro F1, `0.9728` ROC-AUC) and `M3_clip` (`0.9007` Acc, `0.8991` Macro F1, `0.9621` ROC-AUC) were evaluated on the **same `n = 4,453` test split** (`run_summary.json` calibration temperature for `F4`: `0.9457072192113133`).
