# New Fakeddit Model Compatibility Audit (`F4_Gated_Feature_Fusion_best.pt`)

**Date:** 2026-11-26  
**Scope:** Read-only compatibility audit of `models/text/original/fakeddit_leakage_controlled_run/checkpoints/F4_Gated_Feature_Fusion_best.pt`.  
**Verdict:** **COMPATIBLE WITH CHANGES** (The checkpoint is complete and clean, but requires implementing `M1_text`, `M2_image`, and `M4_vilt` feature extractors + caching their tokenizers/processors before it can run live inference).

---

## 1. Checkpoint Inspection
- **Path & Size:** `models/text/original/fakeddit_leakage_controlled_run/checkpoints/F4_Gated_Feature_Fusion_best.pt` (`581,585` bytes / `0.55 MB`).
- **Structure:** Nested `dict` with `11` top-level keys:
  - `model_name`: `"F4 Gated Feature Fusion"`
  - `model_keys`: `['M1_text', 'M2_image', 'M3_clip', 'M4_vilt']`
  - `embed_dims`: `[256, 256, 256, 256]`
  - `proj_dim`: `128`, `mod_drop_p`: `0.2`, `refit_epochs`: `4`, `seed`: `42`
  - `training_protocol`: `"5-fold validation CV; final refit on all validation embeddings; held-out test embeddings used only for sanity check"`
  - `state_dict`: `24` tensors.

---

## 2. Architecture Findings
`F4_Gated_Feature_Fusion_best.pt` is a **feature-level gated fusion head** (not a standalone end-to-end model):
- `projs.0..3`: 4 modality projection branches (`Linear(256, 128) -> LayerNorm(128)`).
- `gate_net`: `Linear(512, 4)` (takes concatenated `4 * 128 = 512`-d projected embeddings and outputs 4 modality gates).
- `classifier`: `Linear(128, 64) -> LayerNorm(64) -> [activation/dropout] -> Linear(64, 2)`.

---

## 3. Comparison with `M3_clip_best.pt` & Dependency Requirements
- `M3_clip_best.pt` (`572.92 MB`) is a standalone CLIP ViT-B/16 dual-encoder.
- `F4_Gated_Feature_Fusion_best.pt` (`0.55 MB`) **depends on all four** Fakeddit models (`M1_text_best.pt` `476.32 MB`, `M2_image_best.pt` `106.95 MB`, `M3_clip_best.pt` `572.92 MB`, `M4_vilt_best.pt` `426.54 MB` — **~1.58 GB** total) to extract the four `256`-d embeddings at inference time.
- Requires `roberta-base` (`M1_text`), `ConvNeXt-Tiny` (`M2_image`), `openai/clip-vit-base-patch16` (`M3_clip`, already cached), and `dandelin/vilt-b32-mlm` (`M4_vilt`) tokenizers/processors.

---

## 4. Required Integration Changes
1. Implement `M1_text`, `M2_image`, and `M4_vilt` 256-d embedding extractors in `backend/app/inference/` (currently only `M3_clip` is implemented).
2. Cache `roberta-base` and `dandelin/vilt-b32-mlm` tokenizers/processors locally.
3. Add `F4FusionPredictor` combining all 4 extractors + `F4_Gated_Feature_Fusion_best.pt` (with calibration temperature `0.9457072192113133` from `run_summary.json`).

---

## 5. Benchmark Comparison (`E1_baselines_vs_multimodal.csv`, `n = 4,453` test split)
- **`F4 Gated Feature Fusion`:** Accuracy `0.9129`, Macro F1 `0.9109`, ROC-AUC `0.9728`, PR-AUC `0.9623`.
- **`M3_clip` (Current Primary):** Accuracy `0.9007`, Macro F1 `0.8991`, ROC-AUC `0.9621`, PR-AUC `0.9410`.
