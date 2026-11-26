# F4 Phase 1.7 — Feature Extractor Recovery & Asset Readiness Report

**Date:** 2026-11-27  
**Verdict:** **PARTIALLY READY** (`M1`–`M4` backbones are identified from `state_dict` shapes and `E1_baselines_vs_multimodal.csv`, but `roberta-base` and `dandelin/vilt-b32-mlm` tokenizers/processors are not cached locally).

---

## 1. Implementation Readiness Table

| Component | Architecture | Preprocessing | Assets | 256-d output verified |
| :--- | :--- | :--- | :--- | :--- |
| **M1 (`M1_text`)** | `roberta-base` + `proj` (`768->256`) + `head` | `text_norm` (CSV); exact pooling unrecovered | `M1_text_best.pt` local; `roberta-base` tokenizer **missing** | Yes (in `M1_text_test_embeddings.npy`) |
| **M2 (`M2_image`)** | `ConvNeXt-Tiny` + `head_proj` (`768->256`) + `head` | RGB image; exact transform unrecovered | `M2_image_best.pt` local; `timm` `convnext_tiny` | Yes (in `M2_image_test_embeddings.npy`) |
| **M3 (`M3_clip`)** | `CLIP ViT-B/16` + `head_proj` (`2049->256`) + `head` | `CLIPProcessor` (`max_length=77`) | **All local** (`backend/models/clip_processor/`) | **Yes** (`backend/app/inference/fakeddit_clip.py`) |
| **M4 (`M4_vilt`)** | `dandelin/vilt-b32-mlm` + `head_proj` (`768->256`) + `head` | `ViltProcessor` (`max_length=40`) | `M4_vilt_best.pt` local; `ViltProcessor` **missing** | Yes (in `M4_vilt_test_embeddings.npy`) |
| **F4 (`F4_Gated_Feature_Fusion`)** | `projs.0..3` (`256->128`), `gate_net` (`512->4`), `classifier` (`128->64->2`) | `4 x 256-d` in order `['M1_text', 'M2_image', 'M3_clip', 'M4_vilt']` | **All local** (`F4_Gated_Feature_Fusion_best.pt`) | **Yes** (reproduced `0.9133` Acc) |

---

## 2. F4 Metric Discrepancy Analysis (`0.9133` vs `0.9129`)
- **Cause:** `F4_Gated_Feature_Fusion_best.pt` stores `state_dict` weights only, not activation functions (`GELU` vs `ReLU` in `projs`/`classifier`), plus `refit_epochs: 4` vs CV fold averaging. The `<0.04%` difference (`0.9133` vs `0.9129` Acc, `0.9112` vs `0.9109` F1, `0.9727` vs `0.9728` ROC-AUC) confirms the checkpoint and embeddings match.

---

## 3. Recommended Implementation Sequence
1. Cache `roberta-base` tokenizer/config and `dandelin/vilt-b32-mlm` `ViltProcessor` locally.
2. Implement `M1_text`, `M2_image`, and `M4_vilt` 256-d extractors and verify `strict=True` loading.
3. Expose `extract_features()` on `M3ClipClassifier` without altering `predict()`.
4. Implement `F4FusionPredictor` and wire `fk_f4_fusion`.
