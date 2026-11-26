# F4 Phase 1.9 — M1, M2 & M4 Feature Extractor Specification Recovery Report

**Date:** 2026-11-27  
**Verdict:** **PARTIALLY READY** (All 3 backbones, checkpoints, and 256-d projection heads are verified; `roberta-base` and `dandelin/vilt-b32-mlm` are cached offline; exact pooling/transform details not stored in `state_dict` are documented as hypotheses).

---

## 1. M1 — Text (`M1_text_best.pt`)
- **Verified:** `roberta-base` (`[50265, 768]`, `[514, 768]`, `205` keys) + `proj` (`Linear(768, 256) -> LayerNorm(256)`) + `head` (`Linear(256, 2)`). `AutoTokenizer` + `AutoConfig` cached locally (`local_files_only=True`). `splits/test.csv` includes `text_norm`.
- **Unrecovered / Hypothesis:** Exact pooling (`[:, 0]` `[CLS]` vs mean pooling) and `max_length` (up to `512`) are not recorded in `.py`/`.ipynb` source.
- **Readiness:** Ready for `strict=True` wrapper reconstruction.

---

## 2. M2 — Image (`M2_image_best.pt`)
- **Verified:** `ConvNeXt-Tiny` (`E1_baselines_vs_multimodal.csv`: `"M2_image (ConvNeXt-Tiny)"`; `encoder.stem.0.weight` `(96, 3, 4, 4)`, `186` keys) + `head_proj` (`Linear(768, 256) -> LayerNorm(256)`) + `head` (`Linear(256, 2)`).
- **Unrecovered / Hypothesis:** Exact `timm` (`convnext_tiny`) vs `torchvision` transform (`224x224` ImageNet norm) is not recorded in `.py`/`.ipynb` source.
- **Readiness:** Ready for `strict=True` wrapper reconstruction using `timm.create_model("convnext_tiny", pretrained=False, num_classes=0)`.

---

## 3. M4 — ViLT (`M4_vilt_best.pt`)
- **Verified:** `dandelin/vilt-b32-mlm` (`[30522, 768]`, `[40, 768]`, `145` patches, `212` keys) + `head_proj` (`Linear(768, 256) -> LayerNorm(256)`) + `head` (`Linear(256, 2)`). `ViltProcessor` + `ViltConfig` cached locally (`local_files_only=True`).
- **Unrecovered / Hypothesis:** Exact `ViltProcessor` kwargs (`max_length=40`, `truncation=True`, `padding="max_length"` vs `True`) and `pooler_output` vs `last_hidden_state[:, 0]`.
- **Readiness:** Ready for `strict=True` wrapper reconstruction.
