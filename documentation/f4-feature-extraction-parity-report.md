# F4 Phase 1.10 — Feature Extraction Parity Report

**Date:** 2026-11-27  
**Overall Verdict:** **PARTIALLY VERIFIED** (`M1` verified with `cos = 1.000000`; `M2` and `M4` unverified due to missing raw Fakeddit test images).

---

## 1. Extractor Verdicts

### M1 — RoBERTa (`M1_text_best.pt`): **VERIFIED**
- **Checkpoint Load:** `strict=True` (`205` keys) into `RobertaModel` + `Sequential(Linear(768, 256), LayerNorm(256))` + `Linear(256, 2)`.
- **Numerical Parity (`splits/test.csv` first 5 samples vs `M1_text_test_embeddings.npy`):**
  - `[CLS]` pooling: `cos = 0.770040`, `MAE = 0.269236`
  - **Attention-masked Mean Pooling + `F.gelu(proj(...))`:** **`cos = 1.000000`**, **`MAE = 0.000341`** (exact match).

### M2 — ConvNeXt-Tiny (`M2_image_best.pt`): **UNVERIFIED**
- `E3a_frozen_pretrained_features.npz` confirms `768`-d backbone features, and `M2_image_test_embeddings.npy` confirms `256`-d projected outputs.
- Raw Fakeddit test images are not stored locally, so numerical parity cannot be tested.

### M4 — ViLT (`M4_vilt_best.pt`): **UNVERIFIED**
- `E3a_frozen_pretrained_features.npz` confirms `768`-d backbone features, and `M4_vilt_test_embeddings.npy` confirms `256`-d projected outputs.
- Raw Fakeddit test images are not stored locally, so numerical parity cannot be tested.
