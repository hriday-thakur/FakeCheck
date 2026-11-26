# F4 Phase 1.12 — M2 & M4 Sample Parity Testing Report

**Date:** 2026-11-27  
**M2 Verdict:** **UNVERIFIED**  
**M4 Verdict:** **UNVERIFIED**  
**Overall Verdict:** **BLOCKED** (`datasets` library not installed in active Python environment).

---

## 1. Execution Findings
- Attempted to stream 10 matching test images by `id` (`splits/test.csv`) from `AdoCleanCode/Fakeddit` using `datasets.load_dataset('AdoCleanCode/Fakeddit', split='test', streaming=True)`.
- **Blocker:** `ModuleNotFoundError: No module named 'datasets'` in Python `3.13.2`.
- **Impact:** Without the 10 original Fakeddit test images, M2 (`ConvNeXt-Tiny`) and M4 (`ViLT`) numerical parity against `M2_image_test_embeddings.npy` and `M4_vilt_test_embeddings.npy` cannot be executed.

---

## 2. Recommended Next Action
1. Either install `datasets` (`pip install datasets`) to stream 10 matching test images from `AdoCleanCode/Fakeddit` and run M2/M4 parity, **or**
2. Keep `fk_m3_clip` as the production Fakeddit primary model (already verified end-to-end).
