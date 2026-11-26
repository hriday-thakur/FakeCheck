# F4 Phase 1.11 — Original Fakeddit Test Input Recovery Report

**Date:** 2026-11-27  
**Verdict:** **RECOVERABLE** (Text and IDs for all `4,453` test samples are present locally in `splits/test.csv`; original images are absent locally but recoverable from `AdoCleanCode/Fakeddit` on HuggingFace Hub).

---

## 1. Search Locations & Findings
- **Local Project (`models/text/original/fakeddit_leakage_controlled_run/`):**
  - `splits/test.csv`: Contains all `4,453` test sample IDs, `text`, `text_norm`, and `label` (matching `M1_text_test_ids.npy` and `M2_image_test_ids.npy`).
  - **Images:** `0` Fakeddit image files exist locally (confirmed by Phase 1.4 complete 127-file inventory).

---

## 2. Official Recovery Source & Next Action
- **Official Dataset Source (`run_summary.json`):** `AdoCleanCode/Fakeddit` on HuggingFace Hub.
- **M2 & M4 Parity Testing:** Cannot proceed until a small subset (e.g., 5–10 samples) of the `4,453` test images is fetched from `AdoCleanCode/Fakeddit` matched by exact `id` from `splits/test.csv`.
- **Exact Next Action:** Fetch 5–10 matching test images by `id` from `AdoCleanCode/Fakeddit` (streaming, without downloading the full 794k-row dataset) to run M2 (`ConvNeXt-Tiny`) and M4 (`ViLT`) numerical parity tests against `M2_image_test_embeddings.npy` and `M4_vilt_test_embeddings.npy`.
