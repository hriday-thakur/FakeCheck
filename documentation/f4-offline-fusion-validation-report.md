# F4 Phase 1.6 — Offline Fusion Head Validation Report

**Date:** 2026-11-27  
**Verdict:** **PASS** — `F4_Gated_Feature_Fusion_best.pt` independently reproduced on all `4,453` saved test embeddings.

---

## 1. Artifact & ID Verification
- **Embeddings Verified:** `M1_text_test_embeddings.npy`, `M2_image_test_embeddings.npy`, `M3_clip_test_embeddings.npy`, `M4_vilt_test_embeddings.npy` — all `(4453, 256) float32`, all finite.
- **IDs & Labels Verified:** `*_test_ids.npy` (`(4453,) <U7`) and `*_test_labels.npy` (`(4453,) int64`) are 100% identical across all four models.

---

## 2. Strict Checkpoint Loading & Offline Metrics
- **`strict=True` Load:** Passed (`24` tensors loaded into `projs.0..3`, `gate_net`, `classifier`).
- **Offline Inference (`n = 4,453`):**
  - **Accuracy:** `0.9133` (reported in `E1_baselines_vs_multimodal.csv`: `0.9129`)
  - **Macro Precision:** `0.9106` (reported: `0.9097`)
  - **Macro Recall:** `0.9119` (reported: `0.9124`)
  - **Macro F1:** `0.9112` (reported: `0.9109`)
  - **ROC-AUC:** `0.9727` (reported: `0.9728`)
  - **Confusion Matrix (`0=FAKE, 1=REAL`):** `[[2377, 204], [182, 1690]]`
- **Comparison with `M3_clip`:** Evaluated on the exact same `4,453` test IDs and labels (`M3_clip` Acc `0.9007`, Macro F1 `0.8991`, ROC-AUC `0.9621`).
