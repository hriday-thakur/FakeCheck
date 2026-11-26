# F4 Integration Test Report

**Date:** 2026-11-27  
**Verdict:** **PASS**

- **Strict Checkpoint Loading & CPU Forward Pass:** All 5 checkpoints (`M1_text_best.pt`, `M2_image_best.pt`, `M3_clip_best.pt`, `M4_vilt_best.pt`, `F4_Gated_Feature_Fusion_best.pt`) loaded with `strict=True` and produced `{'prediction': 'FAKE', 'scores': {'REAL': 0.0846, 'FAKE': 0.9154}, 'processing_time': 0.7018}`.
- **API & Regression Suite (`backend/tests/test_api.py`):** Verified `test_fakeddit_f4_default_and_real_prediction`, `test_fakeddit_clip_real_prediction`, `test_snlive_blip_real_prediction_and_singleton`, `test_faceforensics_real_prediction_and_singleton`, `test_health_endpoint` (12 models), and `test_benchmark_endpoint`.
- **Frontend JS (`node --check`):** `0` errors.
