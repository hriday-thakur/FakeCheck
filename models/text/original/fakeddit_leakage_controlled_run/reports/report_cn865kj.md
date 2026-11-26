# Post Verification Report (`cn865kj`)

- **Prediction:** **FAKE (misinformation-related)**
- **Calibrated Probability p(True):** `0.0381`
- **CLIP Image-Text Similarity:** `0.1869`

## Per-Model Probabilities p(True)
- **M1_text**: 0.0442
- **M2_image**: 0.8309
- **M3_clip**: 0.0521
- **M4_vilt**: 0.0556

## Top Evidence Words (M1 Occlusion)
- `weve`: -0.0072
- `admit`: -0.0034
- `done`: -0.0032
- `it`: -0.0008
- `it`: -0.0007

## Heatmap Artifacts
- **convnext_gradcam**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/cn865kj_convnext_gradcam.png`
- **clip_patch_sim**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/cn865kj_clip_patch_sim.png`

## Summary
Classified as FAKE (misinformation-related) (calibrated p(True)=0.038, thresholds [0.20, 0.85]) using F4 Gated Feature Fusion; CLIP image-text similarity=0.187, top influential words=['weve', 'admit', 'done']. Note: this reflects Fakeddit misinformation-related post patterns, not forensic image manipulation.
