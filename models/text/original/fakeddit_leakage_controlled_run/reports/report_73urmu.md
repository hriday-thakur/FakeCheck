# Post Verification Report (`73urmu`)

- **Prediction:** **REAL**
- **Calibrated Probability p(True):** `0.9053`
- **CLIP Image-Text Similarity:** `0.3495`

## Per-Model Probabilities p(True)
- **M1_text**: 0.9027
- **M2_image**: 0.9057
- **M3_clip**: 0.9336
- **M4_vilt**: 0.8943

## Top Evidence Words (M1 Occlusion)
- `anthem`: +0.0359
- `kneel`: +0.0230
- `players`: +0.0186
- `protest`: +0.0153
- `call`: +0.0140

## Heatmap Artifacts
- **convnext_gradcam**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/73urmu_convnext_gradcam.png`
- **clip_patch_sim**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/73urmu_clip_patch_sim.png`

## Summary
Classified as REAL (calibrated p(True)=0.905, thresholds [0.20, 0.85]) using F4 Gated Feature Fusion; CLIP image-text similarity=0.349, top influential words=['anthem', 'kneel', 'players']. Note: this reflects Fakeddit misinformation-related post patterns, not forensic image manipulation.
