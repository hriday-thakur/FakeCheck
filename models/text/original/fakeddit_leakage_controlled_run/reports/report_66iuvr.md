# Post Verification Report (`66iuvr`)

- **Prediction:** **REAL**
- **Calibrated Probability p(True):** `0.8912`
- **CLIP Image-Text Similarity:** `0.3470`

## Per-Model Probabilities p(True)
- **M1_text**: 0.8283
- **M2_image**: 0.9340
- **M3_clip**: 0.9324
- **M4_vilt**: 0.9413

## Top Evidence Words (M1 Occlusion)
- `seized`: +0.2966
- `car`: +0.2165
- `venezuela`: +0.1455
- `our`: -0.1142
- `plant`: +0.0939

## Heatmap Artifacts
- **convnext_gradcam**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/66iuvr_convnext_gradcam.png`
- **clip_patch_sim**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/66iuvr_clip_patch_sim.png`

## Summary
Classified as REAL (calibrated p(True)=0.891, thresholds [0.20, 0.85]) using F4 Gated Feature Fusion; CLIP image-text similarity=0.347, top influential words=['seized', 'car', 'venezuela']. Note: this reflects Fakeddit misinformation-related post patterns, not forensic image manipulation.
