# Post Verification Report (`213rlt`)

- **Prediction:** **FAKE (misinformation-related)**
- **Calibrated Probability p(True):** `0.0304`
- **CLIP Image-Text Similarity:** `0.3062`

## Per-Model Probabilities p(True)
- **M1_text**: 0.0282
- **M2_image**: 0.1749
- **M3_clip**: 0.0782
- **M4_vilt**: 0.0812

## Top Evidence Words (M1 Occlusion)
- `the`: +0.0213
- `antiamerican`: +0.0147
- `american`: +0.0133
- `modern`: +0.0068
- `knocks`: -0.0031

## Heatmap Artifacts
- **convnext_gradcam**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/213rlt_convnext_gradcam.png`
- **clip_patch_sim**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/213rlt_clip_patch_sim.png`

## Summary
Classified as FAKE (misinformation-related) (calibrated p(True)=0.030, thresholds [0.20, 0.85]) using F4 Gated Feature Fusion; CLIP image-text similarity=0.306, top influential words=['the', 'antiamerican', 'american']. Note: this reflects Fakeddit misinformation-related post patterns, not forensic image manipulation.
