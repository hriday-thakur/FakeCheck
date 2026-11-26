# Post Verification Report (`cac0dsf`)

- **Prediction:** **FAKE (misinformation-related)**
- **Calibrated Probability p(True):** `0.0354`
- **CLIP Image-Text Similarity:** `0.1974`

## Per-Model Probabilities p(True)
- **M1_text**: 0.0520
- **M2_image**: 0.7681
- **M3_clip**: 0.0575
- **M4_vilt**: 0.0525

## Top Evidence Words (M1 Occlusion)
- `say`: +0.0214
- `the`: -0.0064
- `jingle`: -0.0056

## Heatmap Artifacts
- **convnext_gradcam**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/cac0dsf_convnext_gradcam.png`
- **clip_patch_sim**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/cac0dsf_clip_patch_sim.png`

## Summary
Classified as FAKE (misinformation-related) (calibrated p(True)=0.035, thresholds [0.20, 0.85]) using F4 Gated Feature Fusion; CLIP image-text similarity=0.197, top influential words=['say', 'the', 'jingle']. Note: this reflects Fakeddit misinformation-related post patterns, not forensic image manipulation.
