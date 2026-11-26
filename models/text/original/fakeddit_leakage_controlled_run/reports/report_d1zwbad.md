# Post Verification Report (`d1zwbad`)

- **Prediction:** **FAKE (misinformation-related)**
- **Calibrated Probability p(True):** `0.0250`
- **CLIP Image-Text Similarity:** `0.2036`

## Per-Model Probabilities p(True)
- **M1_text**: 0.0641
- **M2_image**: 0.0552
- **M3_clip**: 0.0679
- **M4_vilt**: 0.0524

## Top Evidence Words (M1 Occlusion)
- `you`: +0.3116
- `say`: +0.0873
- `drawn`: -0.0220
- `poorly`: -0.0122

## Heatmap Artifacts
- **convnext_gradcam**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/d1zwbad_convnext_gradcam.png`
- **clip_patch_sim**: `/content/drive/MyDrive/NLP_Deepfake_project/fakeddit_leakage_controlled_run/explanations/d1zwbad_clip_patch_sim.png`

## Summary
Classified as FAKE (misinformation-related) (calibrated p(True)=0.025, thresholds [0.20, 0.85]) using F4 Gated Feature Fusion; CLIP image-text similarity=0.204, top influential words=['you', 'say', 'drawn']. Note: this reflects Fakeddit misinformation-related post patterns, not forensic image manipulation.
