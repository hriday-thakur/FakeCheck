# IEEE-Ready Results Tables (Fakeddit Misinformation-Related Post Classification)

## Table I: E1 Baselines vs. Multimodal Fusion (Held-Out Test)
| model                                  |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   roc_auc |   pr_auc |
|:---------------------------------------|-----------:|------------------:|---------------:|-----------:|----------:|---------:|
| M1_text (DeBERTa/RoBERTa)              |     0.8491 |            0.8445 |         0.8497 |     0.8464 |    0.9231 |   0.8978 |
| M2_image (ConvNeXt-Tiny)               |     0.7781 |            0.7813 |         0.7883 |     0.7773 |    0.8653 |   0.797  |
| M3_clip (CLIP Dual-Encoder)            |     0.9007 |            0.8967 |         0.9031 |     0.8991 |    0.9621 |   0.941  |
| M4_vilt (Cross-Modal)                  |     0.8495 |            0.8469 |         0.8553 |     0.848  |    0.9301 |   0.8998 |
| Final Fusion (F4 Gated Feature Fusion) |     0.9129 |            0.9097 |         0.9124 |     0.9109 |    0.9728 |   0.9623 |

## Table II: E2 Fusion Strategy Comparison (Selected by Validation CV)
| method                  |   val_cv_macro_f1 |   test_accuracy |   test_macro_f1 |   test_roc_auc |   test_log_loss |
|:------------------------|------------------:|----------------:|----------------:|---------------:|----------------:|
| M1_text (single)        |            0.8344 |          0.8491 |          0.8464 |         0.9231 |          0.3579 |
| M2_image (single)       |            0.7745 |          0.7781 |          0.7773 |         0.8653 |          0.4675 |
| M3_clip (single)        |            0.8968 |          0.9007 |          0.8991 |         0.9621 |          0.2709 |
| M4_vilt (single)        |            0.8366 |          0.8495 |          0.848  |         0.9301 |          0.3532 |
| Concat Baseline         |            0.8875 |          0.89   |          0.8873 |         0.9557 |          0.2938 |
| F1 Simple Average       |            0.8897 |          0.9028 |          0.9014 |         0.9656 |          0.2898 |
| F2 Weighted Average     |            0.897  |          0.9025 |          0.9009 |         0.9692 |          0.2447 |
| F3 Stacking (LogReg)    |            0.9042 |          0.9124 |          0.9104 |         0.9718 |          0.2145 |
| F4 Gated Feature Fusion |            0.9056 |          0.9129 |          0.9109 |         0.9728 |          0.2244 |

## Table III: E4 Ablation & Missing-Modality Analysis
| experiment                                     |   test_accuracy |   test_macro_f1 |   test_roc_auc |
|:-----------------------------------------------|----------------:|----------------:|---------------:|
| Leave-one-out (drop M1_text)                   |          0.9106 |          0.9086 |         0.9685 |
| Leave-one-out (drop M2_image)                  |          0.9073 |          0.9051 |         0.9697 |
| Leave-one-out (drop M3_clip)                   |          0.8897 |          0.8873 |         0.9564 |
| Leave-one-out (drop M4_vilt)                   |          0.9115 |          0.9095 |         0.9715 |
| Modality: Text-derived only (M1)               |          0.8491 |          0.8446 |         0.9231 |
| Modality: Image-derived only (M2)              |          0.786  |          0.7831 |         0.8653 |
| Modality: Multimodal-only (M3+M4)              |          0.9073 |          0.9052 |         0.9668 |
| Modality: All models (M1+M2+M3+M4)             |          0.9124 |          0.9104 |         0.9718 |
| Missing Image -> Full Stacker (masked image)   |          0.7319 |          0.6772 |         0.9165 |
| Missing Text -> Full Stacker (empty text)      |          0.5823 |          0.3739 |         0.871  |
| Missing Image -> Subset Stacker (M1 text only) |          0.8491 |          0.8446 |         0.9231 |
| Missing Text -> Subset Stacker (M2 image only) |          0.786  |          0.7831 |         0.8653 |
