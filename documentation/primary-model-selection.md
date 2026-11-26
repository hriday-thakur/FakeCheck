# FakeCheck — Primary Model Selection & Evaluation Comparison

**Date:** 2026-11-26  
**Scope:** Verified validation and test evaluation results for the three team-selected primary checkpoints (`M3_clip_best.pt`, `blip-best.pt`, `xception_bilstm_video_best.pt`) compared against all other checkpoints trained on the same dataset.  
**Rules:** No metrics were calculated or invented; validation and test metrics are strictly distinguished, and exact source files are cited.

---

## 1. Fakeddit — Primary: `M3_clip_best.pt` (CLIP Dual-Encoder)

- **Evaluation Protocol:** All four standalone models (`M1_text`, `M2_image`, `M3_clip`, `M4_vilt`) and `Final Fusion (F4)` were evaluated on the **same held-out test split** (`n = 4,453`, `audit/split_summary.csv`).
- **Validation Metrics (`metrics/M3_clip_history.json`):**
  - Epoch 5 (final/best `val_accuracy`): `val_accuracy = 0.8984` (`0.8984497865648169`), `val_macro_f1 = 0.8968` (`0.8967505860810372`), `val_loss = 0.3515`.
- **Test Set Comparison (`metrics/E1_baselines_vs_multimodal.csv`, lines 2–6):**

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`M3_clip` (Primary)** | **`0.9007`** | **`0.8967`** | **`0.9031`** | **`0.8991`** | **`0.9621`** | **`0.9410`** |
| `M4_vilt` | `0.8495` | `0.8469` | `0.8553` | `0.8480` | `0.9301` | `0.8998` |
| `M1_text` (`roberta-base`) | `0.8491` | `0.8445` | `0.8497` | `0.8464` | `0.9231` | `0.8978` |
| `M2_image` (`ConvNeXt-Tiny`) | `0.7781` | `0.7813` | `0.7883` | `0.7773` | `0.8653` | `0.7970` |
| `F4 Gated Feature Fusion` *(no `.pt` checkpoint)* | `0.9129` | `0.9097` | `0.9124` | `0.9109` | `0.9728` | `0.9623` |

- **Finding:** Among checkpoints that actually exist in `checkpoints/`, `M3_clip_best.pt` has the highest test accuracy, macro F1, precision, recall, ROC-AUC, and PR-AUC.

---

## 2. SNLI-VE — Primary: `blip-best.pt` (BLIP)

- **Evaluation Protocol:** All four models (`blip`, `fusion`, `bridgetower`, `vilt`) were evaluated on the **same validation split** (`n = 6,000`) and **same test split** (`n = 6,000`, `dataset-split-distribution.csv`).
- **Validation Metrics (`validation-comparison.csv`):**
  - `blip` (Epoch 3): `val_acc = 0.7650`, `macro_f1 = 0.7638`, `val_loss = 0.5910`
  - `fusion` (Epoch 2): `val_acc = 0.7617`, `macro_f1 = 0.7628`, `val_loss = 0.6170`
  - `bridgetower` (Epoch 3): `val_acc = 0.7513`, `macro_f1 = 0.7506`, `val_loss = 0.6181`
  - `vilt` (Epoch 3): `val_acc = 0.6952`, `macro_f1 = 0.6952`, `val_loss = 0.7553`
- **Test Set Comparison (`test-evaluation/snlive-four-model-test-comparison.csv`):**

| Model | Test Accuracy | Macro F1 | Macro ROC-AUC (OvR) |
| :--- | :--- | :--- | :--- |
| **`blip` (Primary)** | **`0.7575`** | `0.7562` | `0.9028` |
| `fusion` | `0.7553` | **`0.7565`** | **`0.9052`** |
| `bridgetower` | `0.7462` | `0.7455` | `0.8952` |
| `vilt` | `0.6963` | `0.6973` | `0.8562` |

- **Note:** `test-class-report.csv` reports BLIP test accuracy as `0.757333` (minor discrepancy vs `0.7575` in `snlive-four-model-test-comparison.csv`), with per-class precision/recall/F1: `entailment` (`0.7320` / `0.8180` / `0.7726`), `neutral` (`0.7245` / `0.6510` / `0.6858`), `contradiction` (`0.8161` / `0.8030` / `0.8095`).

---

## 3. FaceForensics++ — Primary: `xception_bilstm_video_best.pt` (Xception + BiLSTM)

- **Evaluation Protocol:** Evaluated on a custom `GroupShuffleSplit` (`ori_vid`-grouped) from `KoreaPeter/faceforensics-c23` (`audit/dataset_audit.json`), **not** the official FF++ test protocol.
  - **Fair Comparison Caveat:** `xception_bilstm_video_best.pt` is evaluated at the **video level** (up to 8 frames/video, threshold `0.25`), whereas `SoftVoting_Frame_Ensemble`, `EfficientNet_B4`, `Xception`, and `ResNet50` are evaluated at the **frame level** (`n = 922` test frames). Direct metric comparison across video-level vs. frame-level units is not strictly apples-to-apples.
- **Validation Metric (`xception_bilstm_video_best.pt` checkpoint metadata):** `epoch = 2`, `val_f1 = 0.9127` (vs. `Xception_best.pt` `val_f1 = 0.8859`, `EfficientNet_B4_best.pt` `val_f1 = 0.8691`, `ResNet50_best.pt` `val_f1 = 0.8058`).
- **Test Set Metrics (`models/video/original/FFPP_PJ07_FINAL_corrected/metrics/` & `audit/final_run_summary.json`):**

| Model | Evaluation Level | Test Accuracy | F1 Macro | ROC-AUC | PR-AUC | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Xception + BiLSTM` (Primary)** | Video-level | **`0.9163`** | **`0.8942`** | **`0.9619`** | **`0.9742`** | **`0.7910`** |
| `SoftVoting_Frame_Ensemble` | Frame-level | `0.8709` | `0.8709` | `0.9464` | `0.9493` | `0.7419` |
| `EfficientNet_B4` | Frame-level | `0.8633` | — | `0.9430` | — | — |
| `Xception` | Frame-level | `0.8547` | — | `0.9241` | — | — |
| `ResNet50` | Frame-level | `0.7375` | — | `0.8327` | — | — |
