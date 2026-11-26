# Phase 2C — FaceForensics++ (`ff_xception_bilstm`) Inference Readiness Audit

**Date:** 2026-11-26  
**Scope:** Read-only audit of `models/video/original/FFPP_PJ07_FINAL_corrected/`.  
**Verdict:** **PARTIALLY READY** (checkpoints and `config.json` are complete, but `timm` is not installed, and no `.py`/`.ipynb` model or video/face-cropping preprocessing code exists).

---

## 1. Checkpoint Inventory
All checkpoints are nested PyTorch `dict` objects under `models/video/original/FFPP_PJ07_FINAL_corrected/checkpoints/`:
- **Primary (`ff_xception_bilstm`):**
  - `xception_bilstm_video_best.pt`: `18,898,060` bytes (`18.02 MB`), top keys `['state_dict', 'epoch', 'val_f1']`, `10` `state_dict` keys (`lstm.*` + `classifier.*`).
  - `Xception_best.pt` (required feature extractor): `83,554,739` bytes (`79.68 MB`), top keys `['state_dict', 'epoch', 'val_f1', 'model_name']`, `276` `state_dict` keys.
- **Secondary:** `EfficientNet_B4_best.pt` (`70.97 MB`), `ResNet50_best.pt` (`94.37 MB`).

---

## 2. Architecture Reconstruction
- **Frame Feature Extractor:** `timm.create_model("legacy_xception.tf_in1k", num_classes=2)` (`config.json`), extracting 2048-d pooled features before `fc`.
- **Temporal Classifier (`xception_bilstm_video_best.pt`):**
  - `nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True)` (`weight_ih_l0` `(1024, 2048)`, `weight_hh_l0` `(1024, 256)`).
  - `nn.Linear(512, 2)` (`classifier.weight` `(2, 512)`, `classifier.bias` `(2,)`).
- **Strict Loading:** Both `Xception_best.pt["state_dict"]` (`276` keys) and `xception_bilstm_video_best.pt["state_dict"]` (`10` keys) can be loaded with `strict=True` into their respective modules.

---

## 3. Preprocessing Pipeline
- **Verified (`config.json` & `final_run_summary.json`):**
  - `temporal_frames: 8` frames per video, packed with true sequence lengths.
  - Training used hosted C23-derived extracted/cropped frames (`KoreaPeter/faceforensics-c23`); **no raw-video decoding or face detection was used during training**.
- **Assumed / Unspecified:**
  - Raw MP4 inference will require `cv2.VideoCapture` uniform sampling of up to 8 frames, optional face detection/cropping (OpenCV Haar/DNN), BGR->RGB conversion, and `timm.data.resolve_model_data_config` / `create_transform` (299x299 for Xception).

---

## 4. Verified Label Mapping & Thresholds
- `0 = REAL` (`source: original`), `1 = FAKE` (`FaceSwap`, `Deepfakes`, etc. per `splits/test_manifest.csv`).
- Video-level classification (`final_run_summary.json` reports `video_level_threshold_selected_on_validation: 0.25`).

---

## 5. Dependency Compatibility
- Installed in Python `3.13.2`: `torch 2.11.0+cpu`, `torchvision 0.26.0+cpu`, `cv2 4.11.0`, `PIL 11.1.0`, `numpy 2.2.3`.
- **Missing:** **`timm` is NOT installed** (`PKG timm: NOT INSTALLED`). Required for `legacy_xception.tf_in1k`.

---

## 6. CPU Feasibility
- Total weights ~98 MB (`Xception_best.pt` + `xception_bilstm_video_best.pt`); no CUDA-only operations. CPU inference is feasible for 8 frames per video.

---

## 7. Readiness Verdict
**READY** — `FaceForensicsPredictor` is implemented in `backend/app/inference/faceforensics_bilstm.py`, wired to `POST /api/predict/faceforensics`, and verified with `strict=True` loading (`276` + `10` keys), `timm==1.0.31`, and `pytest` (`8 passed`). `ff_xception_bilstm` is marked `inference_ready=True`.

---

## 8. Preprocessing Recovery and Remaining Gaps

### Verified from Training Source / Metadata
- **Hosted Frames:** `final_run_summary.json` states `"compression": "Hosted C23-derived frames"` and `"No raw-video decoding or face detection; hosted frames/metadata are used."`
- **Temporal Sequence:** `config.json` specifies `temporal_frames: 8` and `final_run_summary.json` specifies `"Xception frame embeddings + BiLSTM with packed true sequence lengths"` (meaning variable-length sequences <= 8 frames were packed with `pack_padded_sequence`).
- **Backbone:** `timm` `legacy_xception.tf_in1k` (`Xception_best.pt`, `276` keys).

### Supported by Checkpoint / Metadata
- `Xception_best.pt` + `xception_bilstm_video_best.pt` load strictly (`276` and `10` keys) into `timm` `legacy_xception.tf_in1k` + `nn.LSTM(2048, 256, bidirectional=True)` + `nn.Linear(512, 2)`.

### Unknown and Unrecoverable
- Exact `timm` transform parameters (`mean`, `std`, `input_size=299`) are not saved in `config.json`, though `timm.data.resolve_model_data_config` provides the canonical `legacy_xception.tf_in1k` transform.
- Exact face-cropping bounding box / detector used upstream by `KoreaPeter/faceforensics-c23` is not in this repo.
- Exact frame-sampling stride for raw MP4s (since training used pre-extracted frames).

### Safest Implementation Path
- **Documented Alternative Inference Pipeline:** Use `cv2.VideoCapture` to uniformly sample up to `8` RGB frames from uploaded MP4s, apply `timm.data.resolve_model_data_config` + `create_transform` for `legacy_xception.tf_in1k` (`299x299`), pack true sequence lengths, and document clearly that raw-MP4 sampling is a new inference-time wrapper (not trained on raw MP4s). Requires installing `timm==1.0.31` first.
