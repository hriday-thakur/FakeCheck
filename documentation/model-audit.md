# FakeCheck — Technical Model Audit Report

**Audit Date:** 2026-11-26  
**Scope:** Read-only technical audit of all three original model pipelines under `models/` (`fakeddit_leakage_controlled_run`, `SNLI_VE_COMPLETE_PROJECT`, and `FFPP_PJ07_FINAL_corrected`), reconciled against `inference_environment.txt` and `checkpoint_inventory.csv`.  
**Constraints:** No files were modified, deleted, or retrained; no mock predictions or metrics were generated.

---

## Inference Environment (`inference_environment.txt`)

Recorded from `models/text/original/fakeddit_leakage_controlled_run/inference_environment.txt` (generated `2026-11-26T10:39:14.610093`):
- **Python:** `3.13.16` (`Linux-6.6.159+-x86_64-with-glibc2.39`)
- **PyTorch / CUDA:** `torch 2.14.1+cu130`, `torchvision 0.29.1+cu130`, CUDA `13.0`
- **Core Packages:** `transformers 5.22.1`, `timm 1.0.31`, `Pillow 11.3.0`, `numpy 2.4.6`, `pandas 2.2.3`, `scikit-learn 1.6.1`, `opencv-python 5.0.0.93`

---

## Checkpoint Inventory Reconciliation (`checkpoint_inventory.csv` vs Local Files)

| Component | `checkpoint_inventory.csv` Status & Size | Local Workspace Path & Actual Size | Reconciliation |
| :--- | :--- | :--- | :--- |
| `fakeddit/M1_text` | `True`, `476.32 MB` | `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M1_text_best.pt` (`499,459,780` bytes / `476.32 MB`) | **Exact Match** |
| `fakeddit/M2_image` | `True`, `106.95 MB` | `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M2_image_best.pt` (`112,148,299` bytes / `106.95 MB`) | **Exact Match** |
| `fakeddit/M3_clip` | `True`, `572.92 MB` | `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M3_clip_best.pt` (`600,748,249` bytes / `572.92 MB`) | **Exact Match** |
| `fakeddit/M4_vilt` | `True`, `426.54 MB` | `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M4_vilt_best.pt` (`447,260,441` bytes / `426.54 MB`) | **Exact Match** |
| `snlive/BLIP` | `True`, `145.02 MB` | `models/image/original/SNLI_VE_COMPLETE_PROJECT/blip-best.pt` (`152,065,355` bytes / `145.02 MB`) | **Exact Match** |
| `snlive/ViLT` | `True`, `111.2 MB` | `models/image/original/SNLI_VE_COMPLETE_PROJECT/vilt-best.pt` (`116,597,987` bytes / `111.20 MB`) | **Exact Match** |
| `snlive/BridgeTower` | `True`, `150.28 MB` | `models/image/original/SNLI_VE_COMPLETE_PROJECT/bridgetower-best.pt` (`157,584,481` bytes / `150.28 MB`) | **Exact Match** |
| `snlive/Fusion` | `True`, `1.77 MB` | `models/image/original/SNLI_VE_COMPLETE_PROJECT/fusion-best.pt` (`1,852,614` bytes / `1.77 MB`) | **Exact Match** |
| `ffpp` | `Not configured`, `False` | `models/video/original/FFPP_PJ07_FINAL_corrected/checkpoints/` (`xception_bilstm_video_best.pt` `18.02 MB`, `Xception_best.pt` `79.68 MB`, `EfficientNet_B4_best.pt` `67.68 MB`, `ResNet50_best.pt` `90.00 MB`) | **Reconciled:** Inventory was generated without FFPP mounted; all 4 FFPP checkpoints exist locally. |

---

## Executive Summary

**Team-Confirmed Primary Production Checkpoints:**
1. **Fakeddit:** `M3_clip_best.pt` — CLIP Dual-Encoder (verified test accuracy `0.9007`, macro F1 `0.8991`, ROC-AUC `0.9621` in `E1_baselines_vs_multimodal.csv`). Secondary experimental checkpoints: `M1_text_best.pt` (RoBERTa-base), `M2_image_best.pt` (ConvNeXt-Tiny), `M4_vilt_best.pt` (ViLT).
2. **SNLI-VE:** `blip-best.pt` — BLIP (`Salesforce/blip-itm-base-coco`, verified test accuracy `0.7575`, macro F1 `0.7562`, ROC-AUC `0.9028`). Secondary checkpoints: `vilt-best.pt`, `bridgetower-best.pt`, `fusion-best.pt`.
3. **FaceForensics++:** `xception_bilstm_video_best.pt` (+ `Xception_best.pt`) — Xception + BiLSTM (verified test accuracy `0.9163`, F1 macro `0.8942`, ROC-AUC `0.9619`). Secondary checkpoints: `EfficientNet_B4_best.pt`, `Xception_best.pt` (frame-level), `ResNet50_best.pt`.

All three model pipelines contain trained PyTorch checkpoints and evaluation artifacts, **but none of the three is drop-in ready for backend inference** without writing a model wrapper / inference module.

---

## 1. Fakeddit — Misinformation Classification (`models/text/original/fakeddit_leakage_controlled_run`)

### 1.1 Pipeline Overview, Role & Image Cache Requirement
- **Dataset & Task (`run_summary.json`, `audit/dataset_audit.json`):** `AdoCleanCode/Fakeddit` binary misinformation-related post classification (*NOT forensic deepfake detection*).
- **Class Labels (`audit/dataset_audit.json`, `splits/test.csv`):**
  - `0`: Non-true / misinformation-related (`6_way_label` 1–5)
  - `1`: True / REAL (`6_way_label` 0)
- **Independent Classifiers vs. Ensemble:** Direct inspection of `checkpoints/` and `metrics/E1_baselines_vs_multimodal.csv` confirms that **`M1_text`, `M2_image`, `M3_clip`, and `M4_vilt` are four independent binary classifiers** (each has its own `256 -> 2` `head.weight` `(2, 256)`). They were also evaluated alongside `Final Fusion (F4 Gated Feature Fusion)`, but **no `F4` checkpoint exists** in `checkpoints/`.
- **Image Cache Requirement:** The Fakeddit image cache referenced in `splits/*.csv` (`/content/drive/MyDrive/...`) is **only required for training/evaluation** on the Fakeddit dataset, **not** for live inference (where `M1_text` takes raw text and `M2_image`/`M3_clip`/`M4_vilt` take user-provided images/text directly).
- **Shared Blockers Across All 4 Components:** No `.py` or `.ipynb` model/preprocessing/inference code exists in `fakeddit_leakage_controlled_run`, and `checkpoints/` contains raw `OrderedDict` `state_dict` files with no HuggingFace `config.json` or tokenizer files.

### 1.2 `M1_text` (`M1_text_best.pt`)
- **Checkpoint:** `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M1_text_best.pt` (`499,459,780` bytes / `476.32 MB`, `205` keys).
- **Architecture:** `roberta-base` (`encoder.embeddings.word_embeddings.weight` `[50265, 768]`, `position_embeddings.weight` `[514, 768]`) + `proj` (`768 -> 256`) + `head` (`256 -> 2`). *(Note: `E1_baselines_vs_multimodal.csv` labels it `M1_text (DeBERTa/RoBERTa)`, but the weights are `roberta-base`.)*
- **Input & Output:** Tokenized text (max 512 tokens; `splits/test.csv` also shows `text_norm` lowercased/punctuation-stripped text) -> 2-class logits `(B, 2)`.
- **Test Metrics (`E1_baselines_vs_multimodal.csv`):** Accuracy `0.8491`, Macro F1 `0.8464`, ROC-AUC `0.9231`, PR-AUC `0.8978`.
- **Missing / Blockers:** Needs custom PyTorch wrapper (`encoder` + `proj` + `head`) and `roberta-base` tokenizer/config.

### 1.3 `M2_image` (`M2_image_best.pt`)
- **Checkpoint:** `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M2_image_best.pt` (`112,148,299` bytes / `106.95 MB`, `186` keys).
- **Architecture:** `ConvNeXt-Tiny` (`encoder.stem.0.weight` `(96, 3, 4, 4)`) + `head_proj` (`-> 256`) + `head` (`256 -> 2`).
- **Input & Output:** RGB image (standard ConvNeXt preprocessing, e.g., 224x224 ImageNet-normalized) -> 2-class logits `(B, 2)`.
- **Test Metrics (`E1_baselines_vs_multimodal.csv`):** Accuracy `0.7781`, Macro F1 `0.7773`, ROC-AUC `0.8653`, PR-AUC `0.7970`.
- **Missing / Blockers:** Needs ConvNeXt-Tiny wrapper (`encoder` + `head_proj` + `head`) and exact image transform config.

### 1.4 `M3_clip` (`M3_clip_best.pt`)
- **Checkpoint:** `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M3_clip_best.pt` (`600,748,249` bytes / `572.92 MB`, `404` keys).
- **Architecture:** `CLIP Dual-Encoder` (`clip.logit_scale`, `clip.text_model.embeddings.token_embedding.weight` `(49408, 512)`, `position_embedding.weight` `(77, 512)`, plus vision encoder) + `head_proj` (`-> 256`) + `head` (`256 -> 2`).
- **Input & Output:** Paired image + text (CLIP processor, max 77 text tokens) -> 2-class logits `(B, 2)`.
- **Test Metrics (`E1_baselines_vs_multimodal.csv`):** Accuracy `0.9007`, Macro F1 `0.8991`, ROC-AUC `0.9621`, PR-AUC `0.9410` (strongest single Fakeddit checkpoint).
- **Missing / Blockers:** Needs CLIP dual-encoder wrapper (`clip` + `head_proj` + `head`) and CLIP processor/tokenizer files.

### 1.5 `M4_vilt` (`M4_vilt_best.pt`)
- **Checkpoint:** `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M4_vilt_best.pt` (`447,260,441` bytes / `426.54 MB`, `212` keys).
- **Architecture:** `ViLT` (`dandelin/vilt-b32-mlm` dimensions: `text_embeddings.word_embeddings.weight` `(30522, 768)`, `position_embeddings.weight` `(40, 768)`, `145` image patch positions) + `head_proj` (`-> 256`) + `head` (`256 -> 2`).
- **Input & Output:** Paired image + text (`ViltProcessor`, max 40 text tokens) -> 2-class logits `(B, 2)`.
- **Test Metrics (`E1_baselines_vs_multimodal.csv`):** Accuracy `0.8495`, Macro F1 `0.8480`, ROC-AUC `0.9301`, PR-AUC `0.8998`.
- **Missing / Blockers:** Needs ViLT wrapper (`encoder` + `head_proj` + `head`) and `ViltProcessor` config/tokenizer files.

---

## 2. SNLI-VE — Image-Text Analysis (`models/image/original/SNLI_VE_COMPLETE_PROJECT`)

### 2.1 Model Architecture & Implementation
- **Verified (`executed-notebook-code.py` lines 179–222 & `complete-experiment-info.json`):**
  - 4-model experiment: `blip`, `vilt`, `bridgetower`, and `fusion`.
  - **Winner (`blip`):** `Branch("blip")` wraps `Salesforce/blip-itm-base-coco` (`BlipForImageTextRetrieval`, HF revision `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`).
    - Vision backbone (`vision_model`) is frozen; last 4 layers of `text_encoder.encoder.layer[-4:]` are unfrozen.
    - `forward()` passes `vision_model(pixel_values).last_hidden_state` as `encoder_hidden_states` to `text_encoder`, takes `[CLS]` (`[:, 0]`), projects `768 -> 256` (`LayerNorm(768) -> Linear(768, 256) -> GELU -> LayerNorm(256)`), and classifies via `head` (`Dropout(0.15) -> Linear(256, 3)`).

### 2.2 Best Trained Checkpoint & Exact Path
- **Selected Winner (`selection.json`):** `blip` at Epoch 3.
- **Best Checkpoint Path:** `models/image/original/SNLI_VE_COMPLETE_PROJECT/blip-best.pt` (**152,065,355 bytes** / ~145.0 MB).
- **Other Checkpoints Present:** `blip-latest.pt` (456.2 MB), `bridgetower-best.pt` (157.6 MB), `bridgetower-latest.pt` (472.8 MB), `vilt-best.pt` (116.6 MB), `vilt-latest.pt` (349.8 MB), `fusion-best.pt` (1.85 MB), `fusion-latest.pt` (5.56 MB).

### 2.3 Checkpoint Completeness & Loadability
- **Verified:** `blip-best.pt` is a partial PyTorch `state_dict` saved via `torch.save(trainable(net), ...)` (`executed-notebook-code.py` lines 229–230, 345). It **only** contains trainable parameters (`text_encoder.encoder.layer[-4:]`, `project.*`, `head.*`).
- **Loadability:** Requires instantiating `Branch("blip")` (which downloads/loads `Salesforce/blip-itm-base-coco` from HuggingFace Hub or local cache) and calling `net.load_state_dict(torch.load("blip-best.pt", map_location="cpu", weights_only=False), strict=False)`.

### 2.4 Dataset & Actual Classification Task
- **Dataset:** `sedrickkeh/snli-ve` (HF revision `662ae119b528a30446472560bf3b1af541f0b292`).
- **Task:** 3-class **Visual Entailment** (given an image premise and a text hypothesis, classify whether the image entails, is neutral to, or contradicts the text). **Not** a standalone AI-generated image detector.
- **Splits (`dataset-split-distribution.csv`):** `train`: 30,000; `validation`: 6,000; `test`: 6,000 (balanced across 3 classes).

### 2.5 Input Preprocessing & Inference Requirements
- **Processor:** `BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco", revision="bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c")`.
- **Image:** `PIL.Image.open(...).convert("RGB")`.
- **Text + Image Collation:** `proc(text=texts, images=pics, padding=True, truncation=True, max_length=64, return_tensors="pt")`.

### 2.6 Required Dependencies & Python Versions
- **Training Environment (`complete-experiment-info.json`):** Python `3.13.16`, PyTorch `2.11.0+cu130`, NumPy `2.1.3`, Pandas `2.3.3`.
- **Pinned/Used (`executed-notebook-code.py`):** `transformers==4.49.0`, `huggingface_hub>=0.26,<1`, `safetensors`, `torch`, `Pillow`, `scikit-learn`, `pyarrow`, `matplotlib`.
- **Full Freeze:** `requirements.txt` (935 lines) is present.

### 2.7 Class Labels & Output Format
- `0`: `"entailment"`
- `1`: `"neutral"`
- `2`: `"contradiction"`
- Output: Raw logits `(B, 3)` from `Branch.forward()`, probabilities via `logits.float().softmax(-1)`.

### 2.8 Available Evaluation Metrics (Test Set, `n = 6,000`)
From `test-evaluation/snlive-four-model-test-comparison.csv`:
- **BLIP (Winner):** Accuracy `0.7575`, Macro F1 `0.7562`, Macro ROC-AUC (OvR) `0.9028`.
- **Fusion:** Accuracy `0.7553`, Macro F1 `0.7565`, Macro ROC-AUC (OvR) `0.9052`.
- **BridgeTower:** Accuracy `0.7462`, Macro F1 `0.7455`, Macro ROC-AUC (OvR) `0.8952`.
- **ViLT:** Accuracy `0.6963`, Macro F1 `0.6973`, Macro ROC-AUC (OvR) `0.8562`.
- *Note:* `test-class-report.csv` lists BLIP test accuracy as `0.757333` (minor discrepancy vs `0.7575`).

### 2.9 Existing Prediction / Inference Code
- No standalone inference script. Inference logic exists only in `executed-notebook-code.py` (`evaluate(model, batches)`, lines 259–268), which hardcodes `.cuda()` and `torch.autocast("cuda", dtype=torch.float16)`.

### 2.10 Additional Files Required for Inference
- `Branch` class definition (from `executed-notebook-code.py`).
- `blip-best.pt`.
- HuggingFace weights & processor for `Salesforce/blip-itm-base-coco` (requires internet or local HF cache).

### 2.11 CPU / GPU Requirements
- Trained on 2x `Tesla T4`.
- Inference can run on CPU if `.cuda()` and CUDA autocast are removed, or on a single GPU.

### 2.12 Missing Files, Inconsistencies & Integration Blockers
- **Blocker:** `blip-best.pt` is a partial checkpoint (trainable params only); requires downloading `Salesforce/blip-itm-base-coco` at runtime unless cached.
- **Blocker:** `executed-notebook-code.py` cannot be imported directly as a backend module (contains Kaggle paths `/kaggle/working`, `/tmp/p07_hf`, and `assert torch.cuda.device_count() >= 1` at module level).
- **Task Mismatch Note:** Classifies image-text entailment, not whether an image is AI-generated.

---

## 3. FaceForensics++ — Video Manipulation Detection (`models/video/original/FFPP_PJ07_FINAL_corrected`)

### 3.1 Model Architecture & Implementation
- **Verified (`config.json` & checkpoint inspection):**
  - **Frame-level models (`timm` backbones, 2-class heads):**
    - `Xception`: `legacy_xception.tf_in1k` (`fc.weight` `(2, 2048)`).
    - `EfficientNet_B4`: `tf_efficientnet_b4.ns_jft_in1k` (`classifier.weight` `(2, 1792)`).
    - `ResNet50`: `resnet50.a1_in1k` (`fc.weight` `(2, 2048)`).
  - **Video-level Temporal Model:** `Xception` frame embeddings (2048-d) + 1-layer bidirectional LSTM (`temporal_hidden: 256` -> 512-d output, `temporal_frames: 8`) + `classifier` `(2, 512)`.

### 3.2 Best Trained Checkpoint & Exact Path
- **Best Overall (Video-level):** `models/video/original/FFPP_PJ07_FINAL_corrected/checkpoints/xception_bilstm_video_best.pt` (**18,898,060 bytes** / ~18.02 MB), paired with `Xception_best.pt` (**83,554,739 bytes** / ~79.68 MB).
- **Best Single Frame-level:** `EfficientNet_B4_best.pt` (**70,966,947 bytes** / ~67.68 MB).
- **Other Checkpoints:** `ResNet50_best.pt` (**94,371,107 bytes** / ~90.00 MB).

### 3.3 Checkpoint Completeness & Loadability
- **Verified:** All 4 `.pt` files are complete, loadable PyTorch `dict` checkpoints (`torch.load(..., map_location='cpu', weights_only=False)`):
  - `xception_bilstm_video_best.pt`: `['state_dict', 'epoch', 'val_f1']`, `epoch=2`, `val_f1=0.9127`, `10` tensors (BiLSTM + `classifier` only).
  - `Xception_best.pt`: `['state_dict', 'epoch', 'val_f1', 'model_name']`, `epoch=10`, `val_f1=0.8859`, `276` tensors.
  - `EfficientNet_B4_best.pt`: `epoch=12`, `val_f1=0.8691`, `706` tensors.
  - `ResNet50_best.pt`: `epoch=11`, `val_f1=0.8058`, `320` tensors.

### 3.4 Dataset & Actual Classification Task
- **Dataset:** `KoreaPeter/faceforensics-c23` (hosted C23-derived frame images, `train` split only; `max_per_class: 3000`, `max_frames_per_video: 12`).
- **Splits (`GroupShuffleSplit` on `ori_vid`):** `train`: 4,191 frames; `validation`: 887 frames; `test`: 922 frames.
- **Task:** Binary classification (`0 = REAL`, `1 = FAKE`), with optional selective prediction (`REAL` / `FAKE` / `UNCERTAIN`).

### 3.5 Input Preprocessing & Inference Requirements
- No preprocessing script is included.
- `timm` backbones use `timm.data.resolve_model_data_config` / `create_transform` (typically 299x299 for Xception, 380x380 for EfficientNet-B4, 224x224 for ResNet50).
- Video-level inference requires extracting up to `temporal_frames: 8` frames per video, passing through `Xception_best.pt` (2048-d features), then `xception_bilstm_video_best.pt`.
- Training used hosted extracted/cropped frames; no face detector or video decoder is included.

### 3.6 Required Dependencies & Python Versions
- **Verified:** `torch`, `timm`, `numpy`, `Pillow` (and `opencv-python`/`decord` needed if decoding raw video).
- **Missing:** No `requirements.txt` or Python version pin in this folder.

### 3.7 Class Labels & Output Format
- `0 = REAL` (`source: original`), `1 = FAKE` (`FaceSwap`, `Deepfakes`, etc.).
- Output: 2-class logits `(batch, 2)`. Validation-selected threshold for `xception_bilstm_video_best.pt` is `0.25` (and `0.56` for `SoftVoting_Frame_Ensemble`).

### 3.8 Available Evaluation Metrics (Test Set)
- **Video-level (`Xception feature + BiLSTM`, threshold `0.25`):** Accuracy `0.9163`, F1 Macro `0.8942`, ROC-AUC `0.9619`, PR-AUC `0.9742`, MCC `0.7910`.
- **Frame-level (`SoftVoting_Frame_Ensemble`, threshold `0.56`):** Accuracy `0.8709`, F1 Macro `0.8709`, ROC-AUC `0.9464`, PR-AUC `0.9493`, MCC `0.7419`.
- **Single Frame Models (Accuracy / ROC-AUC):** `EfficientNet_B4` (`0.8633` / `0.9430`), `Xception` (`0.8547` / `0.9241`), `ResNet50` (`0.7375` / `0.8327`).

### 3.9 Existing Prediction / Inference Code
- **None** (no `.py` or `.ipynb` files in the directory).

### 3.10 Additional Files Required for Inference
- `config.json` (hyperparameters & `timm` backbone names).
- Both `Xception_best.pt` and `xception_bilstm_video_best.pt` for video-level inference.

### 3.11 CPU / GPU Requirements
- *Assumed:* Inference can run on CPU (~102 MB total for Xception + BiLSTM), though GPU is recommended for multi-frame video throughput.

### 3.12 Missing Files, Inconsistencies & Integration Blockers
- **Blocker:** No `.py`/`.ipynb` model definition or inference code; the BiLSTM wrapper must be reconstructed from `config.json` and checkpoint keys.
- **Blocker:** No raw-video frame sampler or face detector/cropper included.
- **Pathing:** `config.json` and `final_run_summary.json` contain hardcoded Colab paths (`/content/drive/MyDrive/...`).

---

## Final Readiness Table

| Model Pipeline | Checkpoint Availability | Architecture Availability | Preprocessing Availability | Inference Readiness | Missing Dependencies | Integration Readiness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fakeddit (Text)** (`M1_text_best.pt`) | **Available** (raw `OrderedDict`, ~476 MB) | **Missing** (no `.py`/`.ipynb`; must reconstruct `encoder` + `proj` + `head`) | **Partial** (`text_norm` example in CSV; no tokenizer files or script) | **Not Ready** (needs wrapper class + `roberta-base` config/tokenizer) | `transformers`, `torch`, `roberta-base` HF files | **Blocked** (needs model wrapper + HF tokenizer/config) |
| **SNLI-VE (Image-Text)** (`blip-best.pt`) | **Partial** (trainable params only, ~145 MB; needs HF base model) | **Available** (`Branch` class in `executed-notebook-code.py`) | **Available** (`BlipProcessor` + `collator` in notebook script) | **Not Ready** (script hardcodes Kaggle paths & `.cuda()`) | `Salesforce/blip-itm-base-coco` HF weights (internet or cache) | **Blocked** (needs standalone CPU/GPU wrapper + HF cache) |
| **FaceForensics++ (Video)** (`xception_bilstm_video_best.pt` + `Xception_best.pt`) | **Available** (complete `.pt` dicts, ~18 MB + ~80 MB) | **Partial** (`timm` backbones in `config.json`; BiLSTM wrapper missing) | **Missing** (no video decoder, frame sampler, or face cropper) | **Not Ready** (no `.py`/`.ipynb` inference code) | `requirements.txt` missing; needs `timm`, `torch`, `opencv-python`/`decord` | **Blocked** (needs BiLSTM wrapper + video/face preprocessing) |
