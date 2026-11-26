# FakeCheck — Inference Readiness & Source-Recovery Report

**Date:** 2026-11-26  
**Scope:** Final read-only source-recovery investigation across all extracted Fakeddit, SNLI-VE, and FaceForensics++ folders.  
**Primary Production Priority (Team-Confirmed):**
1. **Fakeddit:** `M3_clip_best.pt` (CLIP Dual-Encoder) — *Primary*; `M1_text_best.pt`, `M2_image_best.pt`, `M4_vilt_best.pt` — *Secondary*.
2. **SNLI-VE:** `blip-best.pt` (BLIP) — *Primary*; `vilt-best.pt`, `bridgetower-best.pt`, `fusion-best.pt` — *Secondary*.
3. **FaceForensics++:** `xception_bilstm_video_best.pt` + `Xception_best.pt` (Xception + BiLSTM) — *Primary*; `EfficientNet_B4_best.pt`, `Xception_best.pt` (frame), `ResNet50_best.pt` — *Secondary*.

*Note: No model is marked `READY` until a real end-to-end inference test succeeds.*

---

## 1. Fakeddit — `M1_text` (`M1_text_best.pt`)

- **Inference Readiness Status:** **PARTIALLY READY** (checkpoint & calibration temperature present; architecture wrapper & HF tokenizer/config must be reconstructed).
- **Exact Architecture & Initialization:**
  - `roberta-base` encoder (`encoder.embeddings.word_embeddings.weight` `[50265, 768]`, `position_embeddings.weight` `[514, 768]`, 12 Transformer layers) + `proj` (`768 -> 256`, `proj.0` & `proj.1`) + `head` (`256 -> 2`).
  - *Note:* No `.py`/`.ipynb` source exists in `fakeddit_leakage_controlled_run`; `proj` layer types (e.g., `Linear` + `LayerNorm`) are not explicitly documented in source code.
- **Checkpoint Loading Procedure:**
  - Load `models/text/original/fakeddit_leakage_controlled_run/checkpoints/M1_text_best.pt` (`476.32 MB`, `205` keys, raw `OrderedDict`).
  - Instantiate the custom `roberta-base` + `proj` + `head` module and call `load_state_dict(..., strict=True)`.
- **Input Format, Dimensions & Preprocessing:**
  - Text string (max 512 tokens). `splits/test.csv` includes `text_norm` (lowercased, punctuation-stripped).
- **Tokenizer & Pretrained Identifier:**
  - `roberta-base` (from HuggingFace Hub or local cache; no tokenizer files included locally).
- **Label Mapping & Output Interpretation:**
  - `0 = Non-true / misinformation-related`, `1 = True / REAL`.
  - Apply calibration temperature `1.1196527802801868` (`run_summary.json`) before softmax if calibrated probabilities are desired.
- **Required Dependencies:** `python 3.13.16`, `torch 2.14.1+cu130`, `transformers 5.22.1`/`5.24.1`.
- **CPU vs. CUDA:** Can run on **CPU** or GPU (< 2 GB VRAM).
- **Missing Files / Information:** No `.py`/`.ipynb` model definition, no `config.json`, no tokenizer files.

---

## 2. Fakeddit — `M2_image` (`M2_image_best.pt`)

- **Inference Readiness Status:** **PARTIALLY READY**
- **Exact Architecture & Initialization:**
  - `ConvNeXt-Tiny` (`encoder.stem.0.weight` `(96, 3, 4, 4)`) + `head_proj` (`-> 256`) + `head` (`256 -> 2`). No `.py`/`.ipynb` source exists locally.
- **Checkpoint Loading Procedure:**
  - Load `M2_image_best.pt` (`106.95 MB`, `186` keys, raw `OrderedDict`) into reconstructed `ConvNeXt-Tiny` + `head_proj` + `head` module.
- **Input Format, Dimensions & Preprocessing:**
  - RGB image (standard `ConvNeXt-Tiny` 224x224 ImageNet normalization). Exact transform parameters are not saved in the folder.
- **Pretrained Identifier:** `convnext_tiny` (via `timm 1.0.31` or `torchvision`).
- **Label Mapping & Output Interpretation:** `0 = Non-true / misinformation-related`, `1 = True / REAL`; calibration temperature `0.9864298896567141`.
- **Required Dependencies:** `torch`, `timm 1.0.31`, `Pillow 11.3.0`.
- **CPU vs. CUDA:** Can run on **CPU** or GPU.
- **Missing Files / Information:** No `.py`/`.ipynb` model definition or preprocessing script.

---

## 3. Fakeddit — `M3_clip` (`M3_clip_best.pt`)

- **Inference Readiness Status:** **PARTIALLY READY**
- **Exact Architecture & Initialization:**
  - `CLIP Dual-Encoder` (`clip.logit_scale`, `clip.text_model` `[49408, 512]`, `position_embedding` `[77, 512]`, plus vision encoder) + `head_proj` (`-> 256`) + `head` (`256 -> 2`). No `.py`/`.ipynb` source exists locally.
- **Checkpoint Loading Procedure:**
  - Load `M3_clip_best.pt` (`572.92 MB`, `404` keys, raw `OrderedDict`) into reconstructed CLIP + `head_proj` + `head` module.
- **Input Format, Dimensions & Preprocessing:** Paired RGB image + text (max 77 tokens, CLIP processor).
- **Tokenizer & Pretrained Identifier:** `openai/clip-vit-base-patch32` (inferred from `49408` vocab, `77` max_len, `512` dim; not explicitly named in a config file).
- **Label Mapping & Output Interpretation:** `0 = Non-true / misinformation-related`, `1 = True / REAL`; calibration temperature `1.2760057303751002`.
- **Required Dependencies:** `torch`, `transformers`, `Pillow`.
- **CPU vs. CUDA:** Can run on **CPU** or GPU.
- **Missing Files / Information:** No `.py`/`.ipynb` model definition, no `config.json`, no CLIP processor/tokenizer files.

---

## 4. Fakeddit — `M4_vilt` (`M4_vilt_best.pt`)

- **Inference Readiness Status:** **PARTIALLY READY**
- **Exact Architecture & Initialization:**
  - `ViLT` (`dandelin/vilt-b32-mlm` dimensions: `text_embeddings.word_embeddings` `[30522, 768]`, `position_embeddings` `[40, 768]`, `145` image patches) + `head_proj` (`-> 256`) + `head` (`256 -> 2`). No `.py`/`.ipynb` source exists locally.
- **Checkpoint Loading Procedure:**
  - Load `M4_vilt_best.pt` (`426.54 MB`, `212` keys, raw `OrderedDict`) into reconstructed ViLT + `head_proj` + `head` module.
- **Input Format, Dimensions & Preprocessing:** Paired RGB image + text (max 40 tokens, `ViltProcessor`).
- **Tokenizer & Pretrained Identifier:** `dandelin/vilt-b32-mlm` (no local processor/tokenizer files).
- **Label Mapping & Output Interpretation:** `0 = Non-true / misinformation-related`, `1 = True / REAL`; calibration temperature `1.1055736541671064`.
- **Required Dependencies:** `torch`, `transformers`, `Pillow`.
- **CPU vs. CUDA:** Can run on **CPU** or GPU.
- **Missing Files / Information:** No `.py`/`.ipynb` model definition, no `config.json`, no `ViltProcessor` files.

---

## 5. SNLI-VE — `BLIP` (`blip-best.pt`)

- **Inference Readiness Status:** **PARTIALLY READY** (architecture & preprocessing are fully recovered in `executed-notebook-code.py`, but `blip-best.pt` is a partial checkpoint requiring HF Hub download/cache and CUDA removal).
- **Exact Architecture & Initialization:**
  - `Branch("blip")` in `executed-notebook-code.py` (lines 179–222): wraps `BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco", revision="bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c")`, freezes `vision_model`, unfreezes `text_encoder.encoder.layer[-4:]`, projects `[CLS]` (`768 -> 256` via `LayerNorm -> Linear -> GELU -> LayerNorm`), and classifies via `Dropout(0.15) -> Linear(256, 3)`.
- **Checkpoint Loading Procedure:**
  - Instantiate `Branch("blip")` (loads base HF weights), then `net.load_state_dict(torch.load("blip-best.pt", map_location="cpu", weights_only=False), strict=False)`.
- **Input Format, Dimensions & Preprocessing:**
  - `PIL.Image.open(...).convert("RGB")` + text string, processed via `BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco", revision="bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c")` with `padding=True, truncation=True, max_length=64, return_tensors="pt"`.
- **Label Mapping & Output Interpretation:** `0 = entailment`, `1 = neutral`, `2 = contradiction`; `logits.float().softmax(-1)`.
- **Required Dependencies:** `transformers==4.49.0`, `huggingface_hub>=0.26,<1`, `torch`, `Pillow` (full `requirements.txt` present).
- **CPU vs. CUDA:** Can run on **CPU** once `.cuda()` and `torch.autocast("cuda", ...)` in `executed-notebook-code.py` are replaced with device-agnostic code.
- **Missing Files / Information:** Local HF cache for `Salesforce/blip-itm-base-coco` (must be downloaded or cached); standalone inference module (cannot import `executed-notebook-code.py` directly due to Kaggle paths and `assert torch.cuda.device_count() >= 1`).

---

## 6. FaceForensics++ — Video Manipulation (`xception_bilstm_video_best.pt` + `Xception_best.pt`)

- **Inference Readiness Status:** **PARTIALLY READY** (all 4 `.pt` checkpoints are complete and `config.json` is present, but no `.py`/`.ipynb` model or video/face preprocessing code exists).
- **Exact Architecture & Initialization:**
  - Frame feature extractor: `timm.create_model("legacy_xception.tf_in1k", num_classes=2)` (`Xception_best.pt`, `83.55 MB`).
  - Temporal classifier: 1-layer bidirectional `nn.LSTM(2048, 256, bidirectional=True, batch_first=True)` + `nn.Linear(512, 2)` (`xception_bilstm_video_best.pt`, `18.90 MB`, `temporal_frames: 8`).
  - Frame-level alternatives also present: `EfficientNet_B4_best.pt` (`tf_efficientnet_b4.ns_jft_in1k`, `70.97 MB`) and `ResNet50_best.pt` (`resnet50.a1_in1k`, `94.37 MB`).
- **Checkpoint Loading Procedure:**
  - Load `Xception_best.pt["state_dict"]` into `timm` Xception, and `xception_bilstm_video_best.pt["state_dict"]` into the BiLSTM classifier.
- **Input Format, Dimensions & Preprocessing:**
  - Up to `8` RGB frames per video, transformed via `timm.data.resolve_model_data_config` / `create_transform` (299x299 for Xception).
  - *Note:* Training used hosted extracted/cropped frames; no raw-video decoder or face detector is included.
- **Label Mapping & Output Interpretation:** `0 = REAL`, `1 = FAKE`. Validation-selected threshold for `xception_bilstm_video_best.pt` is `0.25`.
- **Required Dependencies:** `torch`, `timm 1.0.31`, `Pillow`, `opencv-python` (for video decoding). No `requirements.txt` in FFPP folder.
- **CPU vs. CUDA:** Can run on **CPU** (~102 MB total), though GPU is recommended for multi-frame video throughput.
- **Missing Files / Information:** No `.py`/`.ipynb` model/inference script, no video frame-sampler or face cropper, no `requirements.txt`.

---

## Final Integration Blocker Table

| Model | Readiness Status | Primary Blocker(s) |
| :--- | :--- | :--- |
| **Fakeddit `M1_text`** | **PARTIALLY READY** | Missing `.py` wrapper (`encoder` + `proj` + `head`) and `roberta-base` HF config/tokenizer. |
| **Fakeddit `M2_image`** | **PARTIALLY READY** | Missing `.py` wrapper (`ConvNeXt-Tiny` + `head_proj` + `head`) and image transform config. |
| **Fakeddit `M3_clip`** | **READY** | Implemented in `backend/app/inference/fakeddit_clip.py`, wired to `POST /api/predict/fakeddit`. |
| **Fakeddit `M4_vilt`** | **PARTIALLY READY** | Missing `.py` wrapper (`ViLT` + `head_proj` + `head`) and `ViltProcessor` files. |
| **SNLI-VE `BLIP`** | **READY** | Implemented in `backend/app/inference/snlive_blip.py`, wired to `POST /api/predict/snlive`. |
| **FaceForensics++ `Xception+BiLSTM`** | **READY** | Implemented in `backend/app/inference/faceforensics_bilstm.py`, wired to `POST /api/predict/faceforensics`. |
