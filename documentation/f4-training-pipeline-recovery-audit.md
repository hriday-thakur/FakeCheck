# F4 Integration — Phase 1 Training Pipeline Recovery Audit

**Date:** 2026-11-27  
**Scope:** Read-only recovery audit for `F4_Gated_Feature_Fusion_best.pt` and its four required feature extractors (`M1_text`, `M2_image`, `M3_clip`, `M4_vilt`).  
**Executive Verdict:** **PARTIALLY READY** (All 5 checkpoints exist locally and `F4` + `M3_clip` are recovered, but **no `.py` or `.ipynb` training scripts** exist in `fakeddit_leakage_controlled_run/`, and `roberta-base` / `dandelin/vilt-b32-mlm` are **not** in the local HuggingFace cache).

---

## 1. Recovered Architectures & Preprocessing

| Component | Checkpoint & Size | Keys | Recovered Architecture | Preprocessing & Pooling |
| :--- | :--- | :--- | :--- | :--- |
| **`M1_text`** | `M1_text_best.pt` (`476.32 MB`) | `205` | `roberta-base` (`[50265, 768]`, `[514, 768]`) + `proj` (`768->256`) + `head` (`256->2`) | Tokenized text (max `512`); `splits/test.csv` shows `text_norm` (lowercased/stripped). *Exact pooling (`[CLS]` vs mean) unverified without `.py` source.* |
| **`M2_image`** | `M2_image_best.pt` (`106.95 MB`) | `186` | `ConvNeXt-Tiny` (`stem.0.weight` `(96,3,4,4)`) + `head_proj` (`->256`) + `head` (`256->2`) | RGB image (`224x224` ImageNet norm). *Exact `timm` vs `torchvision` transform unverified without `.py` source.* |
| **`M3_clip`** | `M3_clip_best.pt` (`572.92 MB`) | `404` | `CLIP ViT-B/16` + `head_proj` (`2049->256`) + `head` (`256->2`) | **Already implemented** in `backend/app/inference/fakeddit_clip.py` (`strict=True` verified). Can expose 256-d `proj` output before `self.head`. |
| **`M4_vilt`** | `M4_vilt_best.pt` (`426.54 MB`) | `212` | `ViLT` (`dandelin/vilt-b32-mlm`, `[30522, 768]`, `[40, 768]`, `145` patches) + `head_proj` (`->256`) + `head` (`256->2`) | `ViltProcessor` (max `40` text tokens + RGB image). |
| **`F4_Gated_Feature_Fusion`** | `F4_Gated_Feature_Fusion_best.pt` (`0.55 MB`) | `24` | `projs.0..3` (`Linear(256,128)->LayerNorm(128)`), `gate_net` (`Linear(512,4)`), `classifier` (`Linear(128,64)->LayerNorm(64)->Linear(64,2)`) | Expects `4 x 256-d` embeddings in exact order `['M1_text', 'M2_image', 'M3_clip', 'M4_vilt']`. Calibration temp: `0.9457072192113133` (`run_summary.json`). |

---

## 2. Local Asset Availability

| Asset | Status | Notes |
| :--- | :--- | :--- |
| `openai/clip-vit-base-patch16` | **Available** | Cached in `backend/models/clip_processor/` and `~/.cache/huggingface/hub/`. |
| `roberta-base` | **Missing** | Not in `~/.cache/huggingface/hub/` (weights are in `M1_text_best.pt`, but tokenizer/config must be cached). |
| `dandelin/vilt-b32-mlm` | **Missing** | Not in `~/.cache/huggingface/hub/` (weights are in `M4_vilt_best.pt`, but `ViltProcessor` must be cached). |

---

## 3. Benchmark Provenance (`E1_baselines_vs_multimodal.csv`)
- `F4` (`0.9129` Acc, `0.9109` Macro F1, `0.9728` ROC-AUC) and `M3_clip` (`0.9007` Acc, `0.8991` Macro F1, `0.9621` ROC-AUC) were evaluated on the **same `n = 4,453` held-out test split** (`run_summary.json` & `F4_Gated_Feature_Fusion_best.pt` `training_protocol`: 5-fold validation CV, final refit on validation embeddings, held-out test embeddings used only for sanity check).

---

## 4. Exact Next Implementation Steps
1. Cache `roberta-base` and `dandelin/vilt-b32-mlm` tokenizers/processors locally (without downloading model weights if avoidable).
2. Implement `M1_text`, `M2_image`, and `M4_vilt` 256-d feature extractors in `backend/app/inference/` and verify `strict=True` loading against their checkpoints.
3. Expose `extract_features()` on `M3ClipClassifier` without altering `predict()`.
4. Implement `F4FusionPredictor` and wire `fk_f4_fusion` into `backend/app/registry.py` and `POST /api/predict/fakeddit`.

---

## Phase 1.1 — Gap Closure

### Component Recovery Table

| Component | Architecture recovered | Preprocessing recovered | Checkpoint load verified | Status |
| :--- | :--- | :--- | :--- | :--- |
| **M1 (`M1_text_best.pt`)** | Partial (`roberta-base` + `proj` + `head` from `state_dict`; no `.py` source) | Partial (`text_norm` in `splits/test.csv`; exact pooling/normalization script unrecovered) | Not yet (`roberta-base` tokenizer/config not cached) | **PARTIALLY READY** |
| **M2 (`M2_image_best.pt`)** | Partial (`ConvNeXt-Tiny` + `head_proj` + `head` from `state_dict`; no `.py` source) | Unrecovered (no image transform script in repo) | Not yet (wrapper not implemented) | **PARTIALLY READY** |
| **M3 (`M3_clip_best.pt`)** | **Yes** (`backend/app/inference/fakeddit_clip.py`) | **Yes** (`backend/models/clip_processor/`) | **Yes** (`strict=True`, `404` keys verified) | **READY** |
| **M4 (`M4_vilt_best.pt`)** | Partial (`dandelin/vilt-b32-mlm` + `head_proj` + `head` from `state_dict`; no `.py` source) | Partial (`max_length=40` from `position_embeddings`; `ViltProcessor` not cached) | Not yet (`ViltProcessor` not cached) | **PARTIALLY READY** |
| **F4 (`F4_Gated_Feature_Fusion_best.pt`)** | **Yes** (`projs.0..3`, `gate_net`, `classifier` from `state_dict` + metadata) | **Yes** (expects `4 x 256-d` embeddings in order `['M1_text', 'M2_image', 'M3_clip', 'M4_vilt']`) | Metadata & `24` keys inspected (wrapper not yet implemented) | **PARTIALLY READY** |

### Conclusion
**PARTIALLY READY — SPECIFIC GAPS REMAIN**
- **Remaining Blockers:**
  1. `roberta-base` (`M1`) and `dandelin/vilt-b32-mlm` (`M4`) tokenizers/processors are not in `~/.cache/huggingface/hub/`.
  2. No `.py`/`.ipynb` training source exists in `models/text/original/fakeddit_leakage_controlled_run/` for `M1` pooling, `M2` image transforms, or `M4` pooling.
  3. Benchmark provenance (`E1_baselines_vs_multimodal.csv`) is reported on the same `n = 4,453` test split, but example-level prediction IDs are not present to independently re-verify exact row-by-row parity.

---

## Phase 1.3 — Original Training Code Recovery

- **Recursive Scan (`models/text/original/fakeddit_leakage_controlled_run`):** Searched for `*.ipynb`, `*.py`, `*.zip`, `*.tar`, `*.gz`, `*.yaml`, `*.yml` — **0 files found**.
- **Architecture & Preprocessing Recovery:** Because no `.py` or `.ipynb` training scripts are present in the downloaded folder, `M1` pooling, `M2` image transforms, and `M4` pooling remain unrecovered from source code (only `M3_clip` and `F4` are fully recovered).
- **Local Model Assets:** `roberta-base` and `dandelin/vilt-b32-mlm` remain absent from `~/.cache/huggingface/hub/`.
- **Verdict:** **PARTIALLY READY** — Provide the original training notebook/script (`.ipynb` or `.py`) for `M1`/`M2`/`M4` or authorize reconstructing them from `state_dict` shapes + caching `roberta-base` and `dandelin/vilt-b32-mlm`.
