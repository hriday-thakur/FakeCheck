# F4 Phase 1.5 — Deep Experiment Metadata Recovery Report

**Date:** 2026-11-27  
**Scope:** Read-only inspection of `.npy`, `.npz`, `.pt`, `.json`, `.csv`, `.md`, `.txt`, `.html`, `.tex` in `models/text/original/fakeddit_leakage_controlled_run/`.

---

## 1. Numerical Artifact Findings (`embeddings/`)
- **`M1_text_test_{embeddings,ids,labels,logits}.npy`:** `(4453, 256) float32`, `(4453,) <U7`, `(4453,) int64`, `(4453, 2) float32`.
- **`M2_image_test_{embeddings,ids}.npy`:** `(4453, 256) float32`, `(4453,) <U7`.
- **`E3a_frozen_pretrained_features.npz`:** `tr_M1 (21000, 768)`, `tr_M2 (21000, 768)`, `tr_M3 (21000, 2049)`, `tr_M4 (21000, 768)`, `tr_cos (21000,)`, `va_M1 (4451, 768)`.

---

## 2. Architecture & Preprocessing Recovery
| Component | Verified from Metadata / Shapes | Not Recoverable (No `.py`/`.ipynb`) |
| :--- | :--- | :--- |
| **`M1_text`** | `roberta-base` (`768`-d -> `proj` `256`-d -> `head` `2`-d); `text_norm` in `splits/test.csv`. | Exact pooling (`[CLS]` vs mean) & text-cleaning regex. |
| **`M2_image`** | `ConvNeXt-Tiny` (`768`-d -> `head_proj` `256`-d -> `head` `2`-d). | Exact `timm` vs `torchvision` resize/crop/norm transform. |
| **`M3_clip`** | **Fully recovered** (`backend/app/inference/fakeddit_clip.py`, `2049`-d -> `256`-d). | — |
| **`M4_vilt`** | `dandelin/vilt-b32-mlm` (`768`-d -> `head_proj` `256`-d -> `head` `2`-d, max `40` tokens). | Exact `ViltProcessor` kwargs. |
| **`F4`** | **Fully recovered** (`F4_Gated_Feature_Fusion_best.pt`: `projs.0..3` `256->128`, `gate_net` `512->4`, `classifier` `128->64->2`, order `['M1_text', 'M2_image', 'M3_clip', 'M4_vilt']`). | — |

---

## 3. Benchmark Provenance Findings
- `M1`–`M4` test embeddings, IDs (`(4453,) <U7`), and labels (`(4453,) int64`) confirm identical `n = 4,453` test-set evaluation across `M3_clip` (`0.9007` Acc) and `F4` (`0.9129` Acc) in `E1_baselines_vs_multimodal.csv`.

---

## 4. Remaining Blockers & Recommended Next Step
- **Blockers:** `roberta-base` and `dandelin/vilt-b32-mlm` are not cached locally, and no `.py`/`.ipynb` training code exists for `M1`/`M2`/`M4`.
- **Recommended Next Step:** Keep `fk_m3_clip` as the production Fakeddit primary model unless the user authorizes reconstructing `M1`/`M2`/`M4` from `state_dict` shapes and caching `roberta-base` + `dandelin/vilt-b32-mlm`.
