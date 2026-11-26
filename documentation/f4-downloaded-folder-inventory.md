# F4 Phase 1.4 — Complete Training Folder Inventory

**Date:** 2026-11-27  
**Target:** `models/text/original/fakeddit_leakage_controlled_run`  
**Total Files:** `127`

---

## 1. Extension Breakdown
- `.npy`: `36` (saved embeddings)
- `.png`: `32` (plots/visualizations)
- `.json`: `20` (histories, summaries, reports)
- `.csv`: `19` (metrics, splits, audits)
- `.md`: `11` (generated reports)
- `.pt`: `5` (all 5 expected checkpoints)
- `.txt`: `1` (`inference_environment.txt`)
- `.npz`: `1`
- `.html`: `1`
- `.tex`: `1`
- **`.py` / `.ipynb` / archives:** **`0`**

---

## 2. Expected Checkpoints Verification
| Checkpoint | Present | Size (Bytes) |
| :--- | :--- | :--- |
| `M1_text_best.pt` | **Yes** | `499,459,780` |
| `M2_image_best.pt` | **Yes** | `112,148,299` |
| `M3_clip_best.pt` | **Yes** | `600,748,249` |
| `M4_vilt_best.pt` | **Yes** | `447,260,441` |
| `F4_Gated_Feature_Fusion_best.pt` | **Yes** | `581,585` |

---

## 3. Conclusion
The downloaded folder contains **Checkpoints and experiment metadata** (no training notebooks or Python source files).
