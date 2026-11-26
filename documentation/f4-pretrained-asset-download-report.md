# F4 Phase 1.8 — Pretrained Asset Download Report

**Date:** 2026-11-27  
**Status:** **PASS** (Both `roberta-base` and `dandelin/vilt-b32-mlm` cached and verified offline with `local_files_only=True`).

---

## 1. Downloaded Assets & Verification
1. **M1 Text (`roberta-base`):**
   - **Downloaded:** `AutoTokenizer` + `AutoConfig` (no duplicate model weights; weights are in `M1_text_best.pt`).
   - **Cache Location:** `~/.cache/huggingface/hub/models--roberta-base`
   - **Offline Load (`local_files_only=True`):** **PASS**
2. **M4 ViLT (`dandelin/vilt-b32-mlm`):**
   - **Downloaded:** `ViltProcessor` + `ViltConfig` (no duplicate model weights; weights are in `M4_vilt_best.pt`).
   - **Cache Location:** `~/.cache/huggingface/hub/models--dandelin--vilt-b32-mlm`
   - **Offline Load (`local_files_only=True`):** **PASS**
