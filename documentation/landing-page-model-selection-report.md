# Landing Page Direct Model Selection Report

**Date:** 2026-11-26  
**File Modified:** `frontend/index.html`

---

## 1. Changes Made
- Updated `#home-pipelines` to **"Choose Your Analysis"** (`Direct Model Selection`) with three keyboard-accessible (`role="button" tabindex="0"` + `Enter`/`Space` support) Neo-Brutalist cards:
  1. **Fakeddit — Misinformation Detection** (`data-launch-pipe="fakeddit"`)
  2. **SNLI-VE — Image-Text Verification** (`data-launch-pipe="snlive"`)
  3. **FaceForensics++ — Deepfake Detection** (`data-launch-pipe="faceforensics"`)
- Clicking any card or its launch button calls `selectPipeline(pipe)` and `navigateTo('analyze')`, pre-selecting the corresponding pipeline and primary model (`fk_m3_clip`, `snli_blip`, `ff_xception_bilstm`) while preserving the existing model switcher.
- **F4 Compatibility:** The Fakeddit card routes by pipeline key (`fakeddit`), so swapping the backend primary Fakeddit predictor to F4 later requires zero landing-page changes.
- **Hero Action:** Removed the standalone `Analyze Media` button from the hero and promoted `Explore Pipelines` (`#home-pipelines`) as the primary landing-page action.

---

## 2. Verification
- `node --check` on `frontend/index.html`: **0 errors**.
- All CSS, Neo-Brutalist styling, Light/Dark themes, and API contracts remain unchanged.
