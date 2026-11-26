# Landing Page Model Selection — Browser Acceptance Test Report

**Date:** 2026-11-27  
**Tested URL:** `http://127.0.0.1:8000/`  
**Browser Evidence:** `C:/Users/Neji Hyuga/.gemini/antigravity/brain/da7bcf9d-e385-439e-8bd7-302a7284742f/browser_evidence/3421c26f-62b8-47c3-8f7e-b35ee22607e3/final_step_12.png`  
**Verdict:** **PASS**

---

## Root Cause & Fix
- **Root Cause:** Clicking `Explore Pipelines` (`href="#home-pipelines"`) previously mutated `location.hash` to `#home-pipelines`, causing subsequent refreshes to jump to `#home-pipelines`.
- **Fix (`frontend/index.html`):** Intercepted `#explorePipelinesBtn` click with `e.preventDefault(); $('home-pipelines').scrollIntoView({ behavior: 'smooth' });` and set `history.scrollRestoration = 'manual'` + `window.scrollTo(0, 0)` on initial load.

---

## Browser Test Results
| Test Case | Status | Observed Result |
| :--- | :--- | :--- |
| **Fresh Load (`http://127.0.0.1:8000/`)** | **PASS** | `scrollY = 0` before & after intro dismissal; hero visible at `y = 56.4`. |
| **Explore Pipelines Click** | **PASS** | Smoothly scrolled to `#home-pipelines` (`scrollY = 944.8`); URL remained `http://127.0.0.1:8000/`. |
| **Home / Logo Click (`#brandHome`)** | **PASS** | Returned cleanly to top (`scrollY = 0`). |
| **Page Reload** | **PASS** | `scrollY = 0` at the top; no hash in URL. |
| **Browser Console Errors** | **PASS** | `0` console errors. |
