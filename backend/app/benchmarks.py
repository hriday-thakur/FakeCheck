"""Load verified evaluation metrics from the original model evaluation files."""

import csv
from pathlib import Path
from app.config import settings


def _f(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _load_fakeddit_benchmarks(models_dir: Path) -> list[dict]:
    path = models_dir / "text/original/fakeddit_leakage_controlled_run/metrics/E1_baselines_vs_multimodal.csv"
    if not path.is_file():
        return []
    id_map = {
        "M3_clip": "fk_m3_clip",
        "M1_text": "fk_m1_text",
        "M2_image": "fk_m2_image",
        "M4_vilt": "fk_m4_vilt",
    }
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_name = row.get("model", "")
            matched_id = next((v for k, v in id_map.items() if raw_name.startswith(k)), None)
            if not matched_id:
                continue
            rows.append({
                "id": matched_id,
                "name": raw_name,
                "status": "Evaluated",
                "metrics": {
                    "accuracy": _f(row.get("accuracy")),
                    "precision": _f(row.get("macro_precision")),
                    "recall": _f(row.get("macro_recall")),
                    "macro_f1": _f(row.get("macro_f1")),
                    "roc_auc": _f(row.get("roc_auc")),
                },
            })
    return rows


def _load_snlive_benchmarks(models_dir: Path) -> list[dict]:
    path = models_dir / "image/original/SNLI_VE_COMPLETE_PROJECT/test-evaluation/snlive-four-model-test-comparison.csv"
    if not path.is_file():
        return []
    id_map = {
        "blip": "snli_blip",
        "vilt": "snli_vilt",
        "bridgetower": "snli_bridgetower",
    }
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_name = (row.get("model") or "").strip().lower()
            matched_id = id_map.get(raw_name)
            if not matched_id:
                continue
            rows.append({
                "id": matched_id,
                "name": row.get("model", ""),
                "status": "Evaluated",
                "metrics": {
                    "accuracy": _f(row.get("accuracy")),
                    "precision": None,
                    "recall": None,
                    "macro_f1": _f(row.get("macro_f1")),
                    "roc_auc": _f(row.get("macro_roc_auc_ovr")),
                },
            })
    return rows


def _load_ffpp_benchmarks(models_dir: Path) -> list[dict]:
    path = models_dir / "video/original/FFPP_PJ07_FINAL_corrected/metrics/video_level_comparison.csv"
    if not path.is_file():
        return []
    id_map = {
        "Xception feature + BiLSTM video-level": "ff_xception_bilstm",
        "Xception frame-level": "ff_xception",
    }
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_name = (row.get("model") or row.get("method") or "").strip()
            matched_id = id_map.get(raw_name)
            if not matched_id:
                continue
            rows.append({
                "id": matched_id,
                "name": raw_name,
                "status": "Evaluated",
                "metrics": {
                    "accuracy": _f(row.get("accuracy")),
                    "precision": _f(row.get("precision_macro")),
                    "recall": _f(row.get("recall_macro")),
                    "macro_f1": _f(row.get("f1_macro") or row.get("macro_f1")),
                    "roc_auc": _f(row.get("roc_auc")),
                },
            })
    return rows


def get_benchmarks(models_dir: Path = settings.models_dir) -> dict:
    return {
        "datasets": {
            "fakeddit": _load_fakeddit_benchmarks(models_dir),
            "snlive": _load_snlive_benchmarks(models_dir),
            "faceforensics": _load_ffpp_benchmarks(models_dir),
        }
    }
