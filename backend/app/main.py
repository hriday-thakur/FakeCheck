"""FastAPI backend foundation for FakeCheck (Phase 1)."""

from io import BytesIO
import threading
import uuid
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.benchmarks import get_benchmarks
from app.config import PROJECT_ROOT, settings
from app.inference.fakeddit_clip import CALIBRATION_TEMP, FakedditClipPredictor
from app.inference.fakeddit_f4 import F4_CALIBRATION_TEMP, FakedditF4Predictor
from app.inference.faceforensics_bilstm import PREPROCESSING_WARNING, FaceForensicsPredictor
from app.inference.snlive_blip import SnliBlipPredictor
from app.registry import MODEL_REGISTRY, PRIMARY_MODEL_BY_PIPELINE

app = FastAPI(title="FakeCheck API", version="0.1.0")

_cors_origins = settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=(_cors_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_VIDEO_TYPES = {"video/mp4"}

_fakeddit_clip_predictor: FakedditClipPredictor | None = None
_fakeddit_f4_predictor: FakedditF4Predictor | None = None
_snli_blip_predictor: SnliBlipPredictor | None = None
_faceforensics_predictor: FaceForensicsPredictor | None = None

_fakeddit_clip_lock = threading.Lock()
_fakeddit_f4_lock = threading.Lock()
_snli_blip_lock = threading.Lock()
_faceforensics_lock = threading.Lock()
_model_runtime_states: dict[str, str] = {}


def _get_runtime_state(model_id: str, spec) -> str:
    if not spec.inference_ready:
        return "unimplemented"
    if not spec.checkpoint_exists(settings.models_dir):
        return "missing_checkpoint"
    return _model_runtime_states.get(model_id, "not_initialized")


def get_fakeddit_clip_predictor() -> FakedditClipPredictor:
    global _fakeddit_clip_predictor
    if _fakeddit_clip_predictor is None:
        with _fakeddit_clip_lock:
            if _fakeddit_clip_predictor is None:
                spec = MODEL_REGISTRY["fk_m3_clip"]
                ckpt_path = settings.models_dir / spec.checkpoint_rel_paths[0]
                _fakeddit_clip_predictor = FakedditClipPredictor(checkpoint_path=ckpt_path)
    return _fakeddit_clip_predictor


def get_fakeddit_f4_predictor() -> FakedditF4Predictor:
    global _fakeddit_f4_predictor
    if _fakeddit_f4_predictor is None:
        with _fakeddit_f4_lock:
            if _fakeddit_f4_predictor is None:
                spec = MODEL_REGISTRY["fk_f4_gated"]
                _fakeddit_f4_predictor = FakedditF4Predictor(
                    m1_ckpt=settings.models_dir / spec.checkpoint_rel_paths[0],
                    m2_ckpt=settings.models_dir / spec.checkpoint_rel_paths[1],
                    m3_ckpt=settings.models_dir / spec.checkpoint_rel_paths[2],
                    m4_ckpt=settings.models_dir / spec.checkpoint_rel_paths[3],
                    f4_ckpt=settings.models_dir / spec.checkpoint_rel_paths[4],
                )
    return _fakeddit_f4_predictor


def get_snli_blip_predictor() -> SnliBlipPredictor:
    global _snli_blip_predictor
    if _snli_blip_predictor is None:
        with _snli_blip_lock:
            if _snli_blip_predictor is None:
                spec = MODEL_REGISTRY["snli_blip"]
                ckpt_path = settings.models_dir / spec.checkpoint_rel_paths[0]
                _snli_blip_predictor = SnliBlipPredictor(checkpoint_path=ckpt_path)
    return _snli_blip_predictor


def get_faceforensics_predictor() -> FaceForensicsPredictor:
    global _faceforensics_predictor
    if _faceforensics_predictor is None:
        with _faceforensics_lock:
            if _faceforensics_predictor is None:
                spec = MODEL_REGISTRY["ff_xception_bilstm"]
                bilstm_ckpt = settings.models_dir / spec.checkpoint_rel_paths[0]
                xception_ckpt = settings.models_dir / spec.checkpoint_rel_paths[1]
                _faceforensics_predictor = FaceForensicsPredictor(xception_ckpt=xception_ckpt, bilstm_ckpt=bilstm_ckpt)
    return _faceforensics_predictor


def _validate_model(pipeline: str, model_id: str):
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None or spec.pipeline != pipeline:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id '{model_id}' for pipeline '{pipeline}'.",
        )
    return spec


async def _validate_upload(file: UploadFile | None, allowed_types: set[str], max_mb: int | None = None):
    if file is None:
        raise HTTPException(status_code=422, detail="Required media file is missing.")
    if file.content_type and file.content_type.lower() not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Allowed: {sorted(allowed_types)}",
        )
    if max_mb is not None:
        data = await file.read()
        if len(data) > max_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {max_mb} MB limit.")


FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
def serve_frontend():
    if not FRONTEND_INDEX.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(FRONTEND_INDEX, media_type="text/html")


@app.get("/api/health")
def health():
    models_status = []
    for spec in MODEL_REGISTRY.values():
        d = spec.to_status_dict(settings.models_dir)
        state = _get_runtime_state(spec.model_id, spec)
        d["runtime_state"] = state
        d["inference_ready"] = bool(spec.inference_ready and d["checkpoint_exists"] and state != "failed")
        models_status.append(d)
    return {
        "status": "ok",
        "phase": "5c-hardened",
        "models": models_status,
    }


@app.get("/api/benchmark")
def benchmark():
    return get_benchmarks()


@app.post("/api/predict/fakeddit")
async def predict_fakeddit(
    model_id: str = Form(default=PRIMARY_MODEL_BY_PIPELINE["fakeddit"]),
    mode: str = Form(default="multimodal"),
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    xai: str = Form(default="false"),
    temperature: str = Form(default="1.276"),
    confidence_threshold: str = Form(default="0.65"),
    entropy_threshold: str = Form(default="0.72"),
):
    spec = _validate_model("fakeddit", model_id)
    if mode not in {"multimodal", "text-only", "image-only"}:
        raise HTTPException(status_code=422, detail=f"Invalid mode '{mode}' for fakeddit.")
    if mode in {"multimodal", "text-only"} and not (text and text.strip()):
        raise HTTPException(status_code=422, detail="Field 'text' is required for this mode.")
    if mode in {"multimodal", "image-only"}:
        await _validate_upload(file, ALLOWED_IMAGE_TYPES, max_mb=settings.max_image_mb)

    if spec.model_id in {"fk_f4_gated", "fk_m3_clip"} and spec.inference_ready:
        if mode != "multimodal" or file is None or not (text and text.strip()):
            raise HTTPException(status_code=422, detail=f"{spec.model_id} requires both 'file' and 'text' in multimodal mode.")
        await file.seek(0)
        raw_bytes = await file.read()
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid image file: {exc}") from exc

        default_temp = F4_CALIBRATION_TEMP if spec.model_id == "fk_f4_gated" else CALIBRATION_TEMP
        try:
            temp = float(temperature)
        except ValueError:
            temp = default_temp

        try:
            predictor = get_fakeddit_f4_predictor() if spec.model_id == "fk_f4_gated" else get_fakeddit_clip_predictor()
            result = predictor.predict(image=image, text=text.strip(), temperature=temp)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return {
            "analysis_id": f"AN-{uuid.uuid4().hex[:10].upper()}",
            **result,
            "abstained": False,
            "abstention_reason": "",
            "warnings": [],
        }

    raise HTTPException(
        status_code=503,
        detail=f"Inference for '{spec.model_id}' ({spec.name}) is not implemented in Phase 1.",
    )


@app.post("/api/predict/snlive")
async def predict_snlive(
    model_id: str = Form(default=PRIMARY_MODEL_BY_PIPELINE["snlive"]),
    mode: str = Form(default="multimodal"),
    hypothesis: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    xai: str = Form(default="false"),
    temperature: str = Form(default="1.0"),
    confidence_threshold: str = Form(default="0.60"),
    entropy_threshold: str = Form(default="0.78"),
    mismatch_heuristic: str = Form(default="false"),
):
    spec = _validate_model("snlive", model_id)
    if mode != "multimodal":
        raise HTTPException(status_code=422, detail=f"Invalid mode '{mode}' for snlive; expected 'multimodal'.")
    if not (hypothesis and hypothesis.strip()):
        raise HTTPException(status_code=422, detail="Field 'hypothesis' is required for snlive.")
    await _validate_upload(file, ALLOWED_IMAGE_TYPES, max_mb=settings.max_image_mb)

    if spec.model_id == "snli_blip" and spec.inference_ready:
        assert file is not None
        await file.seek(0)
        raw_bytes = await file.read()
        try:
            image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid image file: {exc}") from exc

        try:
            predictor = get_snli_blip_predictor()
            result = predictor.predict(image=image, hypothesis=hypothesis.strip())
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return {
            "analysis_id": f"AN-{uuid.uuid4().hex[:10].upper()}",
            **result,
            "abstained": False,
            "abstention_reason": "",
            "warnings": ["SNLI-VE predicts visual entailment, not deepfake manipulation."],
        }

    raise HTTPException(
        status_code=503,
        detail=f"Inference for '{spec.model_id}' ({spec.name}) is not implemented in Phase 1.",
    )


@app.post("/api/predict/faceforensics")
async def predict_faceforensics(
    model_id: str = Form(default=PRIMARY_MODEL_BY_PIPELINE["faceforensics"]),
    mode: str = Form(default="video-only"),
    file: UploadFile | None = File(default=None),
    xai: str = Form(default="false"),
    temperature: str = Form(default="1.0"),
    confidence_threshold: str = Form(default="0.25"),
    entropy_threshold: str = Form(default="0.68"),
):
    spec = _validate_model("faceforensics", model_id)
    if mode != "video-only":
        raise HTTPException(status_code=422, detail=f"Invalid mode '{mode}' for faceforensics; expected 'video-only'.")
    await _validate_upload(file, ALLOWED_VIDEO_TYPES, max_mb=settings.max_video_mb)

    if spec.model_id == "ff_xception_bilstm" and spec.inference_ready:
        assert file is not None
        await file.seek(0)
        video_bytes = await file.read()
        try:
            predictor = get_faceforensics_predictor()
            result = predictor.predict(video_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid MP4 video: {exc}") from exc
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return {
            "analysis_id": f"AN-{uuid.uuid4().hex[:10].upper()}",
            **result,
            "abstained": False,
            "abstention_reason": "",
            "warnings": [PREPROCESSING_WARNING],
        }

    raise HTTPException(
        status_code=503,
        detail=f"Inference for '{spec.model_id}' ({spec.name}) is not implemented in Phase 1.",
    )
