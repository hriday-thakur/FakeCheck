"""Model registry tracking checkpoint existence separately from inference readiness."""

from dataclasses import dataclass
from pathlib import Path
from app.config import settings


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    pipeline: str
    name: str
    is_primary: bool
    checkpoint_rel_paths: tuple[str, ...]
    modes: tuple[str, ...]
    inference_ready: bool = False  # Phase 1: inference not implemented yet

    def checkpoint_exists(self, models_dir: Path = settings.models_dir) -> bool:
        return all((models_dir / rel).is_file() for rel in self.checkpoint_rel_paths)

    def to_status_dict(self, models_dir: Path = settings.models_dir) -> dict:
        return {
            "model_id": self.model_id,
            "pipeline": self.pipeline,
            "name": self.name,
            "is_primary": self.is_primary,
            "checkpoint_exists": self.checkpoint_exists(models_dir),
            "inference_ready": self.inference_ready,
            "modes": list(self.modes),
        }


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # Fakeddit
    "fk_m3_clip": ModelSpec(
        model_id="fk_m3_clip",
        pipeline="fakeddit",
        name="M3_clip (CLIP Dual-Encoder)",
        is_primary=True,
        checkpoint_rel_paths=("text/original/fakeddit_leakage_controlled_run/checkpoints/M3_clip_best.pt",),
        modes=("multimodal",),
        inference_ready=True,
    ),
    "fk_m1_text": ModelSpec(
        model_id="fk_m1_text",
        pipeline="fakeddit",
        name="M1_text (RoBERTa-base)",
        is_primary=False,
        checkpoint_rel_paths=("text/original/fakeddit_leakage_controlled_run/checkpoints/M1_text_best.pt",),
        modes=("text-only",),
    ),
    "fk_m2_image": ModelSpec(
        model_id="fk_m2_image",
        pipeline="fakeddit",
        name="M2_image (ConvNeXt-Tiny)",
        is_primary=False,
        checkpoint_rel_paths=("text/original/fakeddit_leakage_controlled_run/checkpoints/M2_image_best.pt",),
        modes=("image-only",),
    ),
    "fk_m4_vilt": ModelSpec(
        model_id="fk_m4_vilt",
        pipeline="fakeddit",
        name="M4_vilt (ViLT)",
        is_primary=False,
        checkpoint_rel_paths=("text/original/fakeddit_leakage_controlled_run/checkpoints/M4_vilt_best.pt",),
        modes=("multimodal",),
    ),
    # SNLI-VE
    "snli_blip": ModelSpec(
        model_id="snli_blip",
        pipeline="snlive",
        name="BLIP (Salesforce/blip-itm-base-coco)",
        is_primary=True,
        checkpoint_rel_paths=("image/original/SNLI_VE_COMPLETE_PROJECT/blip-best.pt",),
        modes=("multimodal",),
        inference_ready=True,
    ),
    "snli_vilt": ModelSpec(
        model_id="snli_vilt",
        pipeline="snlive",
        name="ViLT (dandelin/vilt-b32-mlm)",
        is_primary=False,
        checkpoint_rel_paths=("image/original/SNLI_VE_COMPLETE_PROJECT/vilt-best.pt",),
        modes=("multimodal",),
    ),
    "snli_bridgetower": ModelSpec(
        model_id="snli_bridgetower",
        pipeline="snlive",
        name="BridgeTower (BridgeTower/bridgetower-base)",
        is_primary=False,
        checkpoint_rel_paths=("image/original/SNLI_VE_COMPLETE_PROJECT/bridgetower-best.pt",),
        modes=("multimodal",),
    ),
    # FaceForensics++
    "ff_xception_bilstm": ModelSpec(
        model_id="ff_xception_bilstm",
        pipeline="faceforensics",
        name="Xception + BiLSTM (Video-level)",
        is_primary=True,
        checkpoint_rel_paths=(
            "video/original/FFPP_PJ07_FINAL_corrected/checkpoints/xception_bilstm_video_best.pt",
            "video/original/FFPP_PJ07_FINAL_corrected/checkpoints/Xception_best.pt",
        ),
        modes=("video-only",),
        inference_ready=True,
    ),
    "ff_efficientnet_b4": ModelSpec(
        model_id="ff_efficientnet_b4",
        pipeline="faceforensics",
        name="EfficientNet-B4 (Frame-level)",
        is_primary=False,
        checkpoint_rel_paths=("video/original/FFPP_PJ07_FINAL_corrected/checkpoints/EfficientNet_B4_best.pt",),
        modes=("video-only",),
    ),
    "ff_xception": ModelSpec(
        model_id="ff_xception",
        pipeline="faceforensics",
        name="Xception (Frame-level)",
        is_primary=False,
        checkpoint_rel_paths=("video/original/FFPP_PJ07_FINAL_corrected/checkpoints/Xception_best.pt",),
        modes=("video-only",),
    ),
    "ff_resnet50": ModelSpec(
        model_id="ff_resnet50",
        pipeline="faceforensics",
        name="ResNet-50 (Frame-level)",
        is_primary=False,
        checkpoint_rel_paths=("video/original/FFPP_PJ07_FINAL_corrected/checkpoints/ResNet50_best.pt",),
        modes=("video-only",),
    ),
}

PRIMARY_MODEL_BY_PIPELINE = {
    "fakeddit": "fk_m3_clip",
    "snlive": "snli_blip",
    "faceforensics": "ff_xception_bilstm",
}
