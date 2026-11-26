"""SNLI-VE BLIP Visual Entailment inference module."""

from pathlib import Path
import time
import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipForImageTextRetrieval, BlipProcessor

BLIP_REPO = "Salesforce/blip-itm-base-coco"
BLIP_REVISION = "bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c"
LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


class BlipBranch(nn.Module):
    """Exact reconstruction of Branch('blip') from executed-notebook-code.py."""

    def __init__(self, local_files_only: bool = True):
        super().__init__()
        self.base = BlipForImageTextRetrieval.from_pretrained(
            BLIP_REPO, revision=BLIP_REVISION, local_files_only=local_files_only
        )
        self.base.requires_grad_(False)
        for block in self.base.text_encoder.encoder.layer[-4:]:
            block.requires_grad_(True)
        self.project = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(256, 3),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor):
        vision = self.base.vision_model(pixel_values=pixel_values).last_hidden_state
        mask = torch.ones(vision.shape[:2], device=vision.device, dtype=torch.long)
        raw = self.base.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=vision,
            encoder_attention_mask=mask,
            return_dict=True,
        ).last_hidden_state[:, 0]
        z = self.project(raw)
        return self.head(z), z


class SnliBlipPredictor:
    """Lazy-loaded CPU/GPU predictor for blip-best.pt."""

    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model: BlipBranch | None = None
        self.processor: BlipProcessor | None = None
        self.load_report: dict | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        model = BlipBranch(local_files_only=True)
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)

        if unexpected:
            raise RuntimeError(f"Unexpected keys in blip-best.pt: {unexpected}")

        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        missing_trainable = [k for k in missing if k in trainable_names]
        if missing_trainable:
            raise RuntimeError(f"Missing trainable parameters in blip-best.pt: {missing_trainable}")

        model.to(self.device)
        model.eval()

        processor = BlipProcessor.from_pretrained(BLIP_REPO, revision=BLIP_REVISION, local_files_only=True)
        self.model = model
        self.processor = processor
        self.load_report = {
            "loaded_keys": len(state),
            "missing_frozen_keys": len(missing),
            "unexpected_keys": len(unexpected),
        }

    @torch.no_grad()
    def predict(self, image: Image.Image, hypothesis: str) -> dict:
        self.load()
        assert self.model is not None and self.processor is not None

        t0 = time.perf_counter()
        inputs = self.processor(
            text=[hypothesis],
            images=[image.convert("RGB")],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)

        logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()

        scores = {
            "entailment": float(probs[0]),
            "neutral": float(probs[1]),
            "contradiction": float(probs[2]),
        }
        pred_idx = max(range(3), key=lambda i: probs[i])
        pred_label = LABEL_MAP[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "prediction": pred_label,
            "scores": scores,
            "confidence": confidence,
            "model_name": "BLIP (Salesforce/blip-itm-base-coco)",
            "architecture": "BlipForImageTextRetrieval + 768->256->3 Branch",
            "processing_time": round(time.perf_counter() - t0, 4),
        }
