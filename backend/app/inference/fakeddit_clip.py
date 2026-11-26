"""Fakeddit M3 CLIP Dual-Encoder inference module."""

from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPProcessor

CALIBRATION_TEMP = 1.2760057303751002
LABEL_MAP = {0: "FAKE", 1: "REAL"}


class M3ClipClassifier(nn.Module):
    """Reconstructed M3_clip_best.pt architecture (404 state_dict keys)."""

    def __init__(self):
        super().__init__()
        self.clip = CLIPModel(CLIPConfig(vision_config={"patch_size": 16}))
        self.head_proj = nn.Sequential(
            nn.Linear(2049, 256),
            nn.LayerNorm(256),
        )
        self.head = nn.Linear(256, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        img_out = self.clip.get_image_features(pixel_values=pixel_values)
        txt_out = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        img_emb = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
        txt_emb = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)
        cos_sim = (img_emb * txt_emb).sum(dim=-1, keepdim=True)
        feat = torch.cat([img_emb, txt_emb, img_emb * txt_emb, torch.abs(img_emb - txt_emb), cos_sim], dim=-1)
        proj = F.gelu(self.head_proj(feat))
        return self.head(proj)


class FakedditClipPredictor:
    """Lazy-loaded CPU/GPU predictor for M3_clip_best.pt."""

    def __init__(self, checkpoint_path: Path, processor_dir: Path | None = None, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.processor_dir = processor_dir or (Path(__file__).resolve().parent.parent.parent / "models" / "clip_processor")
        self.device = torch.device(device)
        self.model: M3ClipClassifier | None = None
        self.processor: CLIPProcessor | None = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.processor_dir.is_dir():
            raise FileNotFoundError(f"CLIP processor directory not found: {self.processor_dir}")

        model = M3ClipClassifier()
        sd = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(sd, strict=True)
        model.to(self.device)
        model.eval()

        processor = CLIPProcessor.from_pretrained(str(self.processor_dir), local_files_only=True)
        self.model = model
        self.processor = processor

    @torch.no_grad()
    def predict(self, image: Image.Image, text: str, temperature: float = CALIBRATION_TEMP) -> dict:
        self.load()
        assert self.model is not None and self.processor is not None

        t0 = time.perf_counter()
        inputs = self.processor(
            text=[text],
            images=[image.convert("RGB")],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        scaled_logits = logits / max(float(temperature), 1e-6)
        probs = torch.softmax(scaled_logits, dim=-1)[0].cpu().tolist()

        prob_fake, prob_real = float(probs[0]), float(probs[1])
        pred_idx = 1 if prob_real >= prob_fake else 0
        pred_label = LABEL_MAP[pred_idx]
        confidence = max(prob_fake, prob_real)

        return {
            "prediction": pred_label,
            "scores": {"REAL": prob_real, "FAKE": prob_fake},
            "confidence": confidence,
            "model_name": "M3_clip (CLIP Dual-Encoder)",
            "architecture": "CLIP ViT-B/16 + 2049->256 Projection",
            "processing_time": round(time.perf_counter() - t0, 4),
        }
