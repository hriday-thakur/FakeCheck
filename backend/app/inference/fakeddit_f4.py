"""Fakeddit F4 Gated Feature Fusion inference module (M1 + M2 + M3 + M4 -> F4)."""

from pathlib import Path
import time
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPProcessor,
    RobertaModel,
    ViltConfig,
    ViltModel,
    ViltProcessor,
)

from app.inference.fakeddit_clip import M3ClipClassifier

F4_CALIBRATION_TEMP = 0.9457072192113133
LABEL_MAP = {0: "FAKE", 1: "REAL"}


class M1TextExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = AutoConfig.from_pretrained("roberta-base", local_files_only=True)
        self.encoder = RobertaModel(cfg)
        self.proj = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256))
        self.head = nn.Linear(256, 2)

    def extract_256(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.gelu(self.proj(pooled))


class M2ImageExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
        self.head_proj = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256))
        self.head = nn.Linear(256, 2)

    def extract_256(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(pixel_values)
        return F.gelu(self.head_proj(feats))


class M4ViltExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = ViltConfig.from_pretrained("dandelin/vilt-b32-mlm", local_files_only=True)
        self.encoder = ViltModel(cfg)
        self.head_proj = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256))
        self.head = nn.Linear(256, 2)

    def extract_256(self, **inputs) -> torch.Tensor:
        out = self.encoder(**inputs).pooler_output
        return F.gelu(self.head_proj(out))


class F4GatedFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128)) for _ in range(4)])
        self.gate_net = nn.Linear(512, 4)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(64, 2),
        )

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        ps = [torch.relu(p(x)) for p, x in zip(self.projs, xs)]
        gates = torch.softmax(self.gate_net(torch.cat(ps, dim=-1)), dim=-1)
        fused = sum(gates[:, i : i + 1] * ps[i] for i in range(4))
        return self.classifier(fused)


def extract_m3_256(m3: M3ClipClassifier, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
    img_out = m3.clip.get_image_features(pixel_values=pixel_values)
    txt_out = m3.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    img_emb = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
    txt_emb = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    cos_sim = (img_emb * txt_emb).sum(dim=-1, keepdim=True)
    feat = torch.cat([img_emb, txt_emb, img_emb * txt_emb, torch.abs(img_emb - txt_emb), cos_sim], dim=-1)
    return F.gelu(m3.head_proj(feat))


class FakedditF4Predictor:
    """Lazy-loaded CPU/GPU predictor for F4_Gated_Feature_Fusion_best.pt."""

    def __init__(
        self,
        m1_ckpt: Path,
        m2_ckpt: Path,
        m3_ckpt: Path,
        m4_ckpt: Path,
        f4_ckpt: Path,
        clip_processor_dir: Path | None = None,
        device: str = "cpu",
    ):
        self.m1_ckpt = m1_ckpt
        self.m2_ckpt = m2_ckpt
        self.m3_ckpt = m3_ckpt
        self.m4_ckpt = m4_ckpt
        self.f4_ckpt = f4_ckpt
        self.clip_processor_dir = clip_processor_dir or (
            Path(__file__).resolve().parent.parent.parent / "models" / "clip_processor"
        )
        self.device = torch.device(device)
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for p in (self.m1_ckpt, self.m2_ckpt, self.m3_ckpt, self.m4_ckpt, self.f4_ckpt):
            if not p.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {p}")

        self.m1 = M1TextExtractor()
        self.m1.load_state_dict(torch.load(self.m1_ckpt, map_location=self.device, weights_only=False), strict=True)
        self.m1.to(self.device).eval()

        self.m2 = M2ImageExtractor()
        self.m2.load_state_dict(torch.load(self.m2_ckpt, map_location=self.device, weights_only=False), strict=True)
        self.m2.to(self.device).eval()

        self.m3 = M3ClipClassifier()
        self.m3.load_state_dict(torch.load(self.m3_ckpt, map_location=self.device, weights_only=False), strict=True)
        self.m3.to(self.device).eval()

        self.m4 = M4ViltExtractor()
        self.m4.load_state_dict(torch.load(self.m4_ckpt, map_location=self.device, weights_only=False), strict=True)
        self.m4.to(self.device).eval()

        self.f4 = F4GatedFusionNet()
        f4_raw = torch.load(self.f4_ckpt, map_location=self.device, weights_only=False)
        self.f4.load_state_dict(f4_raw["state_dict"], strict=True)
        self.f4.to(self.device).eval()

        self.m1_tok = AutoTokenizer.from_pretrained("roberta-base", local_files_only=True)
        data_cfg = timm.data.resolve_model_data_config(self.m2.encoder)
        self.m2_transform = timm.data.create_transform(**data_cfg, is_training=False)
        self.m3_proc = CLIPProcessor.from_pretrained(str(self.clip_processor_dir), local_files_only=True)
        self.m4_proc = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm", local_files_only=True)
        self._loaded = True

    def predict(self, image: Image.Image, text: str, temperature: float = F4_CALIBRATION_TEMP) -> dict:
        self.load()
        t0 = time.perf_counter()
        rgb = image.convert("RGB")

        with torch.inference_mode():
            m1_in = self.m1_tok([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
            e1 = self.m1.extract_256(m1_in["input_ids"].to(self.device), m1_in["attention_mask"].to(self.device))

            m2_in = self.m2_transform(rgb).unsqueeze(0).to(self.device)
            e2 = self.m2.extract_256(m2_in)

            m3_in = self.m3_proc(text=[text], images=[rgb], padding=True, truncation=True, max_length=77, return_tensors="pt")
            e3 = extract_m3_256(
                self.m3,
                m3_in["input_ids"].to(self.device),
                m3_in["attention_mask"].to(self.device),
                m3_in["pixel_values"].to(self.device),
            )

            m4_in = self.m4_proc(text=[text], images=[rgb], padding=True, truncation=True, max_length=40, return_tensors="pt")
            m4_in = {k: v.to(self.device) for k, v in m4_in.items()}
            e4 = self.m4.extract_256(**m4_in)

            logits = self.f4([e1, e2, e3, e4])
            probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)[0].cpu().tolist()

        prob_fake, prob_real = float(probs[0]), float(probs[1])
        pred_idx = 1 if prob_real >= prob_fake else 0
        return {
            "prediction": LABEL_MAP[pred_idx],
            "scores": {"REAL": prob_real, "FAKE": prob_fake},
            "confidence": max(prob_fake, prob_real),
            "model_name": "F4 Gated Feature Fusion (M1+M2+M3+M4)",
            "architecture": "Gated Feature Fusion (RoBERTa + ConvNeXt-Tiny + CLIP + ViLT -> 128-d Gate)",
            "processing_time": round(time.perf_counter() - t0, 4),
        }
