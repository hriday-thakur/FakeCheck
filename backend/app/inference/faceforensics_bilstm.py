"""FaceForensics++ Xception + BiLSTM video manipulation inference module."""

import os
from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
from PIL import Image
import timm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

LABEL_MAP = {0: "REAL", 1: "FAKE"}
MAX_TEMPORAL_FRAMES = 8
PREPROCESSING_WARNING = (
    "Original upstream face-cropping and raw-video sampling pipeline could not be recovered; "
    "inference uses uniformly sampled full RGB frames."
)


class XceptionBiLSTMClassifier(nn.Module):
    """Reconstructed Xception (2048-d) + 1-layer BiLSTM (hidden=256) + Linear(512, 2)."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(512, 2)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n shape: (2, B, 256) -> concat forward & backward final hidden states -> (B, 512)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.classifier(h_cat)


def sample_mp4_frames(video_bytes: bytes, max_frames: int = MAX_TEMPORAL_FRAMES) -> list[Image.Image]:
    """Decode MP4 bytes with OpenCV and uniformly sample up to max_frames RGB PIL Images."""
    if not video_bytes:
        raise ValueError("Empty video file.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open MP4 video stream.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Fallback: read sequentially
            raw_frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                raw_frames.append(frame)
            cap.release()
            if not raw_frames:
                raise ValueError("MP4 video contains 0 readable frames.")
            n = min(len(raw_frames), max_frames)
            indices = np.linspace(0, len(raw_frames) - 1, num=n, dtype=int)
            return [Image.fromarray(cv2.cvtColor(raw_frames[i], cv2.COLOR_BGR2RGB)) for i in indices]

        n = min(total_frames, max_frames)
        indices = np.linspace(0, total_frames - 1, num=n, dtype=int)
        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        if not frames:
            raise ValueError("MP4 video contains no readable frames.")
        return frames
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


class FaceForensicsPredictor:
    """Lazy-loaded CPU/GPU predictor for Xception_best.pt + xception_bilstm_video_best.pt."""

    def __init__(self, xception_ckpt: Path, bilstm_ckpt: Path, device: str = "cpu"):
        self.xception_ckpt = xception_ckpt
        self.bilstm_ckpt = bilstm_ckpt
        self.device = torch.device(device)
        self.backbone: nn.Module | None = None
        self.temporal: XceptionBiLSTMClassifier | None = None
        self.transform = None

    def load(self) -> None:
        if self.backbone is not None and self.temporal is not None and self.transform is not None:
            return
        if not self.xception_ckpt.is_file():
            raise FileNotFoundError(f"Xception checkpoint not found: {self.xception_ckpt}")
        if not self.bilstm_ckpt.is_file():
            raise FileNotFoundError(f"BiLSTM checkpoint not found: {self.bilstm_ckpt}")

        backbone = timm.create_model("legacy_xception.tf_in1k", pretrained=False, num_classes=2)
        x_ckpt = torch.load(self.xception_ckpt, map_location=self.device, weights_only=False)
        backbone.load_state_dict(x_ckpt["state_dict"], strict=True)
        backbone.to(self.device)
        backbone.eval()

        temporal = XceptionBiLSTMClassifier()
        b_ckpt = torch.load(self.bilstm_ckpt, map_location=self.device, weights_only=False)
        temporal.load_state_dict(b_ckpt["state_dict"], strict=True)
        temporal.to(self.device)
        temporal.eval()

        data_cfg = timm.data.resolve_model_data_config(backbone)
        self.transform = timm.data.create_transform(**data_cfg, is_training=False)
        self.backbone = backbone
        self.temporal = temporal

    @torch.no_grad()
    def predict(self, video_bytes: bytes) -> dict:
        self.load()
        assert self.backbone is not None and self.temporal is not None and self.transform is not None

        t0 = time.perf_counter()
        frames = sample_mp4_frames(video_bytes, max_frames=MAX_TEMPORAL_FRAMES)
        tensors = torch.stack([self.transform(f) for f in frames], dim=0).to(self.device)  # (T, 3, 299, 299)

        feats = self.backbone.forward_features(tensors)
        pooled = self.backbone.forward_head(feats, pre_logits=True)  # (T, 2048)
        seq = pooled.unsqueeze(0)  # (1, T, 2048)
        lengths = torch.tensor([len(frames)], dtype=torch.long)

        logits = self.temporal(seq, lengths)  # (1, 2)
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()

        prob_real, prob_fake = float(probs[0]), float(probs[1])
        pred_idx = 1 if prob_fake >= prob_real else 0
        pred_label = LABEL_MAP[pred_idx]
        confidence = max(prob_real, prob_fake)

        return {
            "prediction": pred_label,
            "scores": {"REAL": prob_real, "FAKE": prob_fake},
            "confidence": confidence,
            "model_name": "Xception + BiLSTM (Video-level)",
            "architecture": "legacy_xception.tf_in1k (2048-d) + BiLSTM (256x2) + Linear(512, 2)",
            "processing_time": round(time.perf_counter() - t0, 4),
            "frames_sampled": len(frames),
        }
