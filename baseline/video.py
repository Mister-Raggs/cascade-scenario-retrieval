from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path


def frame_times(video_path: Path, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("Frame count must be positive")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video_path)],
        check=True, capture_output=True, text=True,
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"Invalid video duration: {video_path}")
    return [duration * (index + 0.5) / count for index in range(count)]


def read_frame(video_path: Path, timestamp: float):
    from PIL import Image

    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(timestamp), "-i", str(video_path),
         "-frames:v", "1", "-vf", "scale=640:-2", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
        check=True, capture_output=True,
    )
    if not result.stdout:
        raise ValueError(f"No frame at {timestamp:.3f}s in {video_path}")
    with Image.open(io.BytesIO(result.stdout)) as frame:
        return frame.convert("RGB")


class BlipCaptioner:
    def __init__(self, model_path: Path, device: str = "cpu"):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        if not model_path.is_dir():
            raise FileNotFoundError(f"Local model missing: {model_path}; run scripts/download_model.py")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.manual_seed(0)
        self.torch, self.device = torch, device
        self.processor = BlipProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True,
        ).to(device).eval()

    def describe(self, frames: list) -> list[str]:
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=40, do_sample=False, num_beams=1)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)
