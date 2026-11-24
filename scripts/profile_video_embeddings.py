"""Deferred 32-video Qwen embedding profile, never a subset validation score."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.dense import normalize
from baseline.environment import environment
from baseline.io import sha256_file, unique_rows
from baseline.video import frame_times, read_frame
from scripts.feature_retrieval import atomic_json, save_vectors

OUT = Path("runs/retrieval-next-v1/video-profile")
QUEUE = Path("runs/retrieval-next-v1/queue.json")
DOC_INSTRUCTION = "Represent this driving video for retrieving road users, ego actions, traffic controls, and temporally ordered interactions."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-queue", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    status = OUT / "status.json"
    atomic_json(status, {"stage": "queued-after-semantic-and-visual", "is_validation_score": False})
    deadline = datetime.fromisoformat("2026-10-10T01:14:02+00:00")
    if args.wait_for_queue:
        while True:
            jobs = json.loads(QUEUE.read_text())["jobs"]
            if all(j["state"] in {"complete", "failed", "timed-out"} for j in jobs):
                break
            if datetime.now(timezone.utc) >= deadline:
                atomic_json(status, {"stage": "review-window-reached", "is_validation_score": False})
                return
            time.sleep(15)
    # This profile starts only after every registered GPU job has exited.
    # Also refuse any unrelated CUDA process rather than sharing GPU memory.
    active = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if active:
        raise RuntimeError("GPU is occupied; defer the video profile")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    spec = json.loads(Path("configs/retrieval_models.json").read_text())["qwen3-video"]
    model_path = Path(spec["local_path"])
    helper = Path("vendor/qwen3-vl-embedding/qwen3_vl_embedding.py")
    import_spec = importlib.util.spec_from_file_location("_pinned_qwen3_video_embedder", helper)
    module = importlib.util.module_from_spec(import_spec)
    sys.modules[import_spec.name] = module
    import_spec.loader.exec_module(module)
    initialization = time.perf_counter()
    embedder = module.Qwen3VLEmbedder(str(model_path.resolve()), max_length=2048,
                num_frames=4, max_frames=4, total_pixels=4 * 224 * 224,
                dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    startup_seconds = time.perf_counter() - initialization
    inputs = Path("data/prepared-train-blip-v1/input")
    videos = unique_rows(inputs / "videos.jsonl", "video_id")
    ids = sorted(videos)[:32]
    vectors, rows = [], []
    began = time.perf_counter()
    for index, vid in enumerate(ids, 1):
        video = (inputs / videos[vid]["path"]).resolve()
        stamps = frame_times(video, 4)
        pixels = [read_frame(video, stamp) for stamp in stamps]
        conversations = [embedder.format_model_input(video=pixels, instruction=DOC_INSTRUCTION)]
        batch = embedder._preprocess_inputs(conversations)
        if "pixel_values_videos" not in batch or batch["pixel_values_videos"].numel() == 0:
            raise ValueError("Vision preprocessing fell back to text; profile invalid")
        visual_tokens = int(batch["video_grid_thw"].prod(dim=-1).sum().item())
        batch = {k: v.to(embedder.model.device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = embedder.forward(batch)
            vector = normalize(embedder._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"]).float().cpu().numpy())[0]
        vectors.append(vector)
        rows.append({"video_id": vid, "video_sha256": sha256_file(video), "timestamps": stamps, "visual_patch_tokens": visual_tokens})
        elapsed = time.perf_counter() - began
        atomic_json(status, {"stage": "profiling", "completed": index, "total": 32,
                            "elapsed_seconds": elapsed, "is_validation_score": False})
        print(f"Native video profile {index}/32; elapsed={elapsed:.1f}s", flush=True)
        if elapsed > 600:
            raise RuntimeError("Native video pilot exceeded ten-minute processing allowance")
    features = np.stack(vectors)
    if float(features.std(axis=0).mean()) < 1e-5:
        raise ValueError("Collapsed embeddings; profile invalid")
    save_vectors(OUT / "video-vectors.npz", features)
    elapsed = time.perf_counter() - began
    report = {"is_validation_score": False, "model": spec,
        "helper_sha256": sha256_file(helper), "helper_url": "https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda/scripts/qwen3_vl_embedding.py",
        "frames": 4, "total_pixels": 4 * 224 * 224, "max_length": 2048, "instruction": DOC_INSTRUCTION,
        "videos": rows, "startup_seconds": startup_seconds, "processing_seconds": elapsed,
        "projected_corpus_seconds": elapsed / 32 * len(videos),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "environment": environment("cuda"),
        "full_extraction_fits_one_hour": elapsed / 32 * len(videos) <= 3600,
        "full_extraction_started": False, "vectors_sha256": sha256_file(OUT / "video-vectors.npz")}
    atomic_json(OUT / "profile.json", report)
    atomic_json(status, {"stage": "complete", "is_validation_score": False,
                        "projected_corpus_seconds": report["projected_corpus_seconds"]})
    # Copy immutable report artifacts separately from the live log/status.
    import shutil
    published = OUT / "results"
    published.mkdir(exist_ok=True)
    for name in ("profile.json", "video-vectors.npz"):
        shutil.copyfile(OUT / name, published / name)
    subprocess.run([sys.executable, "scripts/spaces.py", "upload", "--source", str(published),
        "--prefix", "experiments/retrieval-next-v1/video-profile", "--receipt", str(OUT / "receipt.json")], check=True)
    print(json.dumps({k: report[k] for k in ("startup_seconds", "processing_seconds", "projected_corpus_seconds", "full_extraction_fits_one_hour")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json(OUT / "status.json", {"stage": "failed", "error_type": type(error).__name__,
                                        "is_validation_score": False})
        raise
