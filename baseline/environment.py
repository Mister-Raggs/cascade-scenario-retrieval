"""Record reproducible code/software/device identity without credentials."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def environment(device: str, *, load_torch: bool = False) -> dict:
    versions = {}
    for package in ("torch", "transformers", "Pillow", "huggingface-hub", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git = {"commit": None, "dirty": None}
    if (PROJECT / ".git").exists() and shutil.which("git"):
        try:
            git["commit"] = subprocess.check_output(
                ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                text=True, timeout=10).strip()
            git["dirty"] = bool(subprocess.check_output(
                ["git", "-C", str(PROJECT), "status", "--porcelain"], text=True, timeout=10).strip())
        except (subprocess.SubprocessError, OSError):
            pass
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}
    result = {"git": git, "inference_module_hashes": code, "python": platform.python_version(),
              "system": platform.system(), "machine": platform.machine(), "packages": versions,
              "requested_device": device, "torch_cuda": None, "gpu": None, "compiled_arches": []}
    if load_torch:
        import torch
        result["torch_cuda"] = torch.version.cuda
        result["compiled_arches"] = torch.cuda.get_arch_list() if torch.version.cuda else []
        resolved = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        result["resolved_device"] = resolved
        if resolved == "cuda" and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            result["gpu"] = {"name": properties.name, "capability": list(torch.cuda.get_device_capability(0)),
                             "memory_bytes": properties.total_memory}
    if shutil.which("nvidia-smi"):
        try:
            result["nvidia_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL, timeout=10).strip().splitlines()
        except (subprocess.SubprocessError, OSError):
            result["nvidia_driver"] = None
    return result


def feature_identity(device: str) -> dict:
    info = environment(device, load_torch=True)
    # Code/git changes in retrieval must not invalidate visual features.
    identity = {key: info.get(key) for key in
                ("packages", "system", "machine", "resolved_device", "torch_cuda", "gpu", "nvidia_driver")}
    identity["visual_code_sha256"] = hashlib.sha256((Path(__file__).parent / "video.py").read_bytes()).hexdigest()
    return identity


def configuration_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
