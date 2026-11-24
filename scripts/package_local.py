"""Prepare a local, allowlisted build context; optional devkit build always uses --no-push.

There is deliberately no repository, upload, acceptance, or submission option.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from baseline.devkit import DEFAULT_DEVKIT, DEVKIT_COMMIT
from baseline.io import sha256_file, write_json

BASE_IMAGE = "python:3.12@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146"
MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
               "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model_provenance.json")


def stage(context: Path, model_path: Path) -> dict:
    if context.exists():
        raise ValueError("Local package directory already exists; choose a fresh path")
    for required in ("model.safetensors", "model_provenance.json", "config.json", "preprocessor_config.json", "vocab.txt"):
        if not (model_path / required).is_file():
            raise FileNotFoundError(f"Model file missing: {model_path / required}")
    context.mkdir(parents=True)
    (context / "baseline").mkdir()
    for filename in ("__init__.py", "io.py", "retrieval.py", "video.py", "environment.py", "run.py"):
        shutil.copy2(PROJECT / "baseline" / filename, context / "baseline" / filename)
    shutil.copy2(PROJECT / "requirements.txt", context / "requirements.txt")
    shutil.copy2(PROJECT / "AGENTS.md", context / "AGENTS.md")
    for filename in ("LICENSE", "NOTICE", "THIRD_PARTY.md"):
        shutil.copy2(PROJECT / filename, context / filename)
    shutil.copytree(PROJECT / "third_party_licenses", context / "third_party_licenses")
    (context / "model").mkdir()
    for filename in MODEL_FILES:
        path = model_path / filename
        if path.is_file():
            shutil.copy2(path, context / "model" / filename)
    (context / "run.py").write_text("from baseline.run import main\n\nif __name__ == '__main__':\n    main()\n")
    (context / "Dockerfile").write_text(f'''ARG BASE_IMAGE={BASE_IMAGE}
FROM ${{BASE_IMAGE}}
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/app/.venv
COPY requirements.txt /opt/app/requirements.txt
RUN /opt/app/.venv/bin/pip install --no-cache-dir -r /opt/app/requirements.txt
COPY baseline /opt/app/baseline
COPY run.py /opt/app/run.py
COPY model /opt/app/model
WORKDIR /opt/app
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODEL_PATH=/opt/app/model DEVICE=auto
''')
    (context / "submission.yaml").write_text('''entrypoint: ["/opt/app/.venv/bin/python", "/opt/app/run.py"]
env:
  MODEL_PATH: "/opt/app/model"
  DEVICE: "auto"
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"
cuda: "PyTorch runtime; A100 sm_80 compatibility requires local verification"
notes: "Local BLIP captions + BM25 baseline; user handles all official submissions"
''')
    (context / ".dockerignore").write_text(".cascade-build\n__pycache__\nAGENTS.md\nHANDOFF.md\npackage_manifest.json\n")
    (context / "HANDOFF.md").write_text('''# Local baseline handoff

This folder is a build context, not yet a scored or verified challenge artifact.
It contains code and the pinned public BLIP model, with no challenge data or credentials.
Use the pinned CASCADE devkit's build tool with `--no-push` to create local delta layers.
Then test with the Self-Evaluation Kit, network disabled, and verify A100/sm_80 compatibility.
The user alone handles publication, Agreement acceptance, and portal submission.
External pretrained-model eligibility still needs organizer clarification.
The root project README describes evaluation and known limitations.
''')
    manifest = {
        "package_kind": "local-build-context", "official_submission_made": False,
        "devkit_commit": DEVKIT_COMMIT, "base_image": BASE_IMAGE,
        "model": json.loads((model_path / "model_provenance.json").read_text()),
        "files": {str(path.relative_to(context)): sha256_file(path)
                  for path in sorted(context.rglob("*")) if path.is_file()},
    }
    write_json(context / "package_manifest.json", manifest)
    return manifest


def local_build_command(context: Path) -> list[str]:
    return [sys.executable, str(DEFAULT_DEVKIT / "docker-prep-kit/build_submission.py"),
            "--context", str(context.resolve()), "--base-image", BASE_IMAGE, "--no-push"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/handoff/baseline-v0"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/blip-base"))
    parser.add_argument("--build", action="store_true", help="Build local delta layers with --no-push; requires Docker")
    args = parser.parse_args()
    result = stage(args.out, args.model_path)
    print(f"Prepared local context with {len(result['files'])} files at {args.out}")
    if args.build:
        subprocess.run(local_build_command(args.out), check=True)


if __name__ == "__main__":
    main()
