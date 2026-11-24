"""Offline semantic/visual feature experiments; tune on train, evaluate one winner."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.dense import fuse, normalize, rank_scores
from baseline.environment import environment
from baseline.evaluate import evaluate
from baseline.io import sha256_file, unique_rows, write_jsonl
from baseline.retrieval import BM25, caption_documents
from baseline.video import frame_times, read_frame

TASK = "Given a driving scenario query, retrieve video descriptions matching the events, road users, and spatial and temporal relationships."


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".tmp")
    part.write_text(json.dumps(value, indent=2) + "\n")
    part.replace(path)


def save_vectors(path, vectors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, vectors=np.asarray(vectors, dtype=np.float32))
    temporary.replace(path)


def load_vectors(path):
    with np.load(path, allow_pickle=False) as archive:
        value = archive["vectors"]
    if not np.isfinite(value).all():
        raise ValueError("Invalid cached embeddings")
    return value


def query_texts(inputs):
    return {split: unique_rows(path / "queries.jsonl", "query_id") for split, path in inputs.items()}


def encode_text(model, tokenizer, texts, path, *, device="cuda", batch_size=32):
    if path.exists():
        result = load_vectors(path)
        if len(result) != len(texts):
            raise ValueError("Embedding coverage mismatch")
        return result
    import torch
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(device)
        with torch.inference_mode():
            vectors = model(**batch).last_hidden_state[:, -1].float()
        chunks.append(normalize(vectors.cpu().numpy()))
        if start % (batch_size * 10) == 0:
            print(f"Encoded text {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    result = np.concatenate(chunks)
    save_vectors(path, result)
    return result


def semantic_features(captions, inputs, model_path, out):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(model_path, local_files_only=True, use_safetensors=True,
                                      dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    docs = caption_documents(captions)
    texts = sorted(set(docs.values()) | {f["text"] for row in captions.values() for f in row["frames"]})
    embeddings = encode_text(model, tokenizer, texts, out / "unique-text.npz")
    lookup = {text: embeddings[index] for index, text in enumerate(texts)}
    ids = sorted(captions)
    aggregate = np.stack([lookup[docs[vid]] for vid in ids])
    pooled = normalize(np.stack([np.mean([lookup[f["text"]] for f in captions[vid]["frames"]], axis=0) for vid in ids]))
    save_vectors(out / "aggregate.npz", aggregate)
    save_vectors(out / "pooled.npz", pooled)
    queries = {}
    for split, rows in query_texts(inputs).items():
        queries[split] = encode_text(model, tokenizer,
            [f"Instruct: {TASK}\nQuery:{row['text']}" for row in rows.values()], out / f"queries-{split}.npz")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ids, {"aggregate": aggregate, "pooled": pooled}, queries


def visual_features(videos, inputs, model_path, out, status):
    import torch
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, use_safetensors=True,
                                      dtype=torch.float32).to("cuda").eval()
    ids = sorted(videos)
    started = time.perf_counter()
    initialization_done = time.perf_counter()
    features = []
    for index, vid in enumerate(ids):
        path = out / "videos" / f"{vid}.npz"
        if path.exists():
            vectors = load_vectors(path)
        else:
            video = (inputs["train"] / videos[vid]["path"]).resolve()
            pixels = [read_frame(video, t) for t in frame_times(video, 4)]
            batch = processor(images=pixels, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                vectors = normalize(model.get_image_features(**batch).float().cpu().numpy())
            save_vectors(path, vectors)
        if vectors.shape[0] != 4:
            raise ValueError("Expected four image embeddings")
        features.append(vectors)
        if index % 25 == 0 or index == 31:
            elapsed = time.perf_counter() - initialization_done
            atomic_json(status, {"stage": "visual-features", "completed_videos": index + 1,
                                 "total_videos": len(ids), "elapsed_seconds": elapsed})
            print(f"Visual embeddings {index + 1}/{len(ids)}; elapsed={elapsed:.1f}s", flush=True)
            if index == 31:
                estimate = elapsed / 32 * len(ids)
                atomic_json(out / "profile-32.json", {"is_validation_score": False,
                    "videos": 32, "elapsed_seconds": elapsed, "projected_corpus_seconds": estimate})
                if estimate > 1800:
                    raise RuntimeError("32-video profile exceeds 30-minute extraction allowance")
    features = np.stack(features)
    save_vectors(out / "frames.npz", features)
    queries = {}
    for split, rows in query_texts(inputs).items():
        path = out / f"queries-{split}.npz"
        if path.exists():
            queries[split] = load_vectors(path)
            continue
        chunks = []
        texts = [row["text"] for row in rows.values()]
        for index in range(0, len(texts), 64):
            batch = processor(text=texts[index:index + 64], padding="max_length", truncation=True,
                              max_length=64, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                chunks.append(normalize(model.get_text_features(**batch).float().cpu().numpy()))
        queries[split] = np.concatenate(chunks)
        save_vectors(path, queries[split])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Visual feature extraction finished in {time.perf_counter() - started:.1f}s", flush=True)
    return ids, {"mean": features.mean(axis=1), "max": features}, queries


def rankings(ids, docs, query_vectors, mode):
    # Full exact ranking; no ANN service and no arbitrary candidate truncation.
    for query in query_vectors:
        scores = np.max(np.einsum("vfd,d->vf", docs, query), axis=1) if mode == "max" else docs @ query
        yield rank_scores(ids, scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("semantic", "visual"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    inputs = {split: Path(f"data/prepared-{split}-blip-v1/input") for split in ("train", "val")}
    labels = {split: Path(f"data/prepared-{split}-blip-v1/labels/qrels.jsonl") for split in inputs}
    for split, path in inputs.items():
        if json.loads((path / "provenance.json").read_text())["split"] != split:
            raise ValueError("Incorrect tuning split")
        if labels[split].resolve().is_relative_to(path.resolve()):
            raise ValueError("Labels cannot be inference input")
    videos = unique_rows(inputs["train"] / "videos.jsonl", "video_id")
    if set(videos) != set(unique_rows(inputs["val"] / "videos.jsonl", "video_id")):
        raise ValueError("Corpus mismatch")
    captions_path = Path("runs/blip-bm25-val-v1/captions.jsonl")
    captions = unique_rows(captions_path, "video_id")
    if set(captions) != set(videos):
        raise ValueError("Caption coverage mismatch")
    specs = json.loads(Path("configs/retrieval_models.json").read_text())
    spec = specs["qwen3-text" if args.kind == "semantic" else "siglip2"]
    model_path = Path(spec["local_path"])
    provenance = json.loads((model_path / "model_provenance.json").read_text())
    if provenance["spec"] != spec:
        raise ValueError("Model identity mismatch")
    identity = {"kind": args.kind, "model": provenance, "code_sha256": sha256_file(Path(__file__)),
        "dense_code_sha256": sha256_file(ROOT / "baseline/dense.py"), "seed": 0,
        "video_code_sha256": sha256_file(ROOT / "baseline/video.py"), "runtime": environment("cuda"),
        "caption_sha256": sha256_file(captions_path), "task_instruction": TASK,
        "query_hashes": {s: sha256_file(p / "queries.jsonl") for s, p in inputs.items()},
        "qrels_hashes": {s: sha256_file(p) for s, p in labels.items()},
        "video_manifest_hashes": {s: sha256_file(p / "videos.jsonl") for s, p in inputs.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "experiment_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Existing experiment identity differs; use a fresh output")
    atomic_json(manifest, identity)
    status = args.out / "status.json"
    feature_out = args.out / "features"
    feature_out.mkdir(exist_ok=True)
    feature_manifest = feature_out / "manifest.json"
    if feature_manifest.exists():
        recorded = json.loads(feature_manifest.read_text())
        for name, digest in recorded["files"].items():
            if sha256_file(feature_out / name) != digest:
                raise ValueError("Completed feature cache changed")

    def backup(path, suffix):
        if args.backup:
            subprocess.run([sys.executable, "scripts/spaces.py", "upload", "--source", str(path),
                            "--prefix", f"experiments/retrieval-next-v1/{args.kind}/{suffix}",
                            "--receipt", str(args.out / "receipts" / f"{suffix.replace('/', '-')}.json")], check=True)

    import torch
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    atomic_json(status, {"stage": "extracting-features", "kind": args.kind})
    ids, modes, query_vectors = (semantic_features(captions, inputs, model_path, feature_out)
        if args.kind == "semantic" else visual_features(videos, inputs, model_path, feature_out, status))
    if not feature_manifest.exists():
        atomic_json(feature_manifest, {"identity": identity, "video_ids": ids,
            "query_ids": {s: list(r) for s, r in query_texts(inputs).items()},
            "seconds": time.perf_counter() - started, "environment": environment("cuda"),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "files": {str(p.relative_to(feature_out)): sha256_file(p) for p in sorted(feature_out.rglob("*.npz"))}})
    backup(feature_out, "features")
    bm25 = BM25(caption_documents(captions), k1=2.0, b=0.0)
    queries = query_texts(inputs)
    rows = []

    def score(split, name, mode, weight=None):
        job = args.out / name
        metrics_path = job / "evaluation.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            if metrics["predictions_sha256"] != sha256_file(job / "predictions.jsonl"):
                raise ValueError("Completed predictions changed")
        else:
            dense_rankings = rankings(ids, modes[mode], query_vectors[split], mode)
            def predictions():
                for (qid, query), ranking in zip(queries[split].items(), dense_rankings, strict=True):
                    if weight is not None:
                        ranking = fuse(ids, ranking, bm25.rank(query["text"], len(ids)), weight)
                    yield {"query_id": qid, "video_ids": ranking}
            write_jsonl(job / "predictions.jsonl", predictions())
            metrics = evaluate(inputs[split], labels[split], job / "predictions.jsonl", metrics_path)
            atomic_json(job / "run_manifest.json", {"kind": args.kind, "mode": mode, "dense_weight": weight,
                "split": split, "top_k": len(ids), "feature_manifest_sha256": sha256_file(feature_out / "manifest.json"),
                "predictions_sha256": sha256_file(job / "predictions.jsonl")})
        backup(job, name)
        return {"name": name, "mode": mode, "dense_weight": weight,
                "average_r_precision": metrics["average_r_precision"], "map": metrics["map"], "split": split}

    for mode in modes:
        atomic_json(status, {"stage": "training-dense", "mode": mode})
        rows.append(score("train", f"train-{mode}", mode))
        atomic_json(args.out / "train_results.json", rows)
    best_dense = max(rows, key=lambda r: (r["average_r_precision"], r["map"]))
    for weight in (0.25, 0.5, 0.75):
        atomic_json(status, {"stage": "training-fusion", "dense_weight": weight})
        rows.append(score("train", f"train-fusion-{weight}", best_dense["mode"], weight))
        atomic_json(args.out / "train_results.json", rows)
    winner = max(rows, key=lambda r: (r["average_r_precision"], r["map"]))
    atomic_json(args.out / "selected_on_train.json", winner)
    atomic_json(status, {"stage": "selected-validation"})
    result = score("val", "selected-validation", winner["mode"], winner["dense_weight"])
    atomic_json(status, {"stage": "complete", "selected_on_train": winner, "validation": result})
    summary = args.out / "summary"
    summary.mkdir(exist_ok=True)
    import shutil
    for name in ("status.json", "train_results.json", "selected_on_train.json", "experiment_manifest.json"):
        shutil.copyfile(args.out / name, summary / name)
    backup(summary, "summary")
    print(json.dumps({"kind": args.kind, "selected": winner, "validation": result}), flush=True)


if __name__ == "__main__":
    main()
