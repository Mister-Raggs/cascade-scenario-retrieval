"""Private research jobs: factorial retrieval, native video features, and grounded descriptions."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.dense import normalize, rank_scores
from baseline.environment import environment
from baseline.evaluate import evaluate
from baseline.experiment_analysis import interaction, paired_delta, prefix_diagnostic
from baseline.io import sha256_file, unique_rows, write_jsonl
from baseline.retrieval import BM25, caption_documents
from baseline.video import frame_times, read_frame
from scripts.feature_retrieval import atomic_json, load_vectors, save_vectors
from scripts.profile_video_embeddings import DOC_INSTRUCTION

RUN = Path("runs/research-queue-v1")
OLD = Path("runs/retrieval-next-v1")
INPUTS = {s: Path(f"data/prepared-{s}-blip-v1/input") for s in ("train", "val")}
LABELS = {s: p.parent / "labels/qrels.jsonl" for s, p in INPUTS.items()}
QUERY_INSTRUCTION = "Retrieve driving videos matching every described event, road user, traffic control, and spatial or temporal relation."
DESCRIPTION_PROMPT = """Describe observable driving events in these chronologically ordered front-camera frames.
Return ONLY one compact JSON object with keys summary, events, causal_hypotheses, unknowns.
events is a list of records with actor, action, target, t_start, t_end, polarity, evidence.
Use timestamps printed with the images, in seconds. Actor IDs must stay consistent.
Distinguish ego from other cars. Include traffic light colors only when visible and distinguish
lights relevant to ego's path from other signals. Preserve stops, turns, lane changes,
pedestrians, cyclists, crossings, and event order when actually supported by the frames.
Do not infer ego stopping from a single still frame or absence of action from missing frames.
Use unknowns for ambiguous motion, signal colors, or actor identity. Extra scene facts are fine.
Keep inferred causal explanations in causal_hypotheses, never assert them as observations.
Keep output under 450 tokens, using up to six events; empty events is acceptable if uncertain."""


def corpus():
    for split, path in INPUTS.items():
        if json.loads((path / "provenance.json").read_text())["split"] != split:
            raise ValueError("Training/validation split mismatch")
        if LABELS[split].resolve().is_relative_to(path.resolve()):
            raise ValueError("Labels may not be inference inputs")
    videos = unique_rows(INPUTS["train"] / "videos.jsonl", "video_id")
    if set(videos) != set(unique_rows(INPUTS["val"] / "videos.jsonl", "video_id")):
        raise ValueError("Corpus mismatch")
    queries = {s: unique_rows(p / "queries.jsonl", "query_id") for s, p in INPUTS.items()}
    return videos, queries


def identity(out, kind, extra=None):
    value = {"kind": kind, "code_sha256": sha256_file(Path(__file__)),
             "analysis_code_sha256": sha256_file(ROOT / "baseline/experiment_analysis.py"),
             "input_hashes": {s: {"queries": sha256_file(p / "queries.jsonl"),
                                   "videos": sha256_file(p / "videos.jsonl"),
                                   "qrels": sha256_file(LABELS[s])} for s, p in INPUTS.items()},
             "seed": 0, "extra": extra or {}, "environment": environment("cpu")}
    out.mkdir(parents=True, exist_ok=True)
    p = out / "experiment_manifest.json"
    if p.exists() and json.loads(p.read_text()) != value:
        raise ValueError("Changed experiment inputs/code: use a fresh run identifier")
    atomic_json(p, value)
    return value


def backup(source, suffix, receipt):
    subprocess.run([sys.executable, "scripts/spaces.py", "upload", "--source", str(source),
                    "--prefix", f"experiments/research-queue-v1/{suffix}",
                    "--receipt", str(receipt)], check=True)


def finish(out, status, *, include_features=False):
    atomic_json(out / "status.json", status)
    snapshot = out / "results"
    snapshot.mkdir(exist_ok=True)
    for p in out.glob("*.json"):
        shutil.copyfile(p, snapshot / p.name)
    for p in out.glob("*.jsonl"):
        shutil.copyfile(p, snapshot / p.name)
    backup(snapshot, f"{out.name}/results", out / "receipts/results.json")
    if include_features:
        backup(out / "features", f"{out.name}/features", out / "receipts/features.json")


def scored_predictions(out, split, name, predictions):
    job = out / name
    p = job / "predictions.jsonl"
    if (job / "evaluation.json").exists():
        result = json.loads((job / "evaluation.json").read_text())
        if result["predictions_sha256"] != sha256_file(p):
            raise ValueError("Completed prediction hash changed")
    else:
        write_jsonl(p, predictions)
        result = evaluate(INPUTS[split], LABELS[split], p, job / "evaluation.json")
    atomic_json(job / "run_manifest.json", {"split": split,
                "predictions_sha256": sha256_file(p), "parent_manifest_sha256": sha256_file(out / "experiment_manifest.json")})
    backup(job, f"{out.name}/{name}", out / f"receipts/{name}.json")
    return result


def verified_features(path, videos, queries):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["video_ids"] != sorted(videos):
        raise ValueError("Feature video order/coverage differs")
    if manifest["query_ids"] != {s: list(rows) for s, rows in queries.items()}:
        raise ValueError("Feature query order/coverage differs")
    expected = manifest.get("identity", {}).get("query_hashes")
    if expected and expected != {s: sha256_file(p / "queries.jsonl") for s, p in INPUTS.items()}:
        raise ValueError("Feature query text differs")
    for name, digest in manifest["files"].items():
        if sha256_file(path / name) != digest:
            raise ValueError("Feature file changed")
    return manifest


def reciprocal_scores(ids, ranking):
    positions = {v: i for i, v in enumerate(ids)}
    scores = np.zeros(len(ids), dtype=np.float32)
    for rank, vid in enumerate(ranking, 1):
        scores[positions[vid]] = 1 / (60 + rank)
    return scores


def factor_rankings(ids, queries, lexical, channels, factors):
    for index, (qid, query) in enumerate(queries.items()):
        scores = reciprocal_scores(ids, lexical.rank(query["text"], len(ids)))
        for factor in factors:
            documents, vectors, mode = channels[factor]
            similarities = (np.max(np.einsum("vfd,d->vf", documents, vectors[index]), axis=1)
                            if mode == "max" else documents @ vectors[index])
            scores += 3 * reciprocal_scores(ids, rank_scores(ids, similarities))
        yield {"query_id": qid, "video_ids": rank_scores(ids, scores)}


def factorial(out, with_native=False):
    videos, queries = corpus()
    ids = sorted(videos)
    text_dir, visual_dir = OLD / "semantic/features", OLD / "visual/features"
    verified_features(text_dir, videos, queries)
    verified_features(visual_dir, videos, queries)
    train_visual = json.loads((OLD / "visual/train_results.json").read_text())
    visual_mode = max((r for r in train_visual if r["dense_weight"] is None),
                      key=lambda r: (r["average_r_precision"], r["map"]))["mode"]
    parents = {"text": sha256_file(text_dir / "manifest.json"),
               "visual": sha256_file(visual_dir / "manifest.json")}
    native_dir = RUN / "native/features"
    if with_native:
        verified_features(native_dir, videos, queries)
        parents["native"] = sha256_file(native_dir / "manifest.json")
    identity(out, "factorial-native" if with_native else "factorial-text-visual",
             {"parents": parents, "visual_mode_selected_on_train": visual_mode,
              "fixed_factor_coefficients": {"B": 1, "T": 3, "V": 3, "N": 3}, "rrf_constant": 60})
    captions = unique_rows(Path("runs/blip-bm25-val-v1/captions.jsonl"), "video_id")
    lexical = BM25(caption_documents(captions), k1=2.0, b=0.0)
    recipes = {"B": (), "BT": ("T",), "BV": ("V",), "BTV": ("T", "V")}
    if with_native:
        recipes.update(BN=("N",), BTN=("T", "N"), BVN=("V", "N"), BTVN=("T", "V", "N"))
    training = []
    arrays = {}
    per_query = {}
    coverage = {}
    for name, factors in recipes.items():
        atomic_json(out / "status.json", {"stage": "training-factorial", "recipe": name})
        channels = {"T": (load_vectors(text_dir / "aggregate.npz"), load_vectors(text_dir / "queries-train.npz"), "aggregate"),
                    "V": (load_vectors(visual_dir / "frames.npz"),
                          load_vectors(visual_dir / "queries-train.npz"), visual_mode)}
        if visual_mode == "mean":
            channels["V"] = (channels["V"][0].mean(axis=1), channels["V"][1], "mean")
        if with_native:
            channels["N"] = (load_vectors(native_dir / "aggregate.npz"), load_vectors(native_dir / "queries-train.npz"), "aggregate")
        metrics = scored_predictions(out, "train", f"train-{name}", factor_rankings(ids, queries["train"], lexical, channels, factors))
        training.append({"recipe": name, "factors": ["B", *factors], "split": "train",
                         "average_r_precision": metrics["average_r_precision"], "map": metrics["map"]})
        scored = [json.loads(x) for x in (out / f"train-{name}/evaluation_per_query.jsonl").open()]
        scored = [r for r in scored if r["scored"]]
        arrays[name] = [r["r_precision"] for r in scored]
        coverage[name] = {}
        for k in (20, 50, 100):
            diagnostics = [prefix_diagnostic(r["video_ids"], r["relevant_video_ids"], k) for r in scored]
            coverage[name][str(k)] = {field: float(np.mean([row[field] for row in diagnostics]))
                       for field in ("recall", "oracle_r_precision", "prefix_cannot_change_r_precision")}
        per_query[name] = [{k: r[k] for k in ("query_id", "text", "r_precision")} for r in scored]
        del scored
        atomic_json(out / "train_results.json", training)
        del channels
        gc.collect()
    # All paired cells share the official evaluation query order and coverage.
    order = [r["query_id"] for r in per_query["B"]]
    if any([r["query_id"] for r in rows] != order for rows in per_query.values()):
        raise ValueError("Paired comparison coverage/order differs")
    comparisons = {name: paired_delta(arrays["B"], values) for name, values in arrays.items() if name != "B"}
    interactions = {"T_V": interaction(arrays["B"], arrays["BT"], arrays["BV"], arrays["BTV"])}
    if with_native:
        interactions.update(T_N=interaction(arrays["B"], arrays["BT"], arrays["BN"], arrays["BTN"]),
                            V_N=interaction(arrays["B"], arrays["BV"], arrays["BN"], arrays["BVN"]))
    slices = {"negation": lambda t: any(x in t.lower() for x in ("does not", "did not", "without", "never")),
              "causal": lambda t: any(x in t.lower() for x in ("because", "due to", "causes")),
              "temporal": lambda t: any(x in t.lower() for x in ("before", "after", "while", "during")),
              "traffic_control": lambda t: any(x in t.lower() for x in ("traffic light", "stop sign"))}
    slice_metrics = {}
    for label, predicate in slices.items():
        indices = [i for i, r in enumerate(per_query["B"]) if predicate(r["text"])]
        slice_metrics[label] = {"queries": len(indices), "definition": "Overlapping heuristic text slice, not official difficulty",
                                "r_precision": {name: float(np.mean(np.asarray(a)[indices])) if indices else None for name, a in arrays.items()}}
    diagnostics = {"split": "train", "candidate_coverage": coverage, "paired_vs_B": comparisons,
                   "factor_interactions": interactions, "heuristic_slices": slice_metrics,
                   "fixed_coefficients_required_for_factorial_interpretation": True}
    atomic_json(out / "diagnostics.json", diagnostics)
    write_jsonl(out / "per_query_deltas.jsonl", ({"query_id": qid, "r_precision": {name: float(a[i]) for name, a in arrays.items()}}
                                                for i, qid in enumerate(order)))
    winner = max(training, key=lambda r: (r["average_r_precision"], r["map"]))
    atomic_json(out / "selected_on_train.json", winner)
    # Reuse all existing validation features, but evaluate only the training-selected recipe.
    channels = {"T": (load_vectors(text_dir / "aggregate.npz"), load_vectors(text_dir / "queries-val.npz"), "aggregate"),
                "V": (load_vectors(visual_dir / "frames.npz"), load_vectors(visual_dir / "queries-val.npz"), visual_mode)}
    if visual_mode == "mean":
        channels["V"] = (channels["V"][0].mean(axis=1), channels["V"][1], "mean")
    if with_native:
        channels["N"] = (load_vectors(native_dir / "aggregate.npz"), load_vectors(native_dir / "queries-val.npz"), "aggregate")
    metrics = scored_predictions(out, "val", "selected-validation",
                 factor_rankings(ids, queries["val"], lexical, channels, recipes[winner["recipe"]]))
    finish(out, {"stage": "complete", "selected_on_train": winner,
                 "validation": {k: metrics[k] for k in ("average_r_precision", "map", "num_queries")}})


def native_embedder(frames=4, total_pixels=4 * 224 * 224):
    import torch
    helper = Path("vendor/qwen3-vl-embedding/qwen3_vl_embedding.py")
    spec = importlib.util.spec_from_file_location("_research_qwen_embedder", helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model_spec = json.loads(Path("configs/retrieval_models.json").read_text())["qwen3-video"]
    return module.Qwen3VLEmbedder(str(Path(model_spec["local_path"]).resolve()),
                max_length=2048, num_frames=frames, max_frames=frames, total_pixels=total_pixels,
                dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)


def encode_video(embedder, pixels):
    import torch
    batch = embedder._preprocess_inputs([embedder.format_model_input(video=pixels, instruction=DOC_INSTRUCTION)])
    if "pixel_values_videos" not in batch or not batch["pixel_values_videos"].numel():
        raise ValueError("Missing vision tensors; text fallback invalidates experiment")
    tokens = int(batch["video_grid_thw"].prod(dim=-1).sum().item())
    sequence_tokens = int(batch["attention_mask"].sum().item())
    batch = {k: v.to(embedder.model.device) for k, v in batch.items()}
    with torch.inference_mode():
        output = embedder.forward(batch)
        vector = normalize(embedder._pooling_last(output["last_hidden_state"], output["attention_mask"]).float().cpu().numpy())
    return vector[0], tokens, sequence_tokens


def decode(video, count):
    stamps = frame_times(video, count)
    return stamps, [read_frame(video, t) for t in stamps]


def native(out):
    import torch
    videos, queries = corpus()
    ids = sorted(videos)
    spec = json.loads(Path("configs/retrieval_models.json").read_text())["qwen3-video"]
    model_manifest = Path(spec["local_path"]) / "model_provenance.json"
    profile = json.loads((OLD / "video-profile/profile.json").read_text())
    if not profile["full_extraction_fits_one_hour"]:
        raise ValueError("Native video profile did not pass the one-hour extraction gate")
    identity(out, "native-video", {"model_provenance_sha256": sha256_file(model_manifest),
            "helper_sha256": sha256_file(Path("vendor/qwen3-vl-embedding/qwen3_vl_embedding.py")),
            "frames": 4, "total_pixels": 4 * 224 * 224, "max_length": 2048,
            "document_instruction": DOC_INSTRUCTION, "query_instruction": QUERY_INSTRUCTION,
            "frame_time_metadata": "Official helper list-input defaults; actual sampled timestamps recorded separately"})
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    embedder = native_embedder()
    features = out / "features"
    features.mkdir(exist_ok=True)
    # Reuse the checked profile's exact vectors only for identical preprocessing/model.
    if profile["model"] != spec or profile["instruction"] != DOC_INSTRUCTION or profile["frames"] != 4:
        raise ValueError("Profile recipe differs")
    if profile["helper_sha256"] != sha256_file(Path("vendor/qwen3-vl-embedding/qwen3_vl_embedding.py")):
        raise ValueError("Profile helper changed")
    if sha256_file(OLD / "video-profile/video-vectors.npz") != profile["vectors_sha256"]:
        raise ValueError("Profile vector hash differs")
    pilot = load_vectors(OLD / "video-profile/video-vectors.npz")
    for row, vector in zip(profile["videos"], pilot, strict=True):
        video = (INPUTS["train"] / videos[row["video_id"]]["path"]).resolve()
        if sha256_file(video) != row["video_sha256"]:
            raise ValueError("Profile source video changed")
        p = features / "videos" / f"{row['video_id']}.npz"
        if not p.exists():
            save_vectors(p, vector[None])
            atomic_json(p.with_suffix(".json"), row)
    vectors = []
    # One CPU video decode runs ahead of GPU inference. Bound prefetch memory to one clip.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for index, vid in enumerate(ids):
            p = features / "videos" / f"{vid}.npz"
            video = (INPUTS["train"] / videos[vid]["path"]).resolve()
            if p.exists():
                meta = json.loads(p.with_suffix(".json").read_text())
                if sha256_file(video) != meta["video_sha256"]:
                    raise ValueError("Cached video changed")
                vector = load_vectors(p)[0]
            else:
                if pending is None:
                    pending = pool.submit(decode, video, 4)
                stamps, pixels = pending.result()
                pending = None
                if index + 1 < len(ids):
                    next_id = ids[index + 1]
                    if not (features / "videos" / f"{next_id}.npz").exists():
                        next_video = (INPUTS["train"] / videos[next_id]["path"]).resolve()
                        pending = pool.submit(decode, next_video, 4)
                vector, tokens, sequence_tokens = encode_video(embedder, pixels)
                save_vectors(p, vector[None])
                atomic_json(p.with_suffix(".json"), {"video_id": vid, "video_sha256": sha256_file(video),
                    "timestamps": stamps, "visual_patch_tokens": tokens, "sequence_tokens": sequence_tokens})
            vectors.append(vector)
            if index % 25 == 0 or index == len(ids) - 1:
                elapsed = time.perf_counter() - began
                atomic_json(out / "status.json", {"stage": "native-features", "completed_videos": index + 1,
                            "total_videos": len(ids), "elapsed_seconds": elapsed,
                            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()})
                print(f"Native features {index + 1}/{len(ids)} elapsed={elapsed:.1f}s", flush=True)
    documents = np.stack(vectors)
    if float(documents.std(axis=0).mean()) < 1e-5:
        raise ValueError("Collapsed video embeddings")
    save_vectors(features / "aggregate.npz", documents)
    for split, rows in queries.items():
        path = features / f"queries-{split}.npz"
        if path.exists():
            cached = load_vectors(path)
            if cached.shape != (len(rows), documents.shape[1]):
                raise ValueError("Cached query dimensions/coverage differ")
            continue
        chunks = []
        texts = [row["text"] for row in rows.values()]
        for start in range(0, len(texts), 16):
            with torch.inference_mode():
                encoded = embedder.process([{"text": t, "instruction": QUERY_INSTRUCTION} for t in texts[start:start + 16]])
            chunks.append(normalize(encoded.float().cpu().numpy()))
        save_vectors(path, np.concatenate(chunks))
    atomic_json(features / "manifest.json", {"video_ids": ids, "query_ids": {s: list(r) for s, r in queries.items()},
                "experiment_manifest_sha256": sha256_file(out / "experiment_manifest.json"),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "seconds": time.perf_counter() - began,
                "files": {str(p.relative_to(features)): sha256_file(p) for p in sorted(features.rglob("*"))
                          if p.is_file() and p.name != "manifest.json"}})
    del embedder
    gc.collect()
    torch.cuda.empty_cache()
    # Factorial analysis is a separate CPU job; extract once and reuse all rankings.
    finish(out, {"stage": "complete", "feature_seconds": time.perf_counter() - began,
                 "is_validation_score": False}, include_features=True)


def pilot_ids(videos, count=32):
    qrels = unique_rows(LABELS["train"], "query_id")
    queries = unique_rows(INPUTS["train"] / "queries.jsonl", "query_id")
    groups = [[qid for qid, row in queries.items() if needle in row["text"].lower()]
              for needle in ("because", "before", "does not", "traffic light")]
    chosen = []
    for offset in range(count):
        group = groups[offset % len(groups)]
        qid = group[(offset // len(groups)) % len(group)]
        candidates = sorted(set(qrels[qid]["relevant_video_ids"]) & set(videos) - set(chosen))
        if candidates:
            chosen.append(candidates[0])
    train_positive_ids = set().union(*(set(row["relevant_video_ids"]) for row in qrels.values()))
    chosen.extend(v for v in sorted(set(videos) & train_positive_ids) if v not in chosen)
    return chosen[:count]


def parse_description(raw, duration):
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return JSON")
    value = json.loads(raw[start:end + 1])
    if not isinstance(value, dict) or not isinstance(value.get("summary"), str):
        raise ValueError("Invalid summary")
    for key in ("events", "causal_hypotheses", "unknowns"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"Missing list field {key}")
    for event in value["events"]:
        if not isinstance(event, dict) or not all(key in event for key in ("actor", "action", "target", "t_start", "t_end", "polarity", "evidence")):
            raise ValueError("Malformed event")
        lo, hi = event["t_start"], event["t_end"]
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or not 0 <= lo <= hi <= duration:
            raise ValueError("Invalid event interval")
    return value


def descriptions(out):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    videos, _ = corpus()
    ids = pilot_ids(videos)
    spec = json.loads(Path("configs/retrieval_models.json").read_text())["qwen3-description"]
    model_dir = Path(spec["local_path"])
    identity(out, "structured-description-pilot", {"model_provenance_sha256": sha256_file(model_dir / "model_provenance.json"),
             "prompt": DESCRIPTION_PROMPT, "video_ids": ids, "selection": "Training positive examples covering causal/order/negation/control query text",
             "frames": 8, "max_pixels_per_image": 384 * 384, "max_new_tokens": 512,
             "is_validation_score": False, "input_queries_passed_to_model": False})
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    rows = []
    for index, vid in enumerate(ids, 1):
        p = out / "clips" / f"{vid}.json"
        video = (INPUTS["train"] / videos[vid]["path"]).resolve()
        if p.exists():
            row = json.loads(p.read_text())
            if sha256_file(video) != row["video_sha256"]:
                raise ValueError("Pilot video changed")
        else:
            stamps, frames = decode(video, 8)
            content = [{"type": "text", "text": DESCRIPTION_PROMPT}]
            for stamp, frame in zip(stamps, frames, strict=True):
                content.extend([{"type": "text", "text": f"Frame at {stamp:.3f} seconds:"},
                                {"type": "image", "image": frame, "min_pixels": 128 * 128, "max_pixels": 384 * 384}])
            conversation = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            images, video_inputs = process_vision_info(conversation, image_patch_size=16)
            batch = processor(text=[text], images=images, videos=video_inputs, do_resize=False,
                              return_tensors="pt", padding=True)
            if "pixel_values" not in batch or not batch["pixel_values"].numel():
                raise ValueError("No image tensors in description input")
            tokens = int(batch["input_ids"].shape[-1])
            batch = batch.to("cuda")
            with torch.inference_mode():
                outputs = model.generate(**batch, max_new_tokens=512, do_sample=False, num_beams=1)
            raw = processor.batch_decode(outputs[:, tokens:], skip_special_tokens=True)[0]
            duration = stamps[-1] * 8 / 7.5
            row = {"video_id": vid, "video_sha256": sha256_file(video), "timestamps": stamps,
                   "input_tokens": tokens, "raw_output": raw, "is_ground_truth": False}
            try:
                row["description"] = parse_description(raw, duration)
                row["parse_valid"] = True
            except (ValueError, TypeError) as error:
                row.update(parse_valid=False, parse_error=str(error))
            atomic_json(p, row)
        rows.append(row)
        elapsed = time.perf_counter() - began
        atomic_json(out / "status.json", {"stage": "describing", "completed": index, "total": len(ids),
                    "elapsed_seconds": elapsed, "parse_valid": sum(r["parse_valid"] for r in rows),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "is_validation_score": False})
        print(f"Structured descriptions {index}/32 elapsed={elapsed:.1f}s valid={row['parse_valid']}", flush=True)
    write_jsonl(out / "descriptions.jsonl", rows)
    manifest = {"is_validation_score": False, "is_ground_truth": False, "videos": len(rows),
                "parse_valid": sum(r["parse_valid"] for r in rows), "processing_seconds": time.perf_counter() - began,
                "projected_corpus_seconds": (time.perf_counter() - began) / len(rows) * len(videos),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "grounding_review": "pending; valid JSON does not establish correct observations",
                "full_corpus_started": False}
    atomic_json(out / "profile.json", manifest)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    finish(out, {"stage": "complete", **manifest})


def temporal_budget(out):
    import torch
    videos, _ = corpus()
    ids = pilot_ids(videos, 16)
    identity(out, "temporal-budget-pilot", {"video_ids": ids, "frames": [4, 8, 16],
             "total_pixels_target": 4 * 224 * 224, "max_length": 2048,
             "helper_sha256": sha256_file(Path("vendor/qwen3-vl-embedding/qwen3_vl_embedding.py")),
             "model_provenance_sha256": sha256_file(Path("artifacts/models/qwen3-video/model_provenance.json")),
             "is_validation_score": False})
    embedder = native_embedder()
    torch.cuda.reset_peak_memory_stats()
    results = []
    began = time.perf_counter()
    for count in (4, 8, 16):
        embedder.num_frames = embedder.max_frames = count
        vectors = []
        for vid in ids:
            started = time.perf_counter()
            video = (INPUTS["train"] / videos[vid]["path"]).resolve()
            stamps, frames = decode(video, count)
            vector, tokens, sequence = encode_video(embedder, frames)
            vectors.append(vector)
            results.append({"video_id": vid, "frames": count, "timestamps": stamps,
                            "video_sha256": sha256_file(video), "visual_patch_tokens": tokens,
                            "sequence_tokens": sequence, "seconds": time.perf_counter() - started})
            atomic_json(out / "status.json", {"stage": "temporal-budget-profile", "frames": count,
                        "completed_encodings": len(results), "total_encodings": 48,
                        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "is_validation_score": False})
        save_vectors(out / "features" / f"frames-{count}.npz", np.stack(vectors))
    summaries = {str(count): {"mean_visual_patch_tokens": float(np.mean([r["visual_patch_tokens"] for r in results if r["frames"] == count])),
                             "mean_sequence_tokens": float(np.mean([r["sequence_tokens"] for r in results if r["frames"] == count])),
                             "mean_seconds": float(np.mean([r["seconds"] for r in results if r["frames"] == count]))} for count in (4, 8, 16)}
    tokens = [summaries[str(count)]["mean_visual_patch_tokens"] for count in (4, 8, 16)]
    report = {"is_validation_score": False, "encodings": results, "summaries": summaries,
              "budget_matched_within_10_percent": max(tokens) / min(tokens) <= 1.1,
              "interpretation": "Preprocessing/throughput diagnostic. No claim of temporal accuracy improvement; token budgets may differ despite equal nominal total_pixels.",
              "processing_seconds": time.perf_counter() - began,
              "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
    atomic_json(out / "profile.json", report)
    finish(out, {"stage": "complete", "is_validation_score": False,
                 "budget_matched_within_10_percent": report["budget_matched_within_10_percent"]}, include_features=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("factorial", "factorial-native", "native", "descriptions", "temporal-budget"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        if args.kind.startswith("factorial"):
            factorial(args.out, args.kind == "factorial-native")
        else:
            import torch
            torch.manual_seed(0)
            torch.set_num_threads(2)
            {"native": native, "descriptions": descriptions, "temporal-budget": temporal_budget}[args.kind](args.out)
    except Exception as error:
        atomic_json(args.out / "status.json", {"stage": "failed", "error_type": type(error).__name__, "message": str(error)[:500]})
        raise


if __name__ == "__main__":
    main()
