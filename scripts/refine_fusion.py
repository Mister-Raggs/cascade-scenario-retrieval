"""Small CPU fusion sweep over pinned cached features; select on train only."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.devkit import load_module
from baseline.evaluate import evaluate
from baseline.experiment_analysis import paired_delta
from baseline.io import sha256_file, unique_rows, write_jsonl
from baseline.retrieval import BM25, caption_documents
from scripts.feature_retrieval import atomic_json, load_vectors
from scripts.research_experiments import corpus, verified_features, INPUTS, LABELS, OLD, RUN
from scripts.robustness_retrieval import partitions


def reciprocal_matrix(ranking, size):
    ranking = np.asarray(ranking)
    if ranking.ndim != 2 or ranking.shape[1] != size:
        raise ValueError("Full rankings required")
    if np.any(np.sort(ranking, axis=1) != np.arange(size)):
        raise ValueError("Rankings must be unique corpus permutations")
    result = np.empty(ranking.shape, dtype=np.float32)
    np.put_along_axis(result, ranking, np.asarray(1 / (60 + np.arange(1, size + 1)), dtype=np.float32)[None, :], axis=1)
    return result


def score_rankings(ranking, relevance):
    if ranking.shape != relevance.shape:
        raise ValueError("Ranking/label dimensions differ")
    hits = np.take_along_axis(relevance, ranking, axis=1)
    counts = relevance.sum(axis=1)
    positions = np.arange(1, ranking.shape[1] + 1)
    rprec = np.divide((hits & (positions <= counts[:, None])).sum(axis=1), counts,
                      out=np.full(len(counts), np.nan), where=counts > 0)
    ap = np.divide(((hits.cumsum(axis=1) / positions) * hits).sum(axis=1), counts,
                   out=np.full(len(counts), np.nan), where=counts > 0)
    return rprec, ap


def rank_fusion(channels, weights):
    if set(weights) != {"B", "T", "V", "N"} or any(not np.isfinite(v) or v < 0 for v in weights.values()) or not any(weights.values()):
        raise ValueError("Finite nonnegative coefficients with one positive signal required")
    scores = np.zeros_like(channels["B"])
    for factor in ("B", "T", "V", "N"):
        if weights[factor]:
            scores += weights[factor] * channels[factor]
    # Corpus IDs are sorted, so stable order matches the parent tie break.
    return np.argsort(-scores, axis=1, kind="stable").astype(np.int32)


def channels_for(split, ids, queries, lexical, mode, out):
    paths = {"T": OLD / "semantic/features", "V": OLD / "visual/features", "N": RUN / "native/features"}
    documents = {"T": load_vectors(paths["T"] / "aggregate.npz"),
                 "V": load_vectors(paths["V"] / "frames.npz"),
                 "N": load_vectors(paths["N"] / "aggregate.npz")}
    if mode == "mean":
        documents["V"] = documents["V"].mean(axis=1)
    query_vectors = {name: load_vectors(path / f"queries-{split}.npz") for name, path in paths.items()}
    matrices = {name: np.zeros((len(queries), len(ids)), dtype=np.float32) for name in ("B", "T", "V", "N")}
    positions = {vid: i for i, vid in enumerate(ids)}
    values = np.asarray(1 / (60 + np.arange(1, len(ids) + 1)), dtype=np.float32)
    for i, row in enumerate(queries.values()):
        matches = lexical.rank(row["text"], len(ids))
        # Keep unmatched lexical items at zero, as in the parent. Do not RRF-pad B.
        matrices["B"][i, [positions[v] for v in matches]] = values[:len(matches)]
        for factor in ("T", "V", "N"):
            vector = query_vectors[factor][i]
            similarity = (np.max(np.einsum("vfd,d->vf", documents[factor], vector), axis=1)
                          if factor == "V" and mode == "max" else documents[factor] @ vector)
            ranking = np.argsort(-similarity, kind="stable")
            matrices[factor][i, ranking] = values
        if i % 500 == 0 or i == len(queries) - 1:
            atomic_json(out / "status.json", {"stage": "cached-channel-ranking", "split": split,
                        "completed_queries": i + 1, "total_queries": len(queries)})
            print(f"Cached channels {split} {i+1}/{len(queries)}", flush=True)
    return matrices


def upload(source, prefix, receipt):
    subprocess.run([sys.executable, "scripts/spaces.py", "upload", "--source", str(source),
                    "--prefix", prefix, "--receipt", str(receipt)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/refinement-v1/fusion"))
    parser.add_argument("--plan", type=Path, default=Path("configs/refinement_experiment_plan.json"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    plan = json.loads(args.plan.read_text())
    videos, queries = corpus()
    ids = sorted(videos)
    paths = {"T": OLD / "semantic/features", "V": OLD / "visual/features", "N": RUN / "native/features"}
    for path in paths.values():
        verified_features(path, videos, queries)
    parent = RUN / "factorial-native"
    if json.loads((parent / "status.json").read_text())["stage"] != "complete" or not (parent / "receipts/results.json").exists():
        raise ValueError("Complete backed-up parent comparison required")
    parent_manifest = json.loads((parent / "experiment_manifest.json").read_text())
    hashes = {s: {"queries": sha256_file(p / "queries.jsonl"), "videos": sha256_file(p / "videos.jsonl"),
                   "qrels": sha256_file(LABELS[s])} for s, p in INPUTS.items()}
    if hashes != parent_manifest["input_hashes"]:
        raise ValueError("Parent inputs changed")
    mode = parent_manifest["extra"]["visual_mode_selected_on_train"]
    incumbent = json.loads((parent / "selected_on_train.json").read_text())
    incumbent_weights = {f: (1 if f == "B" else 3) if f in incumbent["factors"] else 0 for f in ("B", "T", "V", "N")}
    recipes = [{"name": "incumbent", "weights": incumbent_weights, "parent_recipe": incumbent["recipe"]}, *plan["fusion_recipes"]]
    manifest = {"kind": "cached-feature-fusion-sweep", "code_sha256": sha256_file(Path(__file__)),
                "plan_sha256": sha256_file(args.plan), "input_hashes": hashes,
                "parent_manifest_sha256": sha256_file(parent / "experiment_manifest.json"),
                "feature_manifest_sha256": {f: sha256_file(p / "manifest.json") for f, p in paths.items()},
                "dependency_code_sha256": {name: sha256_file(ROOT / name) for name in
                    ("scripts/research_experiments.py", "scripts/feature_retrieval.py", "scripts/robustness_retrieval.py", "baseline/retrieval.py", "baseline/experiment_analysis.py")},
                "recipes": recipes, "visual_mode": mode, "selection_split": "train",
                "query_fold_seed": plan["query_fold_seed"], "query_folds": plan["query_folds"],
                "limitations": plan["limitations"]}
    saved = out / "experiment_manifest.json"
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        raise ValueError("Changed inputs or code; use a fresh run directory")
    atomic_json(saved, manifest)
    captions = unique_rows(Path("runs/blip-bm25-val-v1/captions.jsonl"), "video_id")
    lexical = BM25(caption_documents(captions), k1=2, b=0)
    channels = channels_for("train", ids, queries["train"], lexical, mode, out)
    qrels = unique_rows(LABELS["train"], "query_id")
    position = {vid: i for i, vid in enumerate(ids)}
    relevance = np.zeros((len(queries["train"]), len(ids)), dtype=bool)
    labels = []
    for i, qid in enumerate(queries["train"]):
        relevant = {position[v] for v in qrels[qid]["relevant_video_ids"]}
        labels.append(relevant)
        relevance[i, list(relevant)] = True
    folds = partitions(len(queries["train"]), plan["query_folds"], plan["query_fold_seed"])
    metrics = load_module("metrics")
    results, arrays = [], {}
    for recipe in recipes:
        name = recipe["name"]
        atomic_json(out / "status.json", {"stage": "training-fusion", "recipe": name, "completed": len(results), "total": len(recipes)})
        ranking = rank_fusion(channels, recipe["weights"])
        rp, ap = score_rankings(ranking, relevance)
        for i in np.linspace(0, len(labels)-1, 5, dtype=int):
            if labels[i] and (abs(metrics.r_precision(ranking[i].tolist(), labels[i]) - rp[i]) > 1e-12
                              or abs(metrics.average_precision(ranking[i].tolist(), labels[i]) - ap[i]) > 1e-12):
                raise ValueError("Vectorized scoring differs from official metrics")
        row = {**recipe, "average_r_precision": float(np.nanmean(rp)), "map": float(np.nanmean(ap)),
               "query_fold_r_precision": [float(np.nanmean(rp[group])) for group in folds]}
        if name == "incumbent" and any(abs(row[k] - incumbent[k]) > 1e-10 for k in ("average_r_precision", "map")):
            raise ValueError("Cached-feature reference differs from parent metrics")
        arrays[name] = rp
        results.append(row)
        atomic_json(out / "train_results.json", results)
    winner = max(results, key=lambda row: (row["average_r_precision"], row["map"]))
    atomic_json(out / "selected_on_train.json", winner)
    diagnostics = {"paired_vs_incumbent": {name: paired_delta(arrays["incumbent"][np.isfinite(arrays[name])], a[np.isfinite(a)])
                    for name, a in arrays.items() if name != "incumbent"},
                   "query_fold_deltas_vs_incumbent": {row["name"]: [a-b for a,b in zip(row["query_fold_r_precision"], results[0]["query_fold_r_precision"])] for row in results},
                   "interpretation": "Post-selection training diagnostics; folds reuse previously inspected training data, not independent untouched holdouts."}
    atomic_json(out / "diagnostics.json", diagnostics)
    prefix = "experiments/refinement-v1/fusion"
    if winner["name"] == "incumbent":
        original = parent / "selected-validation/evaluation.json"
        validation = json.loads(original.read_text())
        atomic_json(out / "validation_reused.json", {"parent_evaluation": str(original), "sha256": sha256_file(original),
                    "evaluation": validation, "reason": "Incumbent retained; no additional validation ranking evaluation"})
    else:
        # Retain only the selected prediction runs; avoid large artifacts for each weight recipe.
        selected_ranking = rank_fusion(channels, winner["weights"])
        train_dir = out / "selected-training"
        write_jsonl(train_dir / "predictions.jsonl", ({"query_id": qid, "video_ids": [ids[j] for j in selected_ranking[i]]} for i,qid in enumerate(queries["train"])))
        official = evaluate(INPUTS["train"], LABELS["train"], train_dir / "predictions.jsonl", train_dir / "evaluation.json")
        if any(abs(official[k] - winner[k]) > 1e-10 for k in ("average_r_precision", "map")):
            raise ValueError("Selected full training run does not reproduce official metrics")
        upload(train_dir, prefix + "/selected-training", out / "receipts/selected-training.json")
        del channels, selected_ranking, relevance
        gc.collect()
        channels = channels_for("val", ids, queries["val"], lexical, mode, out)
        ranking = rank_fusion(channels, winner["weights"])
        val_dir = out / "selected-validation"
        write_jsonl(val_dir / "predictions.jsonl", ({"query_id": qid, "video_ids": [ids[j] for j in ranking[i]]} for i,qid in enumerate(queries["val"])))
        validation = evaluate(INPUTS["val"], LABELS["val"], val_dir / "predictions.jsonl", val_dir / "evaluation.json")
        upload(val_dir, prefix + "/selected-validation", out / "receipts/selected-validation.json")
    result = {"stage": "computed-awaiting-backup", "selected_on_train": winner,
              "validation": {k: validation[k] for k in ("average_r_precision", "map", "num_queries")},
              "elapsed_seconds": time.monotonic() - began, "new_video_inference": False}
    atomic_json(out / "status.json", result)
    snapshot = out / "results"
    snapshot.mkdir(exist_ok=True)
    for p in out.glob("*.json"):
        shutil.copyfile(p, snapshot / p.name)
    upload(snapshot, prefix + "/results", out / "receipts/results.json")
    atomic_json(out / "status.json", {**result, "stage": "complete"})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
