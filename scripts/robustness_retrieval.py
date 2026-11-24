"""CPU-only, post-selection stability diagnostics on cached full-corpus rankings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
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
from baseline.devkit import DEVKIT_COMMIT, load_module
from baseline.io import sha256_file, unique_rows

SEEDS = (17, 29, 43, 59, 71)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def partitions(count, folds, seed):
    if folds < 2 or count < folds:
        raise ValueError("At least two nonempty groups required")
    return [np.sort(x) for x in np.array_split(np.random.default_rng(seed).permutation(count), folds)]


def read_rankings(path, query_ids, video_ids):
    """Encode validated complete permutations without retaining millions of UUID strings."""
    queries = {qid: i for i, qid in enumerate(query_ids)}
    videos = {vid: i for i, vid in enumerate(video_ids)}
    matrix = np.empty((len(queries), len(videos)), dtype=np.int32)
    seen = set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["query_id"]
            if qid not in queries or qid in seen:
                raise ValueError("Unknown or duplicate prediction query")
            ids = row["video_ids"]
            if len(ids) != len(videos) or set(ids) != set(videos):
                raise ValueError("Each ranking must be a full unique corpus permutation")
            matrix[queries[qid]] = [videos[vid] for vid in ids]
            seen.add(qid)
    if seen != set(queries):
        raise ValueError("Missing prediction queries")
    return matrix


def score_group(rankings, labels, members, metrics):
    """Filter rankings AND positives; recompute R; skip and count zero-positive queries."""
    membership = set(map(int, members))
    mask = np.zeros(rankings.shape[1], dtype=bool)
    mask[list(membership)] = True
    rprec = np.full(len(labels), np.nan)
    ap = np.full(len(labels), np.nan)
    counts = np.zeros(len(labels), dtype=np.int32)
    for i, relevant in enumerate(labels):
        restricted = relevant & membership
        counts[i] = len(restricted)
        if not restricted:
            continue
        ranking = rankings[i][mask[rankings[i]]].tolist()
        rprec[i] = metrics.r_precision(ranking, restricted)
        ap[i] = metrics.average_precision(ranking, restricted)
    return rprec, ap, counts


def summarize(arrays, counts, query_indices=None):
    chosen = np.arange(len(counts)) if query_indices is None else np.asarray(query_indices)
    scored = chosen[counts[chosen] > 0]
    recipes = {}
    for name, (rprec, ap) in arrays.items():
        if not np.array_equal(np.isfinite(rprec), counts > 0):
            raise ValueError("Paired recipe scoring coverage differs")
        recipes[name] = {"average_r_precision": float(rprec[scored].mean()) if len(scored) else None,
                         "map": float(ap[scored].mean()) if len(scored) else None}
    pairs = [("B", x) for x in arrays if x != "B"]
    pairs.extend((base, changed) for base, changed in (("BV", "BTV"), ("BT", "BTV"),
                 ("BV", "BVN"), ("BN", "BVN"), ("BVN", "BTVN")) if base in arrays and changed in arrays)
    comparisons = {}
    for base, changed in pairs:
        delta = arrays[changed][0][scored] - arrays[base][0][scored]
        comparisons[f"{changed}_minus_{base}"] = {
            "mean_r_precision_delta": float(delta.mean()) if len(scored) else None,
            "mean_map_delta": float((arrays[changed][1][scored] - arrays[base][1][scored]).mean()) if len(scored) else None,
            "improved_queries": int((delta > 1e-12).sum()),
            "regressed_queries": int((delta < -1e-12).sum()),
            "unchanged_queries": int((np.abs(delta) <= 1e-12).sum())}
    return {"num_queries": len(chosen), "num_scored_queries": len(scored),
            "num_queries_without_relevance": len(chosen) - len(scored),
            "positive_pairs": int(counts[chosen].sum()), "recipes": recipes, "comparisons": comparisons}


def wait_for_parent(args):
    if not args.wait_queue:
        return
    while True:
        queue = json.loads(args.wait_queue.read_text())
        job = next(row for row in queue["jobs"] if row["id"] == args.wait_job)
        if datetime.now(timezone.utc) >= datetime.fromisoformat(args.start_before.replace("Z", "+00:00")):
            raise ValueError("New-start review deadline reached while waiting for parent")
        if job["state"] == "complete":
            return
        if job["state"] in {"failed", "timed-out", "blocked", "cancelled"}:
            raise ValueError(f"Required parent {args.wait_job} is {job['state']}")
        atomic_json(args.out / "status.json", {"stage": "waiting-for-parent", "job": args.wait_job,
                    "parent_state": job["state"], "checked_at_utc": datetime.now(timezone.utc).isoformat()})
        time.sleep(15)


def run(args):
    wait_for_parent(args)
    began = time.monotonic()
    inputs = Path("data/prepared-train-blip-v1/input")
    qrels_path = inputs.parent / "labels/qrels.jsonl"
    provenance = json.loads((inputs / "provenance.json").read_text())
    if provenance.get("split") != "train":
        raise ValueError("Only training diagnostics are authorized by this experiment plan")
    queries = unique_rows(inputs / "queries.jsonl", "query_id")
    videos = unique_rows(inputs / "videos.jsonl", "video_id")
    qrels = unique_rows(qrels_path, "query_id")
    if set(queries) != set(qrels):
        raise ValueError("Qrels coverage differs")
    query_ids, video_ids = list(queries), sorted(videos)
    index = {vid: i for i, vid in enumerate(video_ids)}
    labels = [{index[v] for v in qrels[qid]["relevant_video_ids"]} for qid in query_ids]
    metrics = load_module("metrics")
    predictions = {name: args.parent / f"train-{name}/predictions.jsonl" for name in args.recipes}
    manifest = {"kind": "post-selection-cached-ranking-stability", "split": "train",
                "code_sha256": sha256_file(Path(__file__)), "devkit_commit": DEVKIT_COMMIT,
                "metrics_code_sha256": sha256_file(Path(metrics.__file__)),
                "input_hashes": {"queries": sha256_file(inputs / "queries.jsonl"),
                                 "videos": sha256_file(inputs / "videos.jsonl"), "qrels": sha256_file(qrels_path)},
                "parent_manifest_sha256": sha256_file(args.parent / "experiment_manifest.json"),
                "prediction_sha256": {name: sha256_file(path) for name, path in predictions.items()},
                "video_seeds": list(SEEDS), "video_folds_per_repeat": 2,
                "query_seed": 103, "query_folds": 5,
                "limitations": ["Frozen recipes were previously selected on this training split; these are post-selection diagnostics, not untouched holdouts.",
                    "Video groups are disjoint within each repeat but overlap across repeats. Query/video correlations remain; no independent-sample confidence claim.",
                    "Full-corpus fusion rankings are filtered, not re-fused on subset indices. R and candidate corpus both change, so subset absolute scores are not comparable to full-corpus scores.",
                    "Published train/val share videos. This does not measure unseen-video or hidden-test generalization.",
                    "No validation tuning, new GPU inference, or official submission."]}
    saved = args.out / "experiment_manifest.json"
    parent_manifest = json.loads((args.parent / "experiment_manifest.json").read_text())
    if parent_manifest["input_hashes"]["train"] != manifest["input_hashes"]:
        raise ValueError("Parent training input provenance differs")
    if saved.exists() and json.loads(saved.read_text()) != manifest:
        raise ValueError("Inputs or code changed; use a fresh output directory")
    atomic_json(saved, manifest)
    rankings = {}
    for name, path in predictions.items():
        parent_evaluation = json.loads(path.with_name("evaluation.json").read_text())
        if (parent_evaluation["predictions_sha256"] != manifest["prediction_sha256"][name]
                or parent_evaluation["qrels_sha256"] != manifest["input_hashes"]["qrels"]):
            raise ValueError("Parent predictions/labels hash mismatch")
        rankings[name] = read_rankings(path, query_ids, video_ids)
    memberships = {"video_groups": [], "query_groups": []}
    report = {"data_origin": provenance.get("data_origin"), "split": "train",
              "num_videos": len(videos), "num_queries": len(queries),
              "interpretation": manifest["limitations"], "video_groups": [], "query_groups": []}
    full = {name: score_group(ranking, labels, range(len(videos)), metrics) for name, ranking in rankings.items()}
    counts = next(iter(full.values()))[2]
    arrays = {name: values[:2] for name, values in full.items()}
    report["full_corpus"] = summarize(arrays, counts)
    for name, value in report["full_corpus"]["recipes"].items():
        reference = json.loads(predictions[name].with_name("evaluation.json").read_text())
        if any(abs(value[key] - reference[key]) > 1e-10 for key in ("average_r_precision", "map")):
            raise ValueError("Full-corpus regression differs from parent's official metrics")
    for fold, indices in enumerate(partitions(len(queries), 5, 103)):
        report["query_groups"].append({"fold": fold, "num_videos": len(videos),
                                       **summarize(arrays, counts, indices)})
        memberships["query_groups"].append({"fold": fold, "query_ids": [query_ids[i] for i in indices]})
    for seed in SEEDS:
        for fold, members in enumerate(partitions(len(videos), 2, seed)):
            atomic_json(args.out / "status.json", {"stage": "scoring-video-group", "seed": seed, "fold": fold,
                        "completed_groups": len(report["video_groups"]), "total_groups": 10})
            grouped = {name: score_group(ranking, labels, members, metrics) for name, ranking in rankings.items()}
            group_counts = next(iter(grouped.values()))[2]
            if any(not np.array_equal(values[2], group_counts) for values in grouped.values()):
                raise ValueError("Positive coverage differs across recipes")
            report["video_groups"].append({"seed": seed, "fold": fold, "num_videos": len(members),
                    **summarize({name: values[:2] for name, values in grouped.items()}, group_counts)})
            memberships["video_groups"].append({"seed": seed, "fold": fold,
                                               "video_ids": [video_ids[i] for i in members]})
            atomic_json(args.out / "report.json", report)
    report["sign_consistency"] = {}
    for name in report["full_corpus"]["comparisons"]:
        report["sign_consistency"][name] = {}
        for kind in ("video_groups", "query_groups"):
            values = [r["comparisons"][name]["mean_r_precision_delta"] for r in report[kind]]
            report["sign_consistency"][name][kind] = {
                "groups": len(values), "positive_delta_groups": sum(v > 1e-12 for v in values),
                "negative_delta_groups": sum(v < -1e-12 for v in values),
                "minimum_delta": min(values), "maximum_delta": max(values)}
    report["elapsed_seconds"] = time.monotonic() - began
    atomic_json(args.out / "report.json", report)
    atomic_json(args.out / "memberships.json", memberships)
    atomic_json(args.out / "status.json", {"stage": "computed-awaiting-backup", "elapsed_seconds": report["elapsed_seconds"]})
    snapshot = args.out / "results"
    snapshot.mkdir(exist_ok=True)
    for filename in ("report.json", "memberships.json", "experiment_manifest.json"):
        shutil.copyfile(args.out / filename, snapshot / filename)
    subprocess.run([sys.executable, "scripts/spaces.py", "upload", "--source", str(snapshot),
                    "--prefix", args.prefix + "/results", "--receipt", str(args.out / "receipts/results.json")], check=True)
    atomic_json(args.out / "status.json", {"stage": "complete", "elapsed_seconds": report["elapsed_seconds"],
                "results_receipt_sha256": sha256_file(args.out / "receipts/results.json")})
    print(json.dumps(report["sign_consistency"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--recipes", nargs="+", default=["B", "BT", "BV", "BTV"])
    parser.add_argument("--wait-queue", type=Path)
    parser.add_argument("--wait-job", default="factorial-native")
    parser.add_argument("--start-before", default="2026-10-10T01:14:02Z")
    args = parser.parse_args()
    if (len(set(args.recipes)) != len(args.recipes) or "B" not in args.recipes
            or set(args.recipes) - {"B", "BT", "BV", "BTV", "BN", "BTN", "BVN", "BTVN"}):
        parser.error("Unique recipes including reference B required")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(args.out / "worker.json", {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat()})
        try:
            run(args)
        except Exception as error:
            atomic_json(args.out / "status.json", {"stage": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
