from __future__ import annotations

import argparse
import json
from pathlib import Path

from .devkit import DEFAULT_DEVKIT, DEVKIT_COMMIT, load_module
from .io import sha256_file, unique_rows, write_json, write_jsonl


def evaluate(input_dir: Path, qrels_path: Path, predictions_path: Path, out: Path,
             devkit_root: Path = DEFAULT_DEVKIT) -> dict:
    if qrels_path.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("Relevance labels must remain outside the inference input directory")
    queries = unique_rows(input_dir / "queries.jsonl", "query_id")
    videos = unique_rows(input_dir / "videos.jsonl", "video_id")
    labels = unique_rows(qrels_path, "query_id")
    predictions = unique_rows(predictions_path, "query_id")
    if set(labels) != set(queries):
        raise ValueError("Relevance query IDs must exactly match the evaluated input queries")
    if set(predictions) - set(queries):
        raise ValueError("Predictions contain unknown query IDs")
    for rows, key in ((labels, "relevant_video_ids"), (predictions, "video_ids")):
        for query_id, row in rows.items():
            ids = row.get(key)
            if not isinstance(ids, list) or any(not isinstance(v, str) for v in ids):
                raise ValueError(f"{query_id}: {key} must be a list of strings")
            if set(ids) - set(videos):
                raise ValueError(f"{query_id}: IDs outside video corpus")
    metrics = load_module("metrics", devkit_root)
    qrels = {qid: set(row["relevant_video_ids"]) for qid, row in labels.items()}
    run = {qid: row["video_ids"] for qid, row in predictions.items()}
    result = metrics.evaluate_run(run, qrels)
    provenance_path = input_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    result.update(data_origin=provenance.get("data_origin", "unspecified"),
                  split_name=provenance.get("split_name"), split=provenance.get("split"),
                  devkit_commit=DEVKIT_COMMIT, num_predicted=len(predictions),
                  qrels_sha256=sha256_file(qrels_path), predictions_sha256=sha256_file(predictions_path))
    write_json(out, result)
    write_jsonl(out.with_name(out.stem + "_per_query.jsonl"), (
        {"query_id": qid, "text": queries[qid]["text"], "num_relevant": len(relevant),
         "scored": bool(relevant), "r_precision": metrics.r_precision(run.get(qid, []), relevant),
         "average_precision": metrics.average_precision(run.get(qid, []), relevant),
         "relevant_video_ids": sorted(relevant), "video_ids": run.get(qid, [])}
        for qid, relevant in qrels.items()
    ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score locally using the pinned official metric implementation")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--devkit-root", type=Path, default=DEFAULT_DEVKIT)
    args = parser.parse_args()
    result = evaluate(args.input, args.qrels, args.predictions, args.out, args.devkit_root)
    print(f"{result['data_origin']}: R-Precision={result['average_r_precision']:.4f}, MAP={result['map']:.4f}")


if __name__ == "__main__":
    main()
