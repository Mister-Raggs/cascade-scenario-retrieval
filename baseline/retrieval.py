from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


def tokenize(text: str) -> list[str]:
    # Retain negation and temporal words; this baseline has no learned parser.
    return re.findall(r"[a-z0-9]+", text.casefold())


class BM25:
    """A sparse caption index; returns positive-scoring matches without padding."""

    def __init__(self, documents: dict[str, str], *, k1: float = 1.2, b: float = 0.75):
        if not documents:
            raise ValueError("The video corpus is empty")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.lengths = {}
        self.postings = defaultdict(dict)
        for video_id, text in documents.items():
            terms = Counter(tokenize(text))
            self.lengths[video_id] = sum(terms.values())
            for term, frequency in terms.items():
                self.postings[term][video_id] = frequency
        self.average_length = sum(self.lengths.values()) / len(documents) or 1.0
        self.idf = {
            term: math.log(1 + (len(documents) - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.postings.items()
        }

    def rank(self, query: str, top_k: int = 100) -> list[str]:
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        scores = defaultdict(float)
        for term in sorted(set(tokenize(query))):
            for video_id, frequency in self.postings.get(term, {}).items():
                length_factor = 1 - self.b + self.b * self.lengths[video_id] / self.average_length
                scores[video_id] += self.idf[term] * frequency * (self.k1 + 1) / (
                    frequency + self.k1 * length_factor
                )
        ordered = sorted(scores, key=lambda video_id: (-scores[video_id], video_id))
        return [video_id for video_id in ordered if scores[video_id] > 0][:top_k]


def caption_documents(rows: dict[str, dict]) -> dict[str, str]:
    documents = {}
    for video_id, row in rows.items():
        frames = row.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"Missing frame captions for {video_id}")
        texts = []
        for frame in frames:
            if not isinstance(frame.get("text"), str) or not frame["text"].strip():
                raise ValueError(f"Empty caption for {video_id}")
            if frame["text"] not in texts:
                texts.append(frame["text"])
        documents[video_id] = " ".join(texts)
    return documents
