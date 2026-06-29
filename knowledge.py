"""
knowledge.py — a tiny, dependency-free retriever over the product catalog.

It uses bag-of-words vectors + cosine similarity so the demo runs with zero
external services. In production you'd swap `_vectorize` for real embeddings
(OpenAI / Claude / a local model) — the Retriever interface stays the same.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"[a-zăâîșțéè0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class Retriever:
    """Loads products.json and returns the most relevant items for a query."""

    def __init__(self, path: str = "products.json") -> None:
        self.products = json.loads(Path(path).read_text(encoding="utf-8"))
        for p in self.products:
            blob = " ".join([p["name"], p["category"], p["description"], *p["tags"]])
            p["_vec"] = self._vectorize(blob)

    def _vectorize(self, text: str) -> Counter:
        return Counter(_tokenize(text))

    def search(self, query: str, k: int = 3) -> list[dict]:
        qv = self._vectorize(query)
        scored = sorted(
            self.products,
            key=lambda p: _cosine(qv, p["_vec"]),
            reverse=True,
        )
        results = [p for p in scored if _cosine(qv, p["_vec"]) > 0][:k]
        return [{key: p[key] for key in p if key != "_vec"} for p in results]


if __name__ == "__main__":
    r = Retriever()
    for hit in r.search("bormasina percutie pentru beton"):
        print(f"{hit['name']} — {hit['price_ron']} lei "
              f"({'în stoc' if hit['in_stock'] else 'stoc epuizat'})")
