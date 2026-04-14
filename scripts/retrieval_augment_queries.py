"""
Build RAG-style prompts by retrieving top TF-IDF chunks from an in-repo corpus.
No extra services: works offline and is reproducible given fixed corpus + queries.

Example:
  python scripts/retrieval_augment_queries.py \\
    --queries data/queries.json --corpus data/rag_corpus.json \\
    --out outputs/queries_rag_augmented.json --top_k 3

Then run inference on the augmented file:
  python scripts/inference.py --queries outputs/queries_rag_augmented.json --out outputs/rag_outputs.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def build_idf(docs: List[List[str]]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    n = len(docs)
    for toks in docs:
        seen = set(toks)
        for t in seen:
            df[t] = df.get(t, 0) + 1
    idf: Dict[str, float] = {}
    for t, c in df.items():
        idf[t] = math.log((1.0 + n) / (1.0 + c)) + 1.0
    return idf


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    n = len(tokens) or 1
    vec: Dict[str, float] = {}
    for t, c in tf.items():
        if t not in idf:
            continue
        vec[t] = (c / n) * idf[t]
    return vec


def cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--corpus", default="data/rag_corpus.json")
    p.add_argument("--out", default="outputs/queries_rag_augmented.json")
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        corpus = json.load(f)
    chunk_texts = [f"{c.get('title', '')}\n{c['text']}" for c in corpus]
    chunk_docs = [tokenize(t) for t in chunk_texts]
    idf = build_idf(chunk_docs)
    chunk_vecs = [tfidf_vector(d, idf) for d in chunk_docs]

    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)

    out = []
    for q in queries:
        inst = q.get("instruction", "")
        q_tokens = tokenize(inst)
        qvec = tfidf_vector(q_tokens, idf)
        scored: List[Tuple[float, int]] = []
        for i, cvec in enumerate(chunk_vecs):
            scored.append((cosine_sim(qvec, cvec), i))
        scored.sort(reverse=True)
        top = scored[: args.top_k]
        blocks = []
        retrieved_ids = []
        for score, idx in top:
            item = corpus[idx]
            retrieved_ids.append(item.get("id", idx))
            title = item.get("title", f"chunk_{idx}")
            blocks.append(f"[{title}] (score={score:.3f})\n{item['text']}")
        context = "\n\n".join(blocks)
        new_inst = (
            "Use the following reference excerpts (they may be incomplete). "
            "Ground your answer in them when relevant; if they do not cover the question, "
            "say what is missing and answer from general knowledge.\n\n"
            "---\n"
            f"{context}\n"
            "---\n\n"
            f"User request:\n{inst}"
        )
        entry = {
            "id": q.get("id"),
            "instruction": new_inst,
            "system": q.get("system"),
            "rag_chunk_ids": retrieved_ids,
        }
        out.append(entry)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} augmented queries to {outp}")


if __name__ == "__main__":
    main()
