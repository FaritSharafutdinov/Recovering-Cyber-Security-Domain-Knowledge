"""
Build RAG-style prompts by retrieving top chunks from an in-repo corpus.

Modes:
  --mode tfidf (legacy): sparse TF-IDF cosine similarity, no external index.
  --mode embedding_faiss: dense embeddings + FAISS CPU index from scripts/build_rag_index.py.

Examples:
  python scripts/retrieval_augment_queries.py --mode tfidf \\
    --queries data/queries.json --corpus data/rag_corpus.json \\
    --out outputs/queries_rag_augmented.json --top_k 3

  python scripts/build_rag_index.py --corpus data/rag_corpus.json --output_dir outputs/rag_index
  python scripts/retrieval_augment_queries.py --mode embedding_faiss \\
    --queries data/queries.json --index_dir outputs/rag_index \\
    --out outputs/queries_rag_augmented_faiss.json --top_k 3
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lib.rag_corpus import load_corpus_chunks

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


def augment_tfidf(
    queries: List[Dict[str, Any]], corpus: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    chunk_texts = [f"{c.get('title', '')}\n{c['text']}" for c in corpus]
    chunk_docs = [tokenize(t) for t in chunk_texts]
    idf = build_idf(chunk_docs)
    chunk_vecs = [tfidf_vector(d, idf) for d in chunk_docs]

    out = []
    for q in queries:
        inst = q.get("instruction", "")
        q_tokens = tokenize(inst)
        qvec = tfidf_vector(q_tokens, idf)
        scored: List[Tuple[float, int]] = []
        for i, cvec in enumerate(chunk_vecs):
            scored.append((cosine_sim(qvec, cvec), i))
        scored.sort(reverse=True)
        top = scored[:top_k]
        blocks = []
        retrieved_ids = []
        for score, idx in top:
            item = corpus[idx]
            retrieved_ids.append(item.get("id", idx))
            title = item.get("title", f"chunk_{idx}")
            blocks.append(f"[{title}] (score={score:.3f})\n{item['text']}")
        context = "\n\n".join(blocks)
        new_inst = _wrap_instruction(inst, context, backend="tfidf")
        out.append(
            {
                "id": q.get("id"),
                "instruction": new_inst,
                "system": q.get("system"),
                "rag_chunk_ids": retrieved_ids,
                "rag_mode": "tfidf",
            }
        )
    return out


def augment_faiss(
    queries: List[Dict[str, Any]], index_dir: Path, top_k: int
) -> List[Dict[str, Any]]:
    import faiss  # noqa: PLC0415
    import numpy as np
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    meta_path = index_dir / "meta.json"
    idx_path = index_dir / "index.faiss"
    if not meta_path.exists() or not idx_path.exists():
        raise FileNotFoundError(f"Missing meta.json or index.faiss under {index_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks: List[Dict[str, Any]] = meta["chunks"]
    model_name = meta["model_name"]
    index = faiss.read_index(str(idx_path))
    model = SentenceTransformer(model_name)

    inst_texts = [q.get("instruction", "") for q in queries]
    qemb = model.encode(
        inst_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(inst_texts) > 32,
    )
    qemb = np.asarray(qemb, dtype=np.float32)
    sims, idxs = index.search(qemb, top_k)

    out = []
    for qi, q in enumerate(queries):
        blocks = []
        retrieved_ids = []
        for rank in range(top_k):
            idx = int(idxs[qi, rank])
            score = float(sims[qi, rank])
            item = chunks[idx]
            retrieved_ids.append(item.get("id", idx))
            title = item.get("title", f"chunk_{idx}")
            blocks.append(f"[{title}] (score={score:.3f})\n{item['text']}")
        context = "\n\n".join(blocks)
        new_inst = _wrap_instruction(q.get("instruction", ""), context, backend="embedding_faiss")
        out.append(
            {
                "id": q.get("id"),
                "instruction": new_inst,
                "system": q.get("system"),
                "rag_chunk_ids": retrieved_ids,
                "rag_mode": "embedding_faiss",
            }
        )
    return out


def _wrap_instruction(user_inst: str, context: str, backend: str) -> str:
    return (
        "Use the following reference excerpts (they may be incomplete). "
        "Ground your answer in them when relevant; if they do not cover the question, "
        "say what is missing and answer from general knowledge.\n\n"
        f"(retrieval_backend={backend})\n"
        "---\n"
        f"{context}\n"
        "---\n\n"
        f"User request:\n{user_inst}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("tfidf", "embedding_faiss"), default="tfidf")
    p.add_argument("--queries", default="data/queries.json")
    p.add_argument("--corpus", default="data/rag_corpus.json", help="Used only for --mode tfidf.")
    p.add_argument("--index_dir", default="outputs/rag_index", help="Used for --mode embedding_faiss.")
    p.add_argument("--out", default="outputs/queries_rag_augmented.json")
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        queries = json.load(f)

    if args.mode == "tfidf":
        corpus = load_corpus_chunks(args.corpus)
        out = augment_tfidf(queries, corpus, args.top_k)
    else:
        out = augment_faiss(queries, Path(args.index_dir), args.top_k)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} augmented queries ({args.mode}) to {outp}")


if __name__ == "__main__":
    main()
