"""
Build a CPU FAISS index + sentence-transformer embeddings for RAG retrieval.

Artifacts under --output_dir (default outputs/rag_index/):
  - meta.json       : model name, dim, chunk list (id, title, text)
  - embeddings.npy  : float32 (n, d) L2-normalized rows (optional but useful for audit)
  - index.faiss     : FAISS IndexFlatIP (inner product = cosine sim on normalized vectors)

Example:
  python scripts/build_rag_index.py --corpus data/rag_corpus.json --output_dir outputs/rag_index
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lib.rag_corpus import load_corpus_chunks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/rag_corpus.json", help="JSON list or .jsonl corpus.")
    p.add_argument("--output_dir", default="outputs/rag_index")
    p.add_argument(
        "--model_name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformer model (downloads on first use).",
    )
    p.add_argument("--seed", type=int, default=42, help="Reserved for future stochastic steps.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_corpus_chunks(args.corpus)
    texts = [f"{c.get('title', '')}\n{c['text']}" for c in chunks]

    import faiss  # noqa: PLC0415 — heavy import after argparse
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(args.model_name)
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 32,
    )
    emb = np.asarray(emb, dtype=np.float32)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    np.save(out_dir / "embeddings.npy", emb)

    meta = {
        "model_name": args.model_name,
        "embedding_dim": dim,
        "num_chunks": len(chunks),
        "corpus_path": str(Path(args.corpus).resolve()),
        "seed": args.seed,
        "chunks": chunks,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote FAISS index and meta for {len(chunks)} chunks -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
