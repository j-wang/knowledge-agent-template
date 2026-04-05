#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai",
#     "numpy",
# ]
# ///
"""
Build neural embeddings and similarity matrix for the Knowledge Base.

Run this on a machine with OpenAI API access:
    uv run build_embeddings.py

Reads API key from .env (OPENAI_KEY=sk-...) or OPENAI_API_KEY env var.
Outputs .similarity.npy which search.py picks up automatically.
"""

import os
import re
import sys
import time
import numpy as np
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai numpy")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent.resolve()
CONCEPTS_DIR = SCRIPT_DIR / "extracted" / "concepts" / "docs"
THESES_DIR = SCRIPT_DIR / "extracted" / "concepts" / "theses"
SIMILARITY_PATH = SCRIPT_DIR / ".similarity.npy"
EMBEDDINGS_PATH = SCRIPT_DIR / ".embeddings.npz"
DOC_ORDER_PATH = SCRIPT_DIR / ".embedding_doc_order.json"
ENV_PATH = SCRIPT_DIR / ".env"

MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Load API key
# ---------------------------------------------------------------------------

def load_api_key() -> str:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_KEY="):
                return line.split("=", 1)[1].strip()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("ERROR: No API key found. Set OPENAI_KEY in .env or OPENAI_API_KEY env var.")
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# Document loading (mirrors search.py logic exactly)
# ---------------------------------------------------------------------------

def parse_frontmatter(content: str) -> dict:
    fm = {}
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return fm
    raw = match.group(1)
    for key in ["id", "title"]:
        m = re.search(rf'^{key}:\s*(.+)', raw, re.MULTILINE)
        if m:
            fm[key] = m.group(1).strip().strip('"\'')
    tags_match = re.search(r'tags:\n((?:\s+-\s+.+\n)+)', raw)
    if tags_match:
        fm["tags"] = [t.strip().strip('"\'') for t in
                       re.findall(r'^\s+-\s+(.+)', tags_match.group(1), re.MULTILINE)]
    return fm


def extract_section(content: str, header: str, max_chars: int = 500) -> str:
    m = re.search(rf'## {header}\n(.+?)(?:\n##|\Z)', content, re.DOTALL)
    if m:
        text = m.group(1).strip()
        return text[:max_chars] if len(text) > max_chars else text
    return ""


def load_docs() -> list[dict]:
    docs = []
    for doc_type, directory in [("concept", CONCEPTS_DIR), ("thesis", THESES_DIR)]:
        if not directory.exists():
            continue
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith('.md') or fname.startswith('_'):
                continue
            fpath = directory / fname
            content = fpath.read_text()
            fm = parse_frontmatter(content)
            title = fm.get("title", fname.replace('.md', '').replace('-', ' '))
            tags = fm.get("tags", [])
            summary = extract_section(content, "(?:Summary|Thesis Statement)", 400)
            keywords = extract_section(content, "Keywords", 300)

            docs.append({
                "path": str(fpath),
                "type": doc_type,
                "id": fm.get("id", fname.replace('.md', '')),
                "title": title,
                "tags": tags,
                "summary": summary,
                "keywords": keywords,
            })
    return docs


def build_embed_text(doc: dict) -> str:
    """Build text to embed. Title (3x), tags, summary, keywords."""
    parts = [doc["title"], doc["title"], doc["title"]]
    if doc["tags"]:
        parts.append("Topics: " + ", ".join(doc["tags"]))
    if doc["summary"]:
        parts.append(doc["summary"])
    if doc["keywords"]:
        parts.append(doc["keywords"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Embed via OpenAI
# ---------------------------------------------------------------------------

def embed_all(texts: list[str], client: OpenAI) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"  Embedding batch {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1} "
              f"({len(batch)} docs)...", flush=True)
        resp = client.embeddings.create(input=batch, model=MODEL)
        data = sorted(resp.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in data])
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.3)
    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)

    print("Loading documents...")
    docs = load_docs()
    n_concepts = sum(1 for d in docs if d['type'] == 'concept')
    n_theses = sum(1 for d in docs if d['type'] == 'thesis')
    print(f"  {len(docs)} documents ({n_concepts} concepts, {n_theses} theses)")

    print(f"\nEmbedding via {MODEL}...")
    texts = [build_embed_text(doc) for doc in docs]
    embeddings = embed_all(texts, client)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_normed = embeddings / norms

    # Compute similarity matrix
    print("Computing doc-to-doc similarity matrix...")
    similarity = embeddings_normed @ embeddings_normed.T
    print(f"  Similarity matrix shape: {similarity.shape}")

    # Save
    np.save(SIMILARITY_PATH, similarity)
    np.savez_compressed(EMBEDDINGS_PATH, embeddings=embeddings_normed)

    # Save doc order for verification
    import json
    doc_order = [{"id": d["id"], "title": d["title"], "type": d["type"]} for d in docs]
    DOC_ORDER_PATH.write_text(json.dumps(doc_order, indent=2))

    print(f"\nSaved:")
    print(f"  {SIMILARITY_PATH} ({similarity.nbytes / 1024:.0f} KB)")
    print(f"  {EMBEDDINGS_PATH}")
    print(f"  {DOC_ORDER_PATH}")

    # Sanity check
    print(f"\n=== Sanity check ===")
    if len(docs) > 1:
        print(f"  Showing top-3 most similar pairs:")
        top_pairs = []
        for i in range(len(docs)):
            for j in range(i+1, len(docs)):
                top_pairs.append((similarity[i][j], i, j))
        top_pairs.sort(key=lambda x: -x[0])
        for sim, i, j in top_pairs[:3]:
            print(f"    {sim:.3f}: [{docs[i]['type'][0].upper()}] {docs[i]['title']}")
            print(f"           [{docs[j]['type'][0].upper()}] {docs[j]['title']}")

    print("\nDone! Run 'python3 search.py --rebuild --bm25-only' to rebuild BM25,")
    print("then 'python3 search.py \"your query\" --related' to use hybrid search.")


if __name__ == "__main__":
    main()
