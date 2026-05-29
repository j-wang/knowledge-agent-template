#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "scipy",
# ]
# ///
"""
Build doc embeddings + doc-to-doc similarity for the Knowledge Base.

PROVIDER-AGNOSTIC. The embedding provider is chosen via env config
(retrieval_config.py / .env) — there is no hardcoded model or endpoint:

    KB_EMBED_PROVIDER = none (default) | openai | tei | sentence-transformers

What it produces:
  * `.doc_embeddings.npz` — normalized doc vectors + doc_ids + provider/model
    metadata (for cache invalidation). This is what search.py uses for true
    query->doc dense retrieval (RRF-fused with BM25).
  * `.similarity.npy` — the doc-to-doc cosine matrix used ONLY by search.py's
    `--related` display feature.

Behaviour by provider:
  * KB_EMBED_PROVIDER=none (default): no dense retrieval. Builds the doc-doc
    similarity matrix via local TF-IDF (so `--related` still works fully
    offline) and prints that dense retrieval is disabled until you configure
    an embed provider.
  * Any real provider (openai/tei/sentence-transformers): embeds every doc,
    saves `.doc_embeddings.npz`, and derives `.similarity.npy` from those
    vectors.

Run:
    uv run build_embeddings.py

See .env.example for provider config blocks (OpenAI, TEI, local
sentence-transformers).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from retrieval_config import load_config
from providers import get_embedder

SCRIPT_DIR = Path(__file__).parent.resolve()
CONCEPTS_DIR = SCRIPT_DIR / "extracted" / "concepts" / "docs"
THESES_DIR = SCRIPT_DIR / "extracted" / "concepts" / "theses"
SIMILARITY_PATH = SCRIPT_DIR / ".similarity.npy"
DOC_EMBEDDINGS_PATH = SCRIPT_DIR / ".doc_embeddings.npz"
TFIDF_PATH = SCRIPT_DIR / ".tfidf_matrix.npz"

EMBED_BATCH = 100


# ---------------------------------------------------------------------------
# Document loading (mirrors search.py logic)
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
    """Text to embed for each doc. MUST match search.build_embed_text so the
    query is scored against the same doc representation."""
    parts = [doc["title"], doc["title"], doc["title"]]
    if doc["tags"]:
        parts.append("Topics: " + ", ".join(doc["tags"]))
    if doc["summary"]:
        parts.append(doc["summary"])
    if doc["keywords"]:
        parts.append(doc["keywords"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# TF-IDF doc-doc similarity (offline fallback for --related)
# ---------------------------------------------------------------------------

def build_tfidf_text(doc: dict) -> str:
    parts = [doc["title"], doc["title"], doc["title"]]
    if doc["tags"]:
        parts.append(" ".join(doc["tags"]))
        parts.append(" ".join(doc["tags"]))
    if doc["summary"]:
        parts.append(doc["summary"])
    if doc["keywords"]:
        parts.append(doc["keywords"])
    return "\n".join(parts)


def build_tfidf_similarity(docs: list[dict]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy import sparse

    print(f"Building TF-IDF vectors for {len(docs)} documents...", file=sys.stderr)
    texts = [build_tfidf_text(doc) for doc in docs]
    n_docs = len(docs)
    min_df = min(2, n_docs) if n_docs > 0 else 1
    max_df = max(0.8, (n_docs - 0.5) / n_docs) if n_docs > 1 else 1.0
    vectorizer = TfidfVectorizer(
        max_features=10000, min_df=min_df, max_df=max_df,
        ngram_range=(1, 2), sublinear_tf=True, stop_words='english',
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(tfidf_matrix)
    sparse.save_npz(TFIDF_PATH, tfidf_matrix)
    return similarity


# ---------------------------------------------------------------------------
# Embedding (batched, via the configured provider)
# ---------------------------------------------------------------------------

def embed_corpus(embedder, texts: list[str]) -> np.ndarray | None:
    out: list[np.ndarray] = []
    n_batches = (len(texts) - 1) // EMBED_BATCH + 1 if texts else 0
    for bi, i in enumerate(range(0, len(texts), EMBED_BATCH), start=1):
        batch = texts[i:i + EMBED_BATCH]
        print(f"  Embedding batch {bi}/{n_batches} ({len(batch)} docs)...",
              flush=True)
        vecs = embedder.embed(batch)
        if vecs is None:
            return None  # provider already warned; caller degrades
        out.append(np.asarray(vecs, dtype=np.float32))
    if not out:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(out)


def normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()
    embed_cfg = config.embed

    print("Loading documents...")
    docs = load_docs()
    if not docs:
        print("No documents found under extracted/concepts/docs|theses/. "
              "Add docs and re-run.", file=sys.stderr)
        sys.exit(1)
    n_concepts = sum(1 for d in docs if d['type'] == 'concept')
    n_theses = sum(1 for d in docs if d['type'] == 'thesis')
    print(f"  {len(docs)} documents ({n_concepts} concepts, {n_theses} theses)")

    # --- No embed provider: TF-IDF doc-doc only, dense retrieval disabled. ---
    if not embed_cfg.enabled:
        print("\nKB_EMBED_PROVIDER=none — dense retrieval is DISABLED. "
              "search.py will use BM25 only.")
        print("Building local TF-IDF doc-to-doc similarity for --related...")
        similarity = build_tfidf_similarity(docs)
        np.save(SIMILARITY_PATH, similarity)
        # A stale .doc_embeddings.npz must not silently enable dense retrieval
        # against the wrong vectors; remove it.
        if DOC_EMBEDDINGS_PATH.exists():
            DOC_EMBEDDINGS_PATH.unlink()
            print(f"  Removed stale {DOC_EMBEDDINGS_PATH.name} "
                  "(no embed provider configured).")
        print(f"\nSaved {SIMILARITY_PATH.name} ({similarity.shape}).")
        print("To enable dense retrieval, set KB_EMBED_PROVIDER (see .env.example).")
        return

    # --- Real embed provider: embed corpus -> dense + similarity. ---
    embedder = get_embedder(embed_cfg)
    if not embedder.available():
        print(f"\nembed provider={embed_cfg.provider} is configured but not "
              "available (see warning above). No embeddings written; "
              "search.py will use BM25 only.", file=sys.stderr)
        sys.exit(1)

    print(f"\nEmbedding via provider={embed_cfg.provider} "
          f"model={embed_cfg.model or '(provider default)'}...")
    texts = [build_embed_text(doc) for doc in docs]
    embeddings = embed_corpus(embedder, texts)
    if embeddings is None or embeddings.size == 0:
        print("Embedding failed (provider returned no vectors). Nothing written; "
              "search.py will use BM25 only.", file=sys.stderr)
        sys.exit(1)

    embeddings = normalize(embeddings.astype(np.float32))
    print(f"  Embeddings shape: {embeddings.shape}")

    doc_ids = np.array([d["id"] for d in docs], dtype=object)
    np.savez_compressed(
        DOC_EMBEDDINGS_PATH,
        embeddings=embeddings,
        doc_ids=doc_ids,
        provider=np.array(embed_cfg.provider),
        model=np.array(embed_cfg.model or ""),
    )
    print(f"  Saved {DOC_EMBEDDINGS_PATH.name} (normalized doc vectors + metadata)")

    # Doc-to-doc similarity from the same vectors (for --related).
    print("Computing doc-to-doc similarity matrix from embeddings...")
    similarity = embeddings @ embeddings.T
    np.save(SIMILARITY_PATH, similarity)
    print(f"  Saved {SIMILARITY_PATH.name} ({similarity.shape})")

    # Sanity check.
    if len(docs) > 1:
        print("\n=== Sanity check: top-3 most similar pairs ===")
        pairs = []
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                pairs.append((float(similarity[i][j]), i, j))
        pairs.sort(key=lambda x: -x[0])
        for sim, i, j in pairs[:3]:
            print(f"  {sim:.3f}: [{docs[i]['type'][0].upper()}] {docs[i]['title']}")
            print(f"         [{docs[j]['type'][0].upper()}] {docs[j]['title']}")

    print("\nDone. Run: python3 search.py --rebuild --bm25-only   (refresh BM25 index)")
    print("Then:    python3 search.py \"your query\"            (dense+BM25 hybrid via RRF)")


if __name__ == "__main__":
    main()
