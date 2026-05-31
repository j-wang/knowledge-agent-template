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

import sys
from pathlib import Path

import numpy as np

from .retrieval_config import load_config
from .providers import get_embedder
from .kb_docs import load_docs, build_embed_text, build_tfidf_similarity
from ._root import KB_ROOT

SIMILARITY_PATH = KB_ROOT / ".similarity.npy"
DOC_EMBEDDINGS_PATH = KB_ROOT / ".doc_embeddings.npz"

EMBED_BATCH = 100


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
        similarity = build_tfidf_similarity(docs, verbose=True)
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
