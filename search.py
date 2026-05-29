#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "rank-bm25",
#     "scikit-learn",
#     "scipy",
# ]
# ///
"""
Knowledge Base Search — BM25 + optional dense retrieval + optional reranking.

Domain-agnostic search over a concept/thesis knowledge base.

Retrieval pipeline (all stages past BM25 are opt-in via env config — see
retrieval_config.py and .env.example):

  1. BM25 (rank_bm25) over keyword-engineered text. ALWAYS on, local, no network.
  2. Dense query->doc retrieval: if an embedder is configured AND
     `.doc_embeddings.npz` is present, the QUERY is embedded and scored against
     the stored doc vectors. BM25 and dense lists are fused via Reciprocal Rank
     Fusion (RRF, k=60).
  3. Reranking: if a reranker is configured, the top-N fused candidates are
     reordered by a cross-encoder / rerank API.

ZERO CONFIG = BM25 only, fully offline. With KB_EMBED_PROVIDER / KB_RERANK_PROVIDER
set, the extra stages activate. Every network stage degrades gracefully: if the
embed/rerank server is down, the doc embeddings are missing, or a key is absent,
search transparently falls back to the next-best available ranking with a single
stderr note — it never crashes.

The doc-to-doc `.similarity.npy` matrix is no longer used for primary ranking
(benchmarks showed the old "semantic expansion" hurt vs plain BM25). It is kept
ONLY for the `--related` display feature.

Usage:
    python3 search.py "your query here"
    python3 search.py "your query here" --top 20
    python3 search.py "your query here" --type concepts
    python3 search.py "your query here" --type theses
    python3 search.py "your query here" --verbose
    python3 search.py "your query here" --related   # show related docs for each hit
    python3 search.py "your query here" --bm25-only  # force BM25, skip dense/rerank
    python3 search.py --rebuild                      # rebuild BM25 index + TF-IDF similarity
    python3 search.py --rebuild --bm25-only          # rebuild BM25 only (skip TF-IDF)
"""

import os
import sys
import json
import re
import pickle
import argparse
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval_config import load_config
from providers import get_embedder, get_reranker

SCRIPT_DIR = Path(__file__).parent.resolve()
EXTRACTED_DIR = SCRIPT_DIR / "extracted"
CONCEPTS_DIR = EXTRACTED_DIR / "concepts" / "docs"
THESES_DIR = EXTRACTED_DIR / "concepts" / "theses"
INDEX_PATH = SCRIPT_DIR / ".search_index.pkl"
SIMILARITY_PATH = SCRIPT_DIR / ".similarity.npy"
TFIDF_PATH = SCRIPT_DIR / ".tfidf_matrix.npz"
DOC_EMBEDDINGS_PATH = SCRIPT_DIR / ".doc_embeddings.npz"

# RRF constant. 60 is the canonical default from Cormack et al. (2009).
RRF_K = 60


# ---------------------------------------------------------------------------
# Tokenizer (shared between BM25 indexing and query processing)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "that", "this", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "he", "she", "his", "her", "which", "what", "who", "whom",
}

def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter fields."""
    fm = {}
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return fm
    raw = match.group(1)
    for key in ["id", "title", "category", "depth", "type", "analytical_angle", "source"]:
        m = re.search(rf'^{key}:\s*(.+)', raw, re.MULTILINE)
        if m:
            fm[key] = m.group(1).strip().strip('"\'')
    # Extract tags
    tags_match = re.search(r'tags:\n((?:\s+-\s+.+\n)+)', raw)
    if tags_match:
        fm["tags"] = [t.strip().strip('"\'') for t in re.findall(r'^\s+-\s+(.+)', tags_match.group(1), re.MULTILINE)]
    return fm


def extract_keywords(content: str) -> str:
    """Pull the Keywords section."""
    m = re.search(r'## Keywords\n(.+?)(?:\n##|\Z)', content, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_summary(content: str) -> str:
    """Pull the Summary section (first 2-3 sentences)."""
    m = re.search(r'## (?:Summary|Thesis Statement)\n(.+?)(?:\n##)', content, re.DOTALL)
    if m:
        text = m.group(1).strip()
        if len(text) > 300:
            return text[:300].rsplit(' ', 1)[0] + "..."
        return text
    return ""


def load_docs() -> list[dict]:
    """Load all concept and thesis docs with metadata and content."""
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
            keywords = extract_keywords(content)
            summary = extract_summary(content)
            title = fm.get("title", fname.replace('.md', '').replace('-', ' '))
            tags = fm.get("tags", [])

            docs.append({
                "path": str(fpath),
                "relative_path": f"{doc_type}s/{fname}",
                "type": doc_type,
                "id": fm.get("id", fname.replace('.md', '')),
                "title": title,
                "category": fm.get("category", fm.get("analytical_angle", "")),
                "source": fm.get("source", ""),
                "tags": tags,
                "summary": summary,
                "depth": fm.get("depth", ""),
                "keywords": keywords,
                "content": content,
            })
    return docs


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def build_bm25(docs: list[dict]) -> BM25Okapi:
    """Build BM25 index from docs."""
    corpus = []
    for doc in docs:
        search_text = (
            f"{doc['title']} {doc['title']} {doc['title']} "
            f"{' '.join(doc['tags'])} {' '.join(doc['tags'])} "
            f"{doc['keywords']} {doc['keywords']} "
            f"{doc['content']}"
        )
        corpus.append(tokenize(search_text))
    return BM25Okapi(corpus)


# ---------------------------------------------------------------------------
# TF-IDF Semantic Similarity
# ---------------------------------------------------------------------------

def build_embed_text(doc: dict) -> str:
    """Build the text to embed for a doc — shared between build_embeddings.py
    (corpus side) and search.py (so the doc-text representation matches).

    Title (3x weighted by repetition), tags, summary, keywords. Kept identical
    in spirit to build_tfidf_text so dense and TF-IDF paths see the same fields.
    The QUERY is embedded raw (no field engineering) at search time.
    """
    parts = [doc["title"], doc["title"], doc["title"]]
    if doc.get("tags"):
        parts.append("Topics: " + ", ".join(doc["tags"]))
    if doc.get("summary"):
        parts.append(doc["summary"])
    if doc.get("keywords"):
        parts.append(doc["keywords"])
    return "\n".join(parts)


def build_tfidf_text(doc: dict) -> str:
    """Build the text for TF-IDF vectorization.

    Uses title (3x weighted by repetition), tags, summary, and keywords.
    This mirrors what we'd send to an embedding API but stays fully local.
    """
    parts = [
        doc["title"], doc["title"], doc["title"],  # Title weight boost
    ]
    if doc["tags"]:
        parts.append(" ".join(doc["tags"]))
        parts.append(" ".join(doc["tags"]))  # Tags double-weighted
    if doc["summary"]:
        parts.append(doc["summary"])
    if doc["keywords"]:
        parts.append(doc["keywords"])
    if doc.get("content"):
        # Include first ~500 chars of body for broader vocabulary
        parts.append(doc["content"][:500])
    return "\n".join(parts)


def build_tfidf_similarity(docs: list[dict]) -> np.ndarray:
    """Build TF-IDF vectors and compute cosine similarity matrix.

    Fully local — no API calls. Returns (N, N) similarity matrix.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"Building TF-IDF vectors for {len(docs)} documents...", file=sys.stderr)
    texts = [build_tfidf_text(doc) for doc in docs]

    # Adapt TF-IDF params for small corpora
    n_docs = len(docs)
    min_df = min(2, n_docs) if n_docs > 0 else 1
    max_df = max(0.8, (n_docs - 0.5) / n_docs) if n_docs > 1 else 1.0

    vectorizer = TfidfVectorizer(
        max_features=10000,     # Cap vocabulary for efficiency
        min_df=min_df,          # Term must appear in at least 2 docs (or 1 if small corpus)
        max_df=max_df,          # Skip terms in >80% of docs (relaxed for small corpus)
        ngram_range=(1, 2),     # Unigrams and bigrams for phrase matching
        sublinear_tf=True,      # Dampens raw term frequency
        stop_words='english',
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  Vocabulary: {len(vectorizer.vocabulary_)} terms, matrix: {tfidf_matrix.shape}", file=sys.stderr)

    # Compute cosine similarity
    print("Computing doc-to-doc similarity matrix...", file=sys.stderr)
    similarity = cosine_similarity(tfidf_matrix)

    # Save the sparse TF-IDF matrix for potential future use
    from scipy import sparse
    sparse.save_npz(TFIDF_PATH, tfidf_matrix)

    return similarity


# ---------------------------------------------------------------------------
# Index build / load
# ---------------------------------------------------------------------------

def build_and_save_index(bm25_only: bool = False):
    """Build all indexes and save to disk."""
    docs = load_docs()
    if not docs:
        print("No documents found. Add concept/thesis docs to extracted/concepts/docs/ and theses/",
              file=sys.stderr)
        return None, []
    bm25 = build_bm25(docs)

    # Strip content from stored docs (saves space in pickle)
    stored_docs = []
    for d in docs:
        sd = {k: v for k, v in d.items() if k != "content"}
        stored_docs.append(sd)

    with open(INDEX_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'docs': stored_docs}, f)

    n_concepts = sum(1 for d in stored_docs if d['type'] == 'concept')
    n_theses = sum(1 for d in stored_docs if d['type'] == 'thesis')
    print(f"Indexed {len(stored_docs)} documents ({n_concepts} concepts, {n_theses} theses)",
          file=sys.stderr)

    if not bm25_only:
        similarity = build_tfidf_similarity(docs)
        np.save(SIMILARITY_PATH, similarity)
        print(f"Saved similarity matrix ({similarity.shape})", file=sys.stderr)
    else:
        print("Skipping TF-IDF similarity (--bm25-only)", file=sys.stderr)

    return bm25, stored_docs


def load_index() -> tuple:
    """Load cached BM25 index."""
    if INDEX_PATH.exists():
        try:
            with open(INDEX_PATH, 'rb') as f:
                data = pickle.load(f)
            return data['bm25'], data['docs']
        except Exception:
            pass
    # Fallback: build without embeddings
    docs = load_docs()
    if not docs:
        print("No documents found. Run --rebuild after adding concept/thesis docs.", file=sys.stderr)
        sys.exit(1)
    bm25 = build_bm25(docs)
    stored_docs = [{k: v for k, v in d.items() if k != "content"} for d in docs]
    return bm25, stored_docs


def load_similarity() -> np.ndarray | None:
    """Load precomputed doc-doc similarity matrix if available.

    Used ONLY for the `--related` display feature now (not primary ranking)."""
    if SIMILARITY_PATH.exists():
        try:
            return np.load(SIMILARITY_PATH)
        except Exception:
            pass
    return None


def load_doc_embeddings(docs) -> np.ndarray | None:
    """Load precomputed normalized doc vectors for dense retrieval.

    Returns an (N, D) float32 matrix aligned to `docs`, or None if the file is
    missing or doesn't line up with the current index (stale embeddings ->
    fall back to BM25 rather than mis-rank).
    """
    if not DOC_EMBEDDINGS_PATH.exists():
        return None
    try:
        data = np.load(DOC_EMBEDDINGS_PATH, allow_pickle=True)
        emb = data["embeddings"].astype(np.float32)
    except Exception as exc:
        print(f"[retrieval] could not load {DOC_EMBEDDINGS_PATH.name} "
              f"({type(exc).__name__}) — using BM25 only.", file=sys.stderr)
        return None

    # Alignment check: prefer doc_ids, fall back to row count.
    try:
        stored_ids = [str(x) for x in data["doc_ids"]]
        current_ids = [str(d.get("id", "")) for d in docs]
        if stored_ids != current_ids:
            print("[retrieval] .doc_embeddings.npz is stale (doc set changed) — "
                  "using BM25 only. Re-run build_embeddings.py.", file=sys.stderr)
            return None
    except KeyError:
        if emb.shape[0] != len(docs):
            print("[retrieval] .doc_embeddings.npz row count != index — "
                  "using BM25 only. Re-run build_embeddings.py.", file=sys.stderr)
            return None
    return emb


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_bm25(query: str, bm25, docs, top_k=10, doc_type=None) -> list:
    """BM25 lexical search. Returns [(score, doc), ...]."""
    tokens = tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    results = list(zip(scores, range(len(docs))))

    if doc_type:
        results = [(s, i) for s, i in results if docs[i]['type'] == doc_type]

    results.sort(key=lambda x: -x[0])
    return [(s, i) for s, i in results[:top_k] if s > 0]


def search_bm25_ranked(query: str, bm25, docs, doc_type=None) -> list[int]:
    """Full BM25 ranking (all docs with score > 0), most-relevant first.
    Returns a list of doc indices."""
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    order = [(scores[i], i) for i in range(len(docs))]
    if doc_type:
        order = [(s, i) for s, i in order if docs[i]['type'] == doc_type]
    order.sort(key=lambda x: -x[0])
    return [i for s, i in order if s > 0]


def dense_ranked(query: str, embedder, doc_emb, docs, doc_type=None) -> list[int]:
    """Embed the query, cosine-rank docs. Returns doc indices best-first, or []
    if the embedder is unavailable / the call fails (caller degrades to BM25)."""
    if embedder is None or doc_emb is None:
        return []
    q = embedder.embed([query])
    if q is None or len(q) == 0:
        return []
    q = np.asarray(q, dtype=np.float32)[0]
    # doc_emb rows are normalized; normalize the query too -> dot == cosine.
    qn = np.linalg.norm(q)
    if qn:
        q = q / qn
    sims = doc_emb @ q
    order = list(np.argsort(-sims))
    if doc_type:
        order = [int(i) for i in order if docs[int(i)]['type'] == doc_type]
    else:
        order = [int(i) for i in order]
    return order


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = RRF_K) -> list[tuple[float, int]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    Each input is a ranked list of item ids (best first). Score for an item is
    sum over lists of 1 / (k + rank), rank being 1-based. Returns
    [(score, item_id), ...] sorted by score descending.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    fused = [(s, i) for i, s in scores.items()]
    fused.sort(key=lambda x: -x[0])
    return fused


def apply_rerank(query: str, reranker, candidates: list[int], docs, top_n: int) -> list[int] | None:
    """Rerank the top `top_n` candidate doc indices with the reranker.

    Returns the reordered FULL candidate list (reranked head + untouched tail),
    or None if the reranker is unavailable / fails (caller keeps fused order).
    """
    if reranker is None or not candidates:
        return None
    head = candidates[:top_n]
    tail = candidates[top_n:]
    doc_texts = [build_embed_text(docs[i]) for i in head]
    scores = reranker.rerank(query, doc_texts)
    if scores is None or len(scores) != len(head):
        return None
    reordered = [i for _, i in sorted(zip(scores, head), key=lambda x: -x[0])]
    return reordered + tail


def search(query: str, bm25, docs, top_k=10, doc_type=None,
           config=None, embedder=None, reranker=None, doc_emb=None,
           force_bm25=False) -> list:
    """Primary retrieval pipeline.

    Returns [(score, doc_index, source), ...] where source ∈
    {'bm25','dense','hybrid','rerank'} and score is a display-only rank-fusion
    or BM25 score. Honors config.mode and degrades gracefully at every stage.
    """
    if config is None:
        config = load_config()

    bm25_list = search_bm25_ranked(query, bm25, docs, doc_type=doc_type)

    # Decide effective mode.
    mode = "bm25" if force_bm25 else config.mode
    embedder_ok = (not force_bm25 and embedder is not None
                   and embedder.available() and doc_emb is not None)

    dense_list: list[int] = []
    if mode in ("auto", "dense", "hybrid") and embedder_ok:
        dense_list = dense_ranked(query, embedder, doc_emb, docs, doc_type=doc_type)

    # Resolve auto -> concrete mode based on what's actually available.
    if mode == "auto":
        mode = "hybrid" if dense_list else "bm25"

    if mode == "dense":
        if dense_list:
            ordered = dense_list
            source = "dense"
        else:  # dense requested but unavailable -> degrade
            print("[retrieval] dense mode requested but embeddings unavailable "
                  "— falling back to BM25.", file=sys.stderr)
            ordered = bm25_list
            source = "bm25"
    elif mode == "hybrid" and dense_list:
        fused = reciprocal_rank_fusion([bm25_list, dense_list])
        ordered = [i for _, i in fused]
        source = "hybrid"
    else:  # bm25 (explicit, or hybrid with no dense available)
        ordered = bm25_list
        source = "bm25"

    # Optional rerank pass over the top-N fused candidates.
    if (not force_bm25 and reranker is not None and reranker.available()
            and ordered):
        reranked = apply_rerank(query, reranker, ordered, docs,
                                config.rerank.top_n)
        if reranked is not None:
            ordered = reranked
            source = "rerank"

    # Build display scores: descending synthetic score so the ranking is clear.
    n = len(ordered[:top_k])
    results = []
    for rank, idx in enumerate(ordered[:top_k]):
        score = (n - rank) / n if n else 0.0
        results.append((score, idx, source))
    return results


def get_related_docs(doc_index: int, docs, similarity, top_k=5) -> list:
    """Get docs most similar to a given doc via similarity matrix."""
    if similarity is None:
        return []
    sims = similarity[doc_index]
    threshold = np.percentile(sims, 95)
    ranked = np.argsort(-sims)
    results = []
    for idx in ranked:
        if int(idx) == doc_index:
            continue
        if sims[idx] < threshold:
            break
        results.append((float(sims[idx]), int(idx)))
        if len(results) >= top_k:
            break
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_results(results, docs, verbose=False, similarity=None, show_related=False) -> str:
    """Format search results for display."""
    if not results:
        return "No results found."

    lines = []
    for i, item in enumerate(results, 1):
        if len(item) == 3:
            score, idx, source = item
        else:
            score, idx = item
            source = 'bm25'

        doc = docs[idx]
        marker = "T" if doc['type'] == 'thesis' else "C"
        depth = f" [{doc['depth']}]" if doc.get('depth') else ""
        source_tag = f"  [{source}]" if source not in ('bm25', '') else ""
        lines.append(f"{i:2d}. [{marker}] {doc['title']}{depth}  (score: {score:.3f}){source_tag}")
        lines.append(f"    path: {doc['path']}")
        if verbose:
            if doc.get('category'):
                lines.append(f"    category: {doc['category']}")
            if doc.get('source'):
                lines.append(f"    source: {doc['source']}")
            if doc.get('tags'):
                lines.append(f"    tags: {', '.join(doc['tags'][:8])}")
            if doc.get('summary'):
                lines.append(f"    summary: {doc['summary']}")

        if show_related and similarity is not None:
            related = get_related_docs(idx, docs, similarity, top_k=3)
            if related:
                lines.append(f"    related:")
                for sim, ridx in related:
                    rdoc = docs[ridx]
                    rm = "T" if rdoc['type'] == 'thesis' else "C"
                    lines.append(f"      [{rm}] {rdoc['title']} (sim: {sim:.2f})")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Search the Knowledge Base")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--type", choices=["concepts", "theses"], help="Filter by doc type")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show extra detail")
    parser.add_argument("--related", "-r", action="store_true",
                        help="Show semantically related docs for each hit")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild all indexes")
    parser.add_argument("--bm25-only", action="store_true",
                        help="Force BM25-only retrieval (skip dense + rerank). "
                             "With --rebuild: also skip rebuilding TF-IDF similarity.")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.rebuild:
        print("Rebuilding index...", file=sys.stderr)
        bm25, docs = build_and_save_index(bm25_only=args.bm25_only)
        if not args.query:
            return
    else:
        bm25, docs = load_index()

    if not args.query:
        parser.print_help()
        return

    similarity = load_similarity()

    doc_type = None
    if args.type:
        doc_type = args.type.rstrip('s')

    # Wire up the optional dense + rerank stages from env config.
    config = load_config()
    embedder = get_embedder(config.embed)
    reranker = get_reranker(config.rerank)
    doc_emb = load_doc_embeddings(docs)

    results = search(args.query, bm25, docs, top_k=args.top, doc_type=doc_type,
                     config=config, embedder=embedder, reranker=reranker,
                     doc_emb=doc_emb, force_bm25=args.bm25_only)

    if args.json:
        out = []
        for item in results:
            score, idx, source = item if len(item) == 3 else (*item, 'bm25')
            doc = docs[idx]
            entry = {
                "score": round(score, 4),
                "path": doc["path"],
                "type": doc["type"],
                "id": doc["id"],
                "title": doc["title"],
                "category": doc.get("category", ""),
                "source": source,
            }
            if similarity is not None:
                related = get_related_docs(idx, docs, similarity, top_k=3)
                entry["related"] = [
                    {"title": docs[ri]["title"], "similarity": round(rs, 3)}
                    for rs, ri in related
                ]
            out.append(entry)
        print(json.dumps(out, indent=2))
    else:
        print(format_results(results, docs, verbose=args.verbose,
                             similarity=similarity, show_related=args.related))


if __name__ == "__main__":
    main()
