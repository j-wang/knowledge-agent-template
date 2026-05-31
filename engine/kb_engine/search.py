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

import sys
import json
import re
import pickle
import argparse
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .retrieval_config import load_config
from .providers import get_embedder, get_reranker
from .kb_docs import (
    load_docs,
    build_rerank_text,
    build_tfidf_similarity,
)
from ._root import KB_ROOT

INDEX_PATH = KB_ROOT / ".search_index.pkl"
SIMILARITY_PATH = KB_ROOT / ".similarity.npy"
DOC_EMBEDDINGS_PATH = KB_ROOT / ".doc_embeddings.npz"

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
        print(f"Building TF-IDF doc-to-doc similarity for {len(docs)} documents...",
              file=sys.stderr)
        similarity = build_tfidf_similarity(docs, verbose=True)
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


def load_similarity(docs=None) -> np.ndarray | None:
    """Load precomputed doc-doc similarity matrix if available.

    Used ONLY for the `--related` display feature now (not primary ranking).
    Returns None (so `--related` is silently skipped) if the file is missing or
    its shape no longer matches the current corpus — a stale `.similarity.npy`
    would otherwise crash get_related_docs() with an index error."""
    if not SIMILARITY_PATH.exists():
        return None
    try:
        sim = np.load(SIMILARITY_PATH)
    except Exception:
        return None
    if docs is not None:
        n = len(docs)
        if sim.ndim != 2 or sim.shape != (n, n):
            print("[retrieval] .similarity.npy is stale (shape != current corpus) "
                  "— '--related' disabled. Re-run search.py --rebuild.",
                  file=sys.stderr)
            return None
    return sim


def load_doc_embeddings(docs, embed_cfg=None) -> np.ndarray | None:
    """Load precomputed normalized doc vectors for dense retrieval.

    Returns an (N, D) float32 matrix aligned to `docs`, or None if the file is
    missing or doesn't line up with the current index/config (stale embeddings
    -> fall back to BM25 rather than mis-rank or crash).

    Staleness is checked three ways, each of which has bitten us:
      * doc set changed   -> doc_ids differ (or, legacy files, row count differs)
      * embed config swap  -> the provider/model recorded in the npz differs from
        the currently-configured one. Critically, changing KB_EMBED_MODEL to a
        model with a DIFFERENT vector dimension would otherwise sail past the
        doc_ids check and blow up later in `doc_emb @ query` with a shape error.
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

    # Shape sanity: must be a 2-D (N_docs, D) matrix. A 1-D, empty, or
    # wrong-row-count array (malformed/legacy file) would otherwise crash the
    # `doc_emb.shape[1]` dim guard or the `doc_emb @ query` matmul downstream.
    if emb.ndim != 2 or emb.shape[0] != len(docs) or emb.shape[1] == 0:
        print(f"[retrieval] .doc_embeddings.npz has unexpected shape "
              f"{emb.shape} for {len(docs)} docs — using BM25 only. "
              f"Re-run build_embeddings.py.", file=sys.stderr)
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

    # Config check: provider/model must match what built these vectors. This is
    # what catches an embed-model swap (and therefore a dimension change).
    if embed_cfg is not None:
        try:
            stored_provider = str(data["provider"])
            stored_model = str(data["model"])
        except KeyError:
            stored_provider = stored_model = None  # legacy npz without metadata
        if stored_provider is not None:
            current_model = embed_cfg.model or ""
            if (stored_provider != embed_cfg.provider
                    or stored_model != current_model):
                print(f"[retrieval] .doc_embeddings.npz was built with "
                      f"provider={stored_provider!r} model={stored_model!r} but "
                      f"config is provider={embed_cfg.provider!r} "
                      f"model={current_model!r} — using BM25 only. "
                      f"Re-run build_embeddings.py.", file=sys.stderr)
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


def search_bm25_ranked(query: str, bm25, docs,
                       doc_type=None) -> tuple[list[int], dict[int, float]]:
    """Full BM25 ranking (all docs with score > 0), most-relevant first.
    Returns (ordered_indices, {idx: bm25_score})."""
    tokens = tokenize(query)
    if not tokens:
        return [], {}
    scores = bm25.get_scores(tokens)
    order = [(scores[i], i) for i in range(len(docs))]
    if doc_type:
        order = [(s, i) for s, i in order if docs[i]['type'] == doc_type]
    order.sort(key=lambda x: -x[0])
    ranked = [(s, i) for s, i in order if s > 0]
    return [i for _, i in ranked], {i: float(s) for s, i in ranked}


def dense_ranked(query: str, embedder, doc_emb, docs,
                 doc_type=None) -> tuple[list[int], dict[int, float]]:
    """Embed the query, cosine-rank docs.

    Returns (ordered_indices_best_first, {idx: cosine}). Returns ([], {}) if the
    embedder is unavailable, the call fails, or the query/doc vector dimensions
    don't line up — the dimension guard is the last line of defense behind the
    config-staleness check so a mismatched embed model degrades to BM25 instead
    of raising in the matmul.
    """
    if embedder is None or doc_emb is None:
        return [], {}
    q = embedder.embed([query])
    if q is None or len(q) == 0:
        return [], {}
    q = np.asarray(q, dtype=np.float32)[0]
    if q.shape[0] != doc_emb.shape[1]:
        print(f"[retrieval] query embedding dim {q.shape[0]} != doc embedding "
              f"dim {doc_emb.shape[1]} (stale/mismatched embed model) — "
              f"using BM25 only. Re-run build_embeddings.py.", file=sys.stderr)
        return [], {}
    # doc_emb rows are normalized; normalize the query too -> dot == cosine.
    qn = np.linalg.norm(q)
    if qn:
        q = q / qn
    sims = doc_emb @ q
    order = [int(i) for i in np.argsort(-sims)]
    if doc_type:
        order = [i for i in order if docs[i]['type'] == doc_type]
    score_map = {i: float(sims[i]) for i in order}
    return order, score_map


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


def apply_rerank(query: str, reranker, candidates: list[int], docs,
                 top_n: int) -> tuple[list[int], dict[int, float]] | None:
    """Rerank the top `top_n` candidate doc indices with the reranker.

    Docs are presented to the cross-encoder as natural prose (title + summary)
    via build_rerank_text — feeding the keyword-engineered build_embed_text blob
    to a prose-trained reranker degrades it.

    Returns (reordered_full_list, {idx: rerank_score for the reranked head}) —
    the reranked head followed by the untouched tail — or None if the reranker
    is unavailable / fails (caller keeps fused order).
    """
    if reranker is None or not candidates or top_n <= 0:
        return None
    head = candidates[:top_n]
    tail = candidates[top_n:]
    doc_texts = [build_rerank_text(docs[i]) for i in head]
    scores = reranker.rerank(query, doc_texts)
    if scores is None or len(scores) != len(head):
        return None
    paired = sorted(zip(scores, head), key=lambda x: -x[0])
    reordered = [i for _, i in paired]
    score_map = {i: float(s) for s, i in paired}
    return reordered + tail, score_map


def search(query: str, bm25, docs, top_k=10, doc_type=None,
           config=None, embedder=None, reranker=None, doc_emb=None,
           force_bm25=False) -> list:
    """Primary retrieval pipeline.

    Returns [(rank_score, doc_index, source, raw_score), ...] where
      * source ∈ {'bm25','dense','hybrid','rerank'} — the stage that produced
        THIS row's score (per-row: a reranked head row is 'rerank' while any
        tail beyond the rerank window keeps its fusion source);
      * rank_score is a synthetic, display-only value in (0, 1] that is strictly
        monotonic with the final rank (good for a stable on-screen number);
      * raw_score is the REAL stage-specific score for that doc (BM25 score /
        cosine similarity / RRF score / reranker score). Scales differ by source,
        so `source` tells a programmatic consumer how to interpret raw_score.
    Honors config.mode and degrades gracefully at every stage.
    """
    if config is None:
        config = load_config()

    bm25_list, bm25_scores = search_bm25_ranked(query, bm25, docs, doc_type=doc_type)

    # Decide effective mode.
    mode = "bm25" if force_bm25 else config.mode
    embedder_ok = (not force_bm25 and embedder is not None
                   and embedder.available() and doc_emb is not None)

    dense_list: list[int] = []
    dense_scores: dict[int, float] = {}
    if mode in ("auto", "dense", "hybrid") and embedder_ok:
        dense_list, dense_scores = dense_ranked(query, embedder, doc_emb, docs,
                                                doc_type=doc_type)

    # Resolve auto -> concrete mode based on what's actually available.
    if mode == "auto":
        mode = "hybrid" if dense_list else "bm25"

    if mode == "dense":
        if dense_list:
            ordered = dense_list
            raw_map = dense_scores
            source = "dense"
        else:  # dense requested but unavailable -> degrade
            print("[retrieval] dense mode requested but embeddings unavailable "
                  "— falling back to BM25.", file=sys.stderr)
            ordered = bm25_list
            raw_map = bm25_scores
            source = "bm25"
    elif mode == "hybrid" and dense_list:
        fused = reciprocal_rank_fusion([bm25_list, dense_list])
        ordered = [i for _, i in fused]
        raw_map = {i: s for s, i in fused}
        source = "hybrid"
    else:  # bm25 (explicit, or hybrid with no dense available)
        ordered = bm25_list
        raw_map = bm25_scores
        source = "bm25"

    # Per-row source so each row's `source` correctly labels the scale of its
    # `raw_score`. Pre-rerank every row shares the same stage source.
    src_by_idx = {idx: source for idx in ordered}

    # Optional rerank pass over the top-N candidates. Only the reranked HEAD
    # gets reranker scores/source; a tail beyond top_n (possible when
    # top_k > KB_RERANK_TOP_N) keeps its true BM25/dense/hybrid score+source
    # rather than being mislabeled "rerank".
    if (not force_bm25 and reranker is not None and reranker.available()
            and ordered):
        reranked = apply_rerank(query, reranker, ordered, docs,
                                config.rerank.top_n)
        if reranked is not None:
            ordered, rerank_scores = reranked
            raw_map = {**raw_map, **rerank_scores}
            for idx in rerank_scores:
                src_by_idx[idx] = "rerank"

    # rank_score: synthetic, strictly descending, for a clean display number.
    # raw_score + source: the real stage-specific score and its origin (per row).
    n = len(ordered[:top_k])
    results = []
    for rank, idx in enumerate(ordered[:top_k]):
        rank_score = (n - rank) / n if n else 0.0
        results.append((rank_score, idx, src_by_idx.get(idx, "bm25"),
                        raw_map.get(idx, 0.0)))
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
        score, idx, source = item[0], item[1], (item[2] if len(item) > 2 else 'bm25')

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

    similarity = load_similarity(docs)

    doc_type = None
    if args.type:
        doc_type = args.type.rstrip('s')

    # Wire up the optional dense + rerank stages from env config.
    config = load_config()
    embedder = get_embedder(config.embed)
    reranker = get_reranker(config.rerank)
    doc_emb = load_doc_embeddings(docs, embed_cfg=config.embed)

    results = search(args.query, bm25, docs, top_k=args.top, doc_type=doc_type,
                     config=config, embedder=embedder, reranker=reranker,
                     doc_emb=doc_emb, force_bm25=args.bm25_only)

    if args.json:
        out = []
        for item in results:
            rank_score, idx, source = item[0], item[1], item[2]
            raw_score = item[3] if len(item) > 3 else rank_score
            doc = docs[idx]
            entry = {
                # Real, stage-specific relevance score (interpret per `source`:
                # BM25 score / cosine / RRF score / reranker score).
                "score": round(raw_score, 6),
                # Synthetic 0..1 value strictly monotonic with rank (display aid).
                "rank_score": round(rank_score, 4),
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
