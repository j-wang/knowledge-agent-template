#!/usr/bin/env python3
"""
Shared document loading + text-representation helpers for the Knowledge Base.

SINGLE SOURCE OF TRUTH. Both search.py (query time) and build_embeddings.py
(build time) import from here so a document's parsed fields and its
embed/TF-IDF/rerank text representations are byte-identical on both sides.
Previously these were duplicated and had silently drifted (different summary
truncation, keyword caps, and a missing `\\Z` in the summary regex), which
corrupted the build-time-vs-query-time match. Keep all of it here.

No third-party imports at module load: numpy / scikit-learn are imported lazily
inside the functions that need them, so importing this module stays light.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
CONCEPTS_DIR = SCRIPT_DIR / "extracted" / "concepts" / "docs"
THESES_DIR = SCRIPT_DIR / "extracted" / "concepts" / "theses"

# Canonical truncation. One value, used identically at build and query time.
SUMMARY_MAX_CHARS = 400
KEYWORDS_MAX_CHARS = None  # uncapped — keywords drive BM25 recall (see CLAUDE.md)
TFIDF_CONTENT_CHARS = 500  # body prefix folded into the TF-IDF representation


# ---------------------------------------------------------------------------
# Frontmatter + section extraction
# ---------------------------------------------------------------------------

def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter fields (id, title, category, depth, type,
    analytical_angle, source, tags)."""
    fm: dict = {}
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return fm
    raw = match.group(1)
    for key in ["id", "title", "category", "depth", "type",
                "analytical_angle", "source"]:
        m = re.search(rf'^{key}:\s*(.+)', raw, re.MULTILINE)
        if m:
            fm[key] = m.group(1).strip().strip('"\'')
    tags_match = re.search(r'tags:\n((?:\s+-\s+.+\n)+)', raw)
    if tags_match:
        fm["tags"] = [t.strip().strip('"\'') for t in
                      re.findall(r'^\s+-\s+(.+)', tags_match.group(1),
                                 re.MULTILINE)]
    return fm


def extract_section(content: str, header: str, max_chars: int | None = None) -> str:
    """Pull a `## {header}` section body. `header` may be a regex alternation.

    Matches up to the next `## ` heading OR end-of-document (`\\Z`) — the `\\Z`
    matters so a section that is the file's last section is not dropped. When
    `max_chars` is set, truncate on a word boundary and append an ellipsis.
    """
    m = re.search(rf'## {header}\n(.+?)(?:\n##|\Z)', content, re.DOTALL)
    if not m:
        return ""
    text = m.group(1).strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rsplit(' ', 1)[0] + "..."
    return text


def extract_summary(content: str) -> str:
    """The Summary / Thesis Statement section (capped at SUMMARY_MAX_CHARS)."""
    return extract_section(content, "(?:Summary|Thesis Statement)",
                           SUMMARY_MAX_CHARS)


def extract_keywords(content: str) -> str:
    """The Keywords section (uncapped)."""
    return extract_section(content, "Keywords", KEYWORDS_MAX_CHARS)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_docs(concepts_dir: Path = CONCEPTS_DIR,
              theses_dir: Path = THESES_DIR) -> list[dict]:
    """Load all concept and thesis docs with metadata and full content.

    Deterministic ordering (concepts then theses, each filename-sorted) so the
    row order matches across the BM25 index, the dense vectors, and the
    similarity matrix — that alignment is what makes the stale-embeddings check
    in search.py reliable.
    """
    docs = []
    for doc_type, directory in [("concept", concepts_dir), ("thesis", theses_dir)]:
        if not directory.exists():
            continue
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith('.md') or fname.startswith('_'):
                continue
            fpath = directory / fname
            content = fpath.read_text()
            fm = parse_frontmatter(content)
            title = fm.get("title", fname.replace('.md', '').replace('-', ' '))
            docs.append({
                "path": str(fpath),
                "relative_path": f"{doc_type}s/{fname}",
                "type": doc_type,
                "id": fm.get("id", fname.replace('.md', '')),
                "title": title,
                "category": fm.get("category", fm.get("analytical_angle", "")),
                "source": fm.get("source", ""),
                "tags": fm.get("tags", []),
                "summary": extract_summary(content),
                "depth": fm.get("depth", ""),
                "keywords": extract_keywords(content),
                "content": content,
            })
    return docs


# ---------------------------------------------------------------------------
# Text representations (one definition each, shared build/query side)
# ---------------------------------------------------------------------------

def build_embed_text(doc: dict) -> str:
    """Text representation of a doc for DENSE embedding and for the field that
    the embedder/TF-IDF score against. Title (3x by repetition), tags, summary,
    keywords. The QUERY is embedded raw (no field engineering) at search time.
    """
    parts = [doc["title"], doc["title"], doc["title"]]
    if doc.get("tags"):
        parts.append("Topics: " + ", ".join(doc["tags"]))
    if doc.get("summary"):
        parts.append(doc["summary"])
    if doc.get("keywords"):
        parts.append(doc["keywords"])
    return "\n".join(parts)


def build_rerank_text(doc: dict) -> str:
    """Natural-prose text representation for a cross-encoder reranker.

    Cross-encoders are trained on prose, so we feed the title + the summary
    (real sentences) rather than the keyword-engineered `build_embed_text` blob.
    Falls back to keywords only if a doc has no summary.
    """
    parts = [doc.get("title", "")]
    if doc.get("summary"):
        parts.append(doc["summary"])
    elif doc.get("keywords"):
        parts.append(doc["keywords"])
    return "\n".join(p for p in parts if p)


def build_tfidf_text(doc: dict) -> str:
    """Text representation for local TF-IDF doc-to-doc similarity (`--related`).

    Title (3x), tags (2x), summary, keywords, plus a body prefix for broader
    vocabulary. Kept here so the `.similarity.npy` matrix is identical whether it
    was built by `search.py --rebuild` or by `build_embeddings.py`.
    """
    parts = [doc["title"], doc["title"], doc["title"]]
    if doc.get("tags"):
        parts.append(" ".join(doc["tags"]))
        parts.append(" ".join(doc["tags"]))  # tags double-weighted
    if doc.get("summary"):
        parts.append(doc["summary"])
    if doc.get("keywords"):
        parts.append(doc["keywords"])
    if doc.get("content"):
        parts.append(doc["content"][:TFIDF_CONTENT_CHARS])
    return "\n".join(parts)


def build_tfidf_similarity(docs: list[dict], verbose: bool = False):
    """Build TF-IDF vectors and return the (N, N) cosine similarity matrix.

    Fully local (scikit-learn only, imported lazily). Used solely for the
    `--related` display feature.
    """
    import sys
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not docs:
        return np.zeros((0, 0), dtype=np.float32)

    texts = [build_tfidf_text(doc) for doc in docs]
    n_docs = len(docs)
    min_df = min(2, n_docs) if n_docs > 0 else 1
    max_df = max(0.8, (n_docs - 0.5) / n_docs) if n_docs > 1 else 1.0
    vectorizer = TfidfVectorizer(
        max_features=10000, min_df=min_df, max_df=max_df,
        ngram_range=(1, 2), sublinear_tf=True, stop_words='english',
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    if verbose:
        print(f"  TF-IDF vocabulary: {len(vectorizer.vocabulary_)} terms, "
              f"matrix: {tfidf_matrix.shape}", file=sys.stderr)
    return cosine_similarity(tfidf_matrix)
